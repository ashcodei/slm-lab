"""Own-weight recurrent decoder with input injection and normalized residuals.

This adapts the retained own backbone, rather than replicating a published
checkpoint. All passes use the same geometry, and no gate can bypass a pass.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .model import RMSNorm
from .recurrent_model import RecurrentMultimodalLM


class ConditionedRecurrentLM(RecurrentMultimodalLM):
    EXTRA_PREFIXES = ('recurrence_adapter.', 'recurrence_attn_post.',
                      'recurrence_ffn_post.', 'recurrence_output_norm.')

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        del self.recurrence_gates
        count = self.core_end - self.core_start
        self.recurrence_adapter = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
        self.recurrence_attn_post = nn.ModuleList(RMSNorm(config.hidden_size, config.norm_eps) for _ in range(count))
        self.recurrence_ffn_post = nn.ModuleList(RMSNorm(config.hidden_size, config.norm_eps) for _ in range(count))
        self.recurrence_output_norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.reset_recurrence()

    @torch.no_grad()
    def reset_recurrence(self):
        width = self.config.hidden_size
        # Both halves begin active, with an identity first mix when state=input.
        self.recurrence_adapter.weight.zero_()
        identity = torch.eye(width, device=self.recurrence_adapter.weight.device,
                             dtype=self.recurrence_adapter.weight.dtype) * .5
        self.recurrence_adapter.weight[:, :width].copy_(identity)
        self.recurrence_adapter.weight[:, width:].copy_(identity)
        for module in (*self.recurrence_attn_post, *self.recurrence_ffn_post, self.recurrence_output_norm):
            module.weight.fill_(1.)

    def new_state(self):
        return {name: value for name, value in self.state_dict().items()
                if name.startswith(self.EXTRA_PREFIXES)}

    def load_owned_base(self, state):
        extra = set(self.new_state())
        if set(state) != set(self.state_dict()) - extra:
            raise ValueError('Own base tensor inventory differs from conditioned recurrence')
        result = self.load_state_dict(state, strict=False)
        if set(result.missing_keys) != extra or result.unexpected_keys:
            raise ValueError('Conditioned transfer did not preserve every own canonical tensor')
        self.reset_recurrence()

    def _core_layer(self, position, hidden, mask, causal, cached=False, past=None):
        block = self.blocks[position]; local = position - self.core_start
        normalized = block.attn_norm(hidden)
        if cached:
            attention, pair = block.attn.cached(normalized, past, mask, causal)
        else:
            attention = block.attn(normalized, mask, causal); pair = None
        hidden = self.recurrence_attn_post[local](hidden + attention)
        normalized = block.ffn_norm(hidden)
        hidden = self.recurrence_ffn_post[local](hidden + block.down(F.silu(block.gate(normalized)) * block.up(normalized)))
        return hidden, pair

    def _body(self, x, mask, causal, depth, cached=False, past=None):
        stored = []; index = 0

        def layer(position, hidden, core=False):
            nonlocal index
            if core:
                if cached:
                    hidden, pair = self._core_layer(position, hidden, mask, causal, True,
                                                   None if past is None else past[index])
                elif self.gradient_checkpointing and self.training:
                    # Bind position per invocation: recomputation occurs after
                    # the outer recurrent loop has advanced to another layer.
                    def apply(value, selected=position):
                        return self._core_layer(selected, value, mask, causal)[0]
                    hidden = checkpoint(apply, hidden, use_reentrant=False)
                else: hidden = self._core_layer(position, hidden, mask, causal)[0]
            else:
                block = self.blocks[position]
                if cached: hidden, pair = block.cached(hidden, None if past is None else past[index], mask, causal)
                elif self.gradient_checkpointing and self.training:
                    hidden = checkpoint(block, hidden, mask, causal, use_reentrant=False)
                else: hidden = block(hidden, mask, causal)
            if cached: stored.append(pair)
            index += 1
            return hidden

        for position in range(self.core_start): x = layer(position, x)
        original_input = x
        for _ in range(depth):
            x = self.recurrence_adapter(torch.cat((x, original_input), dim=-1))
            for position in range(self.core_start, self.core_end): x = layer(position, x, core=True)
            x = self.recurrence_output_norm(x)
        for position in range(self.core_end, self.config.num_layers): x = layer(position, x)
        return x, tuple(stored)

    @torch.no_grad()
    def calibrate_owned_norms(self, sequences):
        """Set new norm scales using only selected original training windows.

        Canonical backbone tensors remain exact. Calibration approximately
        matches their residual magnitudes; it does not preserve base outputs.
        """
        sums = torch.zeros(2, self.core_end - self.core_start, device=self.embedding.weight.device)
        count = 0
        for ids in sequences:
            x, _, mask = self._inputs(ids, None)
            for position, block in enumerate(self.blocks):
                x = x + block.attn(block.attn_norm(x), mask, mask is None)
                if self.core_start <= position < self.core_end:
                    sums[0, position - self.core_start] += x.float().square().mean(-1).sqrt().mean()
                normalized = block.ffn_norm(x)
                x = x + block.down(F.silu(block.gate(normalized)) * block.up(normalized))
                if self.core_start <= position < self.core_end:
                    sums[1, position - self.core_start] += x.float().square().mean(-1).sqrt().mean()
            count += 1
        if not count or not torch.isfinite(sums).all() or not (sums > 0).all():
            raise ValueError('Norm calibration requires finite original training activations')
        means = sums / count
        for index, (attention, feedforward) in enumerate(zip(self.recurrence_attn_post, self.recurrence_ffn_post)):
            attention.weight.fill_(means[0, index]); feedforward.weight.fill_(means[1, index])
        self.recurrence_output_norm.weight.fill_(means[1, -1])
        return {'original_training_windows': count, 'attention_rms': means[0].cpu().tolist(),
                'feedforward_rms': means[1].cpu().tolist()}
