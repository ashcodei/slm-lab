"""Own shared backbone with explicit code-only FFN and attention adaptation.

LoRA (Hu et al., https://arxiv.org/abs/2106.09685) is established prior work.
The 1728-wide residual / rank96 allocation is an unproven local hypothesis.
It has exactly the same parameter budget as the earlier width2048 residuals.
No original backbone module or parameter is replaced or mutated.
"""
import math

from torch import nn, no_grad

from .code_residual_branch_v1 import CodeResidual, RoutedCodeResidual, _CodeView
from .model import Attention, Block, TinyMultimodalLM


class LowRankUpdate(nn.Module):
    def __init__(self, original, rank, scale):
        super().__init__()
        self.scale = scale
        options = dict(device=original.weight.device, dtype=original.weight.dtype, bias=False)
        self.down = nn.Linear(original.in_features, rank, **options)
        self.up = nn.Linear(rank, original.out_features, **options)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden):
        return self.up(self.down(hidden)) * self.scale


class LayerAdapters(nn.Module):
    def __init__(self, original, hidden, width, rank, scale):
        super().__init__()
        self.ffn = CodeResidual(hidden, width, device=original.gate.weight.device, dtype=original.gate.weight.dtype)
        self.attention = nn.ModuleDict({name: LowRankUpdate(getattr(original.attn, name), rank, scale)
            for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj')})


class _Projection(nn.Module):
    def __init__(self, original, update):
        super().__init__()
        self.original, self.update = original, update

    def forward(self, hidden):
        return self.original(hidden) + self.update(hidden)


class _Attention(Attention):
    """Reuse original full/cached RoPE/GQA code with code-only projections."""
    def __init__(self, original, updates):
        nn.Module.__init__(self)
        for name in ('heads', 'kv_heads', 'head_dim', 'rope_theta'):
            setattr(self, name, getattr(original, name))
        for name, update in updates.items():
            setattr(self, name, _Projection(getattr(original, name), update))


class _Block(Block):
    def __init__(self, original, adapters):
        nn.Module.__init__(self)
        self.original, self.adapters = original, adapters
        for name in ('attn_norm', 'ffn_norm', 'gate', 'up', 'down'):
            setattr(self, name, getattr(original, name))
        self.attn = _Attention(original.attn, adapters.attention)

    def forward(self, hidden, mask=None, causal=False):
        return self.adapters.ffn(super().forward(hidden, mask, causal))

    @no_grad()
    def cached(self, hidden, past=None, mask=None, causal=False):
        hidden, stored = super().cached(hidden, past, mask, causal)
        return self.adapters.ffn(hidden), stored


class _View(_CodeView):
    def __init__(self, base, adapters):
        nn.Module.__init__(self)
        self.config = base.config
        for name in ('embedding', 'vision', 'norm', 'lm_head', 'value_head'):
            setattr(self, name, getattr(base, name))
        self.blocks = nn.ModuleList(_Block(block, added) for block, added in zip(base.blocks, adapters))
        self.gradient_checkpointing = False


class RoutedCodeAttention(RoutedCodeResidual):
    def __init__(self, base, width=1728, rank=96, scale=1., parameter_limit=10**9):
        nn.Module.__init__(self)
        if (type(base) is not TinyMultimodalLM or type(width) is not int or width < 1
                or type(rank) is not int or rank < 1 or type(scale) not in (float, int)
                or not math.isfinite(scale) or scale <= 0 or type(parameter_limit) is not int
                or not 1 <= parameter_limit <= 10**9):
            raise ValueError('Exact own backbone and explicit positive finite adapter configuration required')
        hidden = base.config.hidden_size
        attention_added = sum(rank * (getattr(block.attn, name).in_features + getattr(block.attn, name).out_features)
            for block in base.blocks for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj'))
        added = 2 * hidden * width * base.config.num_layers + attention_added
        if sum(p.numel() for p in base.parameters()) + added >= parameter_limit:
            raise ValueError('All unique base and adapter parameters must fit the total model limit')
        self.base, self.width, self.rank, self.scale, self.parameter_limit = base, width, rank, float(scale), parameter_limit
        for parameter in base.parameters():
            parameter.requires_grad_(False)
        self.residuals = nn.ModuleList(LayerAdapters(block, hidden, width, rank, self.scale) for block in base.blocks)
        self.code = _View(base, self.residuals)
        self.assert_partition()

    def architecture(self):
        return dict(format='own-code-FFN-attention-allocation-v1', residual_width=self.width,
            attention_rank=self.rank, attention_scale=self.scale,
            attention_projections=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
            placement='code-only low-rank projection updates plus post-block SiLU residual')

    def assert_partition(self):
        if len(self.residuals) != len(self.base.blocks) or len(self.code.blocks) != len(self.base.blocks):
            raise ValueError('Every original decoder block needs its code-only adapters')
        for block, original, added in zip(self.code.blocks, self.base.blocks, self.residuals):
            if block.original is not original or block.adapters is not added:
                raise ValueError('Code route must share the exact original frozen block')
            for name in ('attn_norm', 'ffn_norm', 'gate', 'up', 'down'):
                if getattr(block, name) is not getattr(original, name):
                    raise ValueError('Every original non-attention block component must remain shared')
            for name, update in added.attention.items():
                projection = getattr(block.attn, name)
                if (projection.original is not getattr(original.attn, name) or projection.update is not update
                        or update.scale != self.scale or update.down.out_features != self.rank):
                    raise ValueError('Every code attention projection must retain its exact frozen original and adapter')
        for name in ('embedding', 'vision', 'norm', 'lm_head', 'value_head'):
            if getattr(self.code, name) is not getattr(self.base, name):
                raise ValueError('Every original global component must remain shared')
        if (any(p.requires_grad for p in self.base.parameters())
                or not all(p.requires_grad for p in self.residuals.parameters())
                or sum(p.numel() for p in self.parameters()) >= self.parameter_limit):
            raise ValueError('Frozen original backbone and strictly bounded trainable adapters required')


def restore_attention_training_state(model, optimizer, bundle, **kwargs):
    from .code_residual_checkpoint_v1 import restore_training_state
    if bundle.get('architecture') != model.architecture():
        raise ValueError('Exact attention architecture/rank/scale required before restoring any state')
    return restore_training_state(model, optimizer, bundle, **kwargs)
