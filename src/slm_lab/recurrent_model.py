"""Experimental shared-depth multimodal decoder with canonical unique weights.

This is an architecture experiment, not a claim that extra passes improve an
untrained loop. The current training model and checkpoint schema stay separate.
"""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .model import TinyMultimodalLM


@dataclass(frozen=True)
class RecurrentCache:
    layers: tuple
    image_tokens: int
    text_length: int
    owner: int
    recurrences: int
    core: tuple

    @property
    def stored_tokens(self): return self.image_tokens + self.text_length

    @property
    def bytes(self): return sum(t.numel() * t.element_size() for pair in self.layers for t in pair)


class RecurrentMultimodalLM(TinyMultimodalLM):
    def __init__(self, config, core_start=None, core_end=None, max_recurrences=4, recurrences=1):
        super().__init__(config)
        self.core_start = config.num_layers // 4 if core_start is None else core_start
        self.core_end = config.num_layers - config.num_layers // 4 if core_end is None else core_end
        if (type(self.core_start) is not int or type(self.core_end) is not int
                or not 0 <= self.core_start < self.core_end <= config.num_layers):
            raise ValueError('Recurrent core must select a nonempty contiguous original layer range')
        if type(max_recurrences) is not int or not 1 <= max_recurrences <= 16:
            raise ValueError('Recurrent depth budget must be between one and sixteen')
        self.max_recurrences = max_recurrences
        # A zero extra-pass gate preserves the own base exactly while allowing
        # learning of each additional core pass from original supervision.
        self.recurrence_gates = nn.Parameter(torch.zeros(max_recurrences - 1))
        self.recurrences = self.checked_depth(recurrences)

    def checked_depth(self, value):
        if type(value) is not int or not 1 <= value <= self.max_recurrences:
            raise ValueError('Requested recurrent depth exceeds the explicit compute budget')
        return value

    def load_owned_base(self, state):
        keys = set(self.state_dict()) - {'recurrence_gates'}
        if set(state) != keys: raise ValueError('Own base tensor inventory differs from the recurrent architecture')
        result = self.load_state_dict(state, strict=False)
        if result.missing_keys != ['recurrence_gates'] or result.unexpected_keys:
            raise ValueError('Own base transfer did not preserve every canonical tensor')
        with torch.no_grad(): self.recurrence_gates.zero_()

    def execution_layers(self, recurrences=None):
        depth = self.checked_depth(self.recurrences if recurrences is None else recurrences)
        return self.config.num_layers + (depth - 1) * (self.core_end - self.core_start)

    def _inputs(self, input_ids, pixel_values, attention_mask=None):
        if input_ids.ndim != 2 or not 1 <= input_ids.shape[1] <= self.config.max_text_length:
            raise ValueError('Text must be [batch,length] within max_text_length')
        x = self.embedding(input_ids); prefix = 0
        if pixel_values is not None:
            if pixel_values.shape[0] != input_ids.shape[0]: raise ValueError('Text and image batch sizes must match')
            visual = self.vision(pixel_values, self.gradient_checkpointing); prefix = visual.shape[1]
            x = torch.cat((visual, x), dim=1)
        if attention_mask is not None:
            if attention_mask.shape != input_ids.shape: raise ValueError('attention_mask must have the text shape')
            if bool(attention_mask.bool().all()): attention_mask = None
        mask = None
        if prefix or attention_mask is not None:
            size = x.shape[1]; mask = torch.ones(size, size, device=x.device, dtype=torch.bool).tril()
            if prefix: mask[:prefix, :prefix] = True
            if attention_mask is not None:
                valid = torch.cat((torch.ones(input_ids.shape[0], prefix, device=x.device, dtype=torch.bool),
                                   attention_mask.bool()), dim=1)
                mask = mask[None, None] & valid[:, None, None, :]
        return x, prefix, mask

    def _body(self, x, mask, causal, depth, cached=False, past=None):
        stored = []; index = 0

        def layer(position, hidden):
            nonlocal index
            block = self.blocks[position]
            if cached:
                hidden, pair = block.cached(hidden, None if past is None else past[index], mask, causal)
                stored.append(pair)
            elif self.gradient_checkpointing and self.training:
                hidden = checkpoint(block, hidden, mask, causal, use_reentrant=False)
            else: hidden = block(hidden, mask, causal)
            index += 1
            return hidden

        for position in range(self.core_start): x = layer(position, x)
        for iteration in range(depth):
            anchor = x
            for position in range(self.core_start, self.core_end): x = layer(position, x)
            if iteration:
                gate = self.recurrence_gates[iteration - 1].tanh().to(x.dtype)
                x = anchor + gate * (x - anchor)
        for position in range(self.core_end, self.config.num_layers): x = layer(position, x)
        return x, tuple(stored)

    def forward(self, input_ids, pixel_values=None, attention_mask=None, labels=None,
                return_hidden=False, return_logits=True, recurrences=None):
        if labels is not None and not return_logits: raise ValueError('Token supervision requires logits')
        depth = self.checked_depth(self.recurrences if recurrences is None else recurrences)
        x, prefix, mask = self._inputs(input_ids, pixel_values, attention_mask)
        x, _ = self._body(x, mask, mask is None, depth)
        hidden = self.norm(x[:, prefix:]); out = {'values': self.critic(hidden)}
        if return_logits: out['logits'] = self.lm_head(hidden)
        if return_hidden: out['hidden'] = hidden
        if labels is not None:
            if labels.shape != input_ids.shape: raise ValueError('labels must have the text shape')
            targets = labels[:, 1:].contiguous()
            if input_ids.shape[1] < 2 or not torch.any(targets != -100): raise ValueError('Loss requires an unmasked next-token target')
            out['loss'] = F.cross_entropy(out['logits'][:, :-1].float().reshape(-1, self.config.vocab_size),
                                          targets.reshape(-1), ignore_index=-100)
        return out

    @torch.no_grad()
    def prefill(self, input_ids, pixel_values=None, recurrences=None):
        if self.training: raise ValueError('Cached inference requires eval mode')
        depth = self.checked_depth(self.recurrences if recurrences is None else recurrences)
        x, prefix, mask = self._inputs(input_ids, pixel_values)
        x, layers = self._body(x, mask, mask is None, depth, cached=True)
        hidden = self.norm(x[:, -1:])
        return {'logits': self.lm_head(hidden), 'values': self.critic(hidden),
                'cache': RecurrentCache(layers, prefix, input_ids.shape[1], id(self), depth, (self.core_start, self.core_end))}

    @torch.no_grad()
    def decode(self, input_ids, cache):
        if self.training: raise ValueError('Cached inference requires eval mode')
        if (not isinstance(cache, RecurrentCache) or cache.owner != id(self)
                or cache.core != (self.core_start, self.core_end)):
            raise ValueError('Recurrent cache must belong to this exact model/core')
        depth = self.checked_depth(cache.recurrences)
        if (input_ids.ndim != 2 or input_ids.shape[1] < 1
                or cache.text_length + input_ids.shape[1] > self.config.max_text_length):
            raise ValueError('Cached text length must fit max_text_length')
        if (len(cache.layers) != self.execution_layers(depth)
                or any(pair[0].shape[:2] != (input_ids.shape[0], self.config.num_kv_heads)
                       or pair[0].shape[2] != cache.stored_tokens or pair[1].shape != pair[0].shape for pair in cache.layers)):
            raise ValueError('Recurrent cache stage/batch/head/length inventory differs')
        x = self.embedding(input_ids); mask = None
        if input_ids.shape[1] > 1:
            queries = torch.arange(cache.stored_tokens, cache.stored_tokens + input_ids.shape[1], device=x.device)
            keys = torch.arange(cache.stored_tokens + input_ids.shape[1], device=x.device)
            mask = keys[None, :] <= queries[:, None]
        x, layers = self._body(x, mask, False, depth, cached=True, past=cache.layers)
        hidden = self.norm(x)
        return {'logits': self.lm_head(hidden), 'values': self.critic(hidden),
                'cache': RecurrentCache(layers, cache.image_tokens, cache.text_length + input_ids.shape[1],
                                         id(self), depth, cache.core)}
