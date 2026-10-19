"""Initialize alignment on our trained code route without copying its weights.

V2's aligned language view starts from the original general route and does not
use the learned code factors or code vocabulary table. This separate experiment
starts at the existing own code computation, freezes it without changing the
factory-required trainability metadata, and trains the same new rank16 factors.
It has the same complete parameter count. Structural retention is not a coding
improvement, and this is not a promoted or externally pretrained model.
"""
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from .own_aligned_assistant_v1 import _AlignedView
from .own_aligned_assistant_v2 import OwnAlignedAssistantV2, FORMAT as V2_FORMAT
from .own_code_delta_v1 import _DeltaBlock, _TiedHead

FORMAT = 'own-single-backbone-code-initialized-alignment-expanded-word-v3'


class _FrozenEmbedding(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.original = original; self.weight = original.weight
        self.num_embeddings, self.embedding_dim = original.num_embeddings, original.embedding_dim

    def forward(self, ids):
        return F.embedding(ids, self.weight.detach())


class _FrozenTiedHead(_TiedHead):
    def forward(self, hidden):
        return F.linear(hidden, self.weight.detach())


class _FrozenCodeLinear(nn.Module):
    def __init__(self, original):
        super().__init__(); self.original = original

    def forward(self, hidden):
        # Keep derivatives with respect to hidden. no_grad() on this whole
        # computation would sever the earlier new adapters' gradients.
        layer = self.original
        return layer.original(hidden) + F.linear(F.linear(hidden, layer.factors.down.detach()), layer.factors.up.detach())


def _frozen_block(block):
    attention = block.attn
    attn = SimpleNamespace(heads=attention.heads, kv_heads=attention.kv_heads,
        head_dim=attention.head_dim, rope_theta=attention.rope_theta,
        **{name:_FrozenCodeLinear(getattr(attention,name)) for name in ('q_proj','k_proj','v_proj','o_proj')})
    return SimpleNamespace(attn=attn,
        **{name:_FrozenCodeLinear(getattr(block,name)) for name in ('gate','up','down')})


class _AlignedCodeView(_AlignedView):
    def __init__(self, original, parts):
        super().__init__(original, parts)
        code = original.language.code
        self.embedding = _FrozenEmbedding(code.embedding)
        self.lm_head = _FrozenTiedHead(code.embedding)
        self.blocks = nn.ModuleList(_DeltaBlock(_frozen_block(block),
            parts['language_linears'][i*7:(i+1)*7], parts['language_norms'][i*2:i*2+2])
            for i,block in enumerate(code.blocks))


class OwnAlignedAssistantV3(OwnAlignedAssistantV2):
    def __init__(self, original, **kwargs):
        self._code_reference_ready = False
        super().__init__(original, **kwargs)
        # These existing new norm tensors acquire the original own code values;
        # there is no additional stored reference adapter or embedding.
        self.parts['language_norms'].load_state_dict(original.language.parts['norms'].state_dict(), strict=True)
        self.aligned = _AlignedCodeView(original, self.parts)
        self.settings = {**self.settings, 'language_reference_route':'original_own_code'}
        self._code_reference_ready = True
        self.assert_word_contract()

    def assert_shared(self):
        if not getattr(self, '_code_reference_ready', False):
            return super().assert_shared()
        self.original.assert_shared()
        old = {id(p) for p in self.original.parameters()}; added = {id(p) for p in self.parts.parameters()}
        code = self.original.language.code; base = self.original.language.base; counts = self.counts()
        if (type(self.aligned) is not _AlignedCodeView or self.aligned.original is not self.original
                or self.aligned.extensions is not self.parts or self.aligned.embedding.original is not code.embedding
                or self.aligned.embedding.weight is not code.embedding.weight or self.aligned.lm_head.weight is not code.embedding.weight
                or self.aligned.vision.original is not base.vision or old & added
                or any(not p.requires_grad for p in self.parts.parameters())
                or counts['total_unique'] != self.expected_budget['total_unique']
                or counts['total_unique'] != self.original.counts()['total_unique'] + counts['extension_unique']
                or counts['total_unique'] >= self.settings['parameter_limit']
                or sum(p.numel() for p in self.aligned.parameters()) != counts['total_unique']
                or self.settings['language_reference_route'] != 'original_own_code'):
            raise ValueError('Exactly one existing own backbone/code table and the complete bounded code-initialized extensions required')

    def export(self, sources):
        return {**super().export(sources), 'format':FORMAT}

    def validate_export(self, payload, *, expected_sources):
        if payload.get('format') != FORMAT:
            raise ValueError('Exact code-initialized v3 computation and artifact format required')
        super().validate_export({**payload, 'format':V2_FORMAT}, expected_sources=expected_sources)
