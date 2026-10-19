"""Training-only text reference sharing the joint model's unchanged language.

Visual/word updates do not affect text-only logits. Language values and native
computation remain bound; image inputs are rejected. The reference adds no
weights and supplies no external teacher. Construct a fresh binding after
placing or restoring the policy graph. This is not a full DPO/RLHF stage.
"""
import copy
from dataclasses import asdict

import torch
from torch import nn

from .own_dense_growth_v1 import OwnDenseAssistant
from .parallel_function_repair_stage_v1 import value_digest
from .whole_human_sft_evidence_v1 import tensor_digest

FORMAT = 'own-shared-frozen-text-only-reference-with-mutable-visual-branches-v1'


class FrozenOwnTextReferenceV1(nn.Module):
    def __init__(self, base, *, sources, tokenizer_sha256):
        super().__init__()
        if (type(base) is not OwnDenseAssistant or base.network.training or type(sources) is not dict or not sources
                or type(tokenizer_sha256) is not str or len(tokenizer_sha256) != 64
                or any(c not in '0123456789abcdef' for c in tokenizer_sha256)
                or any(type(k) is not str or not k or type(v) is not str or len(v) != 64
                    or any(c not in '0123456789abcdef' for c in v) for k, v in sources.items())):
            raise ValueError('An actual own eval-mode native text computation, tokenizer and complete source bindings are required')
        base.assert_inventory(); self.base = base; self.training = False
        self._binding = dict(format=FORMAT, sources=copy.deepcopy(sources), tokenizer_sha256=tokenizer_sha256,
            configuration=asdict(base.config), frozen_language_sha256=tensor_digest(base.language_named_parameters()),
            reference_is_text_only=True, retained_extra_neural_parameters=0)
        self._identity = value_digest(self._binding)
        self._signature = self.signature()

    def signature(self):
        return tuple((n, id(p), p.data_ptr(), p._version, p.requires_grad, str(p.dtype), str(p.device))
            for n, p in self.base.language_named_parameters())

    @property
    def reference_binding(self): return copy.deepcopy(self._binding)

    def assert_bound(self, *, full=False):
        if (value_digest(self._binding) != self._identity or asdict(self.base.config) != self._binding['configuration']
                or self.base.network.training or self.signature() != self._signature
                or full and tensor_digest(self.base.language_named_parameters()) != self._binding['frozen_language_sha256']):
            raise ValueError('The exact original text-reference values, computation, source map and parameter metadata must remain unchanged')

    def train(self, mode=True):
        if mode is not False: raise ValueError('The shared frozen text reference cannot enter training mode')
        self.training = False
        return self

    @torch.no_grad()
    def forward(self, input_ids, *, attention_mask=None, pixel_values=None):
        if pixel_values is not None: raise ValueError('The immutable reference supports text only; mutable visual inputs cannot be scored')
        self.assert_bound()
        logits = self.base.network(input_ids, attention_mask=attention_mask)['logits']
        self.assert_bound()
        if logits.requires_grad or not bool(torch.isfinite(logits).all()):
            raise ValueError('Complete reference logits must remain finite and gradient free')
        return logits
