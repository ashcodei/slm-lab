"""A frozen snapshot of our own starting actor sharing its counted backbone.

The private snapshot contains adapters/norms/critic only. Its weights never
enter the exported student. Freezing the private snapshot must not change the
shared base's original flags, values or parameter pointers.
"""
import copy

import torch
from torch import nn

from .own_dense_actor_v1 import OwnDenseActor, DensePolicyView
from .whole_human_sft_evidence_v1 import tensor_digest


def signature(parameters):
    return [(id(p), p._version, p.data_ptr(), p.requires_grad) for p in parameters]


class FrozenOwnActorReference(nn.Module):
    def __init__(self, model, *, sources):
        super().__init__()
        if (type(model) is not OwnDenseActor or not sources
                or any(not isinstance(k, str) or not isinstance(v, str) or len(v) != 64
                       or any(c not in '0123456789abcdef' for c in v) for k, v in sources.items())):
            raise ValueError('Exact own actor and immutable own starting-policy SHA bindings required')
        model.assert_shared()
        self.base = model.base
        self.parts = copy.deepcopy(model.parts)
        self.parts.requires_grad_(False)
        self.policy = DensePolicyView(self.base, self.parts).eval()
        self.actor_settings = copy.deepcopy(model.settings)
        self.actual_counts = copy.deepcopy(model.counts())
        self.sources = copy.deepcopy(sources)
        self.initial_base_signature = signature(self.base.parameters())
        self.initial_private_signature = signature(self.parts.parameters())
        self.metadata = dict(format='frozen-own-starting-actor-shared-dense-reference-v1',
            sources=copy.deepcopy(sources), settings=copy.deepcopy(model.settings),
            frozen_dense_sha256=tensor_digest(self.base.named_parameters()),
            frozen_actor_parts_sha256=tensor_digest(self.parts.named_parameters()),
            private_reference_values=sum(p.numel() for p in self.parts.parameters()),
            private_values_excluded_from_student_export=True, external_pretrained_teacher=False)
        self.assert_bound(model)

    def assert_bound(self, model):
        if (type(model) is not OwnDenseActor or model.base is not self.base
                or model.settings != self.actor_settings or model.counts() != self.actual_counts
                or signature(self.base.parameters()) != self.initial_base_signature
                or signature(self.parts.parameters()) != self.initial_private_signature
                or self.policy.base is not self.base or self.policy.parts is not self.parts
                or {id(p) for p in self.parts.parameters()} & {id(p) for p in model.parts.parameters()}
                or any(p.requires_grad or p.grad is not None for p in self.parts.parameters())):
            raise ValueError('Own reference snapshot, shared frozen base and private untrained values changed')

    def binding(self):
        return copy.deepcopy(self.metadata)

    def resident_inventory(self, model):
        self.assert_bound(model)
        unique = {id(p): p for owner in [model, self] for p in owner.parameters()}
        resident = sum(p.numel() for p in unique.values())
        extra = sum(p.numel() for p in self.parts.parameters())
        exported = model.counts()['actual_unique']
        if resident != exported + extra:
            raise ValueError('Exactly one shared base and two explicitly counted actor snapshots required')
        return dict(exported_student_values=exported, extra_training_only_own_reference_values=extra,
                    total_unique_resident_values=resident, extra_reference_excluded_from_export=True)

    @torch.no_grad()
    def forward(self, *args, **kwargs):
        return self.policy(*args, **kwargs)
