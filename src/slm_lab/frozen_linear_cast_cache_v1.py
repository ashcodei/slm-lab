"""Derived BF16 storage for the own actor's detached, immutable FP32 linears.

This changes runtime storage only. Original registered values, optimizer groups,
exports and embedding lookups are untouched. CUDA use requires separate actual
GPU equivalence/capacity evidence; CPU-mechanics mode cannot supply it.
"""
from types import MethodType
import weakref

import torch
from torch.nn import functional as F

from .own_dense_actor_v1 import _FrozenHead, _FrozenLinear


def signature(parameter):
    return (id(parameter), parameter._version, parameter.data_ptr(), parameter.dtype,
        parameter.device, tuple(parameter.shape), tuple(parameter.stride()), parameter.requires_grad)


def parameters(module):
    layer = module.original if type(module) is _FrozenLinear else module
    return (layer.weight, getattr(layer, 'bias', None))


class FrozenLinearCastCache:
    def __init__(self, *, device, purpose='GPU-runtime'):
        self.device = torch.device(device)
        if ((purpose == 'GPU-runtime' and (self.device.type != 'cuda' or self.device.index is None))
                or (purpose == 'CPU-mechanics' and self.device != torch.device('cpu'))
                or purpose not in {'GPU-runtime', 'CPU-mechanics'}):
            raise ValueError('Explicit CUDA index for runtime, or explicit CPU-mechanics purpose required')
        self.purpose = purpose
        self._entries = {}
        self._patched = []
        self.conversions = 0
        self.hits = 0

    def _check(self, current, expected):
        if len(current) != len(expected) or any(
                (p is None) != (s is None) or (p is not None and signature(p) != s)
                for p, s in zip(current, expected)):
            raise ValueError('Frozen source binding or value changed; close the cache before source mutation')

    def _value(self, parameter):
        key = id(parameter)
        prior = self._entries.get(key)
        if prior is not None:
            if prior[0]() is not parameter or prior[1] != signature(parameter):
                raise ValueError('Frozen source changed while its derived cast was resident')
            self.hits += 1
            return prior[2]
        result = parameter.detach().to(device=self.device, dtype=torch.bfloat16, copy=True)
        self._entries[key] = (weakref.ref(parameter), signature(parameter), result)
        self.conversions += 1
        return result

    def attach(self, view):
        """Patch exact detach-only wrappers, validating the entire new set first."""
        known = {id(m) for m, *_ in self._patched}
        selected = [m for m in view.modules() if type(m) in {_FrozenLinear, _FrozenHead} and id(m) not in known]
        prepared = []
        for module in selected:
            original = module.forward
            if getattr(original, '__func__', None) is not type(module).forward:
                raise ValueError('An existing forward patch cannot be overwritten')
            values = parameters(module)
            if any(p is not None and (p.device != self.device or p.dtype != torch.float32) for p in values):
                raise ValueError('Only exact-device original FP32 frozen parameters may be cached')
            expected = tuple(None if p is None else signature(p) for p in values)
            prepared.append((module, original, expected))
        for module, original, expected in prepared:
            cache = self
            def forward(current, hidden, _original=original, _expected=expected):
                values = parameters(current)
                cache._check(values, _expected)
                if (hidden.device != cache.device or not torch.is_autocast_enabled(cache.device.type)
                        or torch.get_autocast_dtype(cache.device.type) != torch.bfloat16):
                    return _original(hidden)
                weight, bias = values
                return F.linear(hidden, cache._value(weight), None if bias is None else cache._value(bias))
            patched = MethodType(forward, module)
            module.forward = patched
            self._patched.append((module, original, patched, expected))
        return self

    def assert_current(self):
        for module, _, patched, expected in self._patched:
            if module.forward is not patched:
                raise ValueError('The cache no longer owns an attached forward')
            self._check(parameters(module), expected)
        if any(ref() is None or signature(ref()) != state for ref, state, _ in self._entries.values()):
            raise ValueError('Frozen source changed while its derived cast was resident')

    def inventory(self):
        return dict(purpose=self.purpose, device=str(self.device),
            additional_derived_values=sum(t.numel() for _, _, t in self._entries.values()),
            additional_derived_bytes=sum(t.numel() * t.element_size() for _, _, t in self._entries.values()),
            cached_original_parameter_tensors=len(self._entries), actual_conversions=self.conversions,
            actual_cache_hits=self.hits, patched_detach_only_wrappers=len(self._patched),
            added_learned_parameters=0, cache_excluded_from_state_dict_and_export=True,
            original_parameter_trainability_metadata_unchanged=True,
            actual_full_size_GPU_equivalence_and_speed_not_proved_by_this_inventory=True)

    def close(self):
        if any(module.forward is not patched for module, _, patched, _ in self._patched):
            raise ValueError('Refuse to overwrite another owner\'s forward during cleanup')
        for module, original, _, _ in self._patched:
            module.forward = original
        self._patched.clear()
        self._entries.clear()
