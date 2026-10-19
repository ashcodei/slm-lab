"""Protect cached-photo backwards from a caller's ambient autocast region.

The preserved v1 diagnostic found both a reference FP32 pair-gradient drift
and a smaller encoder-gradient drift under ambient BF16. Keep forward AMP
explicit and disable ambient AMP for derivative computation and backwards.
No weights, objective, source coverage or numerical tolerance is changed.
"""
import torch

from .global_photo_gradient_cache_stage_v1 import accumulate_global_photo_cached as _original
from .own_actor_vision_prefix_split_backward_v1 import OwnActorVisionPrefixSplitBackward


def accumulate_global_photo_cached(view, *args, **kwargs):
    if type(view) is not OwnActorVisionPrefixSplitBackward:
        # Preserve the original coordinated rejection on every replica.
        return _original(view, *args, **kwargs)
    device = next(view.parameters()).device.type
    if device not in ['cpu','cuda']:
        return _original(view, *args, **kwargs)
    if kwargs.get('amp') is None and torch.is_autocast_enabled(device):
        dtype = torch.get_autocast_dtype(device)
        cache = torch.is_autocast_cache_enabled()
        kwargs['amp'] = lambda: torch.autocast(device, dtype=dtype, cache_enabled=cache)
    with torch.autocast(device, enabled=False):
        record = _original(view, *args, **kwargs)
    return dict(**record, derivative_and_backward_autocast_explicitly_disabled=True)
