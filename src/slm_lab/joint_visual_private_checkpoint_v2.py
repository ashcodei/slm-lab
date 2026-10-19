"""Reject invalid live snapshots before any durable visual commit can publish.

Preserve the v1 finite-state wire representation and recovery math. Add a
coordinated live selected-weight/ordinary-Adam preflight. Recovery may repair
damaged current selected values using a valid independently committed file.
"""
import math

import torch

from .joint_visual_objective_update_v1 import optimizer_groups
from .joint_visual_private_checkpoint_v1 import (FORMAT, SOURCE_KEYS, FixedGuard, digest,
    private_rng_sha256, _common, _control, _agree)
from . import joint_visual_private_checkpoint_v1 as legacy


def settings(view, optimizer):
    groups = optimizer_groups(view, optimizer)
    for group in optimizer.param_groups:
        for name in ['lr', 'eps', 'weight_decay']:
            value = group.get(name)
            if (type(value) not in [float, int] or not math.isfinite(value)
                    or value < 0 or name != 'weight_decay' and value == 0):
                raise ValueError('Every current ordinary visual Adam learning rate/epsilon/decay must be finite and valid')
        betas = group.get('betas')
        if (not isinstance(betas, (tuple, list)) or len(betas) != 2
                or any(type(v) not in [int, float] or not math.isfinite(v) or not 0 <= v < 1 for v in betas)):
            raise ValueError('Both current ordinary visual Adam betas must remain finite in[0,1)')
    return groups


def make_state(view, optimizer, fixed, *, sources, entries, completed, metrics, mode,
               rank=0, world=1, device='cpu', control_group=None):
    _, actual_world = _control(control_group)
    error = None
    try:
        groups = settings(view, optimizer)
        with torch.no_grad():
            if any(not bool(torch.isfinite(p).all()) for rows in groups.values() for _, p in rows):
                raise ValueError('Every live selected visual weight must be finite before snapshot publication')
    except Exception as exc:
        error = type(exc).__name__ + ': ' + str(exc)
    _agree(error, None, control_group, actual_world)
    return legacy.make_state(view, optimizer, fixed, sources=sources, entries=entries, completed=completed,
        metrics=metrics, mode=mode, rank=rank, world=world, device=device, control_group=control_group)


def restore_state(view, optimizer, fixed, state, *, sources, entries, mode, rank=0, world=1,
                  device='cpu', control_group=None, expected_completed, expected_private_RNG_sha256, allow_completed=False):
    _, actual_world = _control(control_group)
    error = None
    try:
        settings(view, optimizer)
    except Exception as exc:
        error = type(exc).__name__ + ': ' + str(exc)
    _agree(error, None, control_group, actual_world)
    return legacy.restore_state(view, optimizer, fixed, state, sources=sources, entries=entries, mode=mode,
        rank=rank, world=world, device=device, control_group=control_group,
        expected_completed=expected_completed, expected_private_RNG_sha256=expected_private_RNG_sha256,
        allow_completed=allow_completed)
