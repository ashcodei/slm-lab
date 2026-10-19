"""Replay complete native/photo/word updates before admitting derived casts.

The six extra physical capacity updates are separate from the original full
nine-family capacity controller. Restore the exact durable untrained boundary
after both prefixes; keep the cache attached only after every rank agrees.
No production update, resource lease or full-size admission is created here.
"""
import time

import torch

from .frozen_linear_cast_cache_v1 import FrozenLinearCastCache
from .joint_visual_durable_state_v2 import read_latest
from .joint_visual_experiment_v1 import live
from .joint_visual_objective_update_v1 import FAMILIES, named_groups
from .joint_visual_private_checkpoint_v2 import _agree, digest
from .joint_visual_training_stage_v1 import JointVisualStage
from .parallel_function_repair_stage_v1 import value_digest


def prove_prefix(stage, directory, *, maximum_bytes, minimum_free_bytes, allow_mechanics=False):
    error = None
    try:
        if (type(stage) is not JointVisualStage or stage.completed or stage.failed
                or stage.mode != 'capacity' and not (allow_mechanics is True and stage.mode == 'mechanics')
                or stage._options['precision'] != 'bf16' or len(stage._entries) < 3
                or [FAMILIES[e['family']] for e in stage._entries[:3]] != ['native', 'photo', 'word']):
            raise ValueError('A fresh BF16capacity stage with complete native/photo/word prefix is required')
    except Exception as exc:
        error = repr(exc)
    _agree(error, None, stage.group, stage.world)
    stage.commit(directory, maximum_bytes=maximum_bytes, minimum_free_bytes=minimum_free_bytes)
    _, receipt, pointer = read_latest(directory)
    if receipt['completed'] != 0:
        raise ValueError('The independent original selected initializer must be durably committed at cursor0')
    initial = live(stage)
    observations = []
    hook_errors = []
    selected = [item for rows in named_groups(stage.view).values() for item in rows]
    def before_step(optimizer, args, kwargs):
        # Never strand a peer in a collective by raising from an optimizer
        # callback. Coordinate any observation error after the ordinary step.
        try:
            observations.append(dict(actual_Adam_input_gradient_sha256=value_digest({n: p.grad for n, p in selected}),
                actual_selected_leaves=len(selected), actual_active_gradient_leaves=sum(p.grad is not None for _, p in selected)))
        except Exception as exc:
            hook_errors.append(repr(exc))
    hook = stage.optimizer.register_step_pre_hook(before_step)
    cache = None
    def next_observation():
        observations.clear()
        hook_errors.clear()
        if stage.device.type == 'cuda':
            torch.cuda.synchronize(stage.device)
        start = time.monotonic()
        row = stage.next_update()
        if stage.device.type == 'cuda':
            torch.cuda.synchronize(stage.device)
        seconds = time.monotonic() - start
        _agree(None if not hook_errors and len(observations) == 1 else
            'Every complete objective must have one observed ordinary Adam-input gradient boundary', None, stage.group, stage.world)
        return dict(actual_record=row, actual_Adam_input_gradients=observations[0], actual_after=live(stage)), seconds
    try:
        expected, uncached_seconds = [], []
        for _ in range(3):
            result, elapsed = next_observation()
            expected.append(result)
            uncached_seconds.append(elapsed)
        stage.restore(directory, expected_pointer_sha256=pointer)
        _agree(None if live(stage) == initial else 'The original durable boundary did not restore exactly', None, stage.group, stage.world)
        error = None
        try:
            cache = FrozenLinearCastCache(device=stage.device,
                purpose='CPU-mechanics' if stage.mode == 'mechanics' else 'GPU-runtime').attach(stage.view)
        except Exception as exc:
            error = repr(exc)
        _agree(error, None, stage.group, stage.world)
        cached_seconds = []
        for wanted in expected:
            actual, elapsed = next_observation()
            cached_seconds.append(elapsed)
            _agree(None if actual == wanted else 'Derived casts changed actual objective, Adam-input gradients, values, moments, cursor, metrics or private RNG',
                None, stage.group, stage.world)
            cache.assert_current()
        stage.restore(directory, expected_pointer_sha256=pointer)
        _agree(None if live(stage) == initial else 'Cached capacity replay did not return to the complete initializer boundary', None, stage.group, stage.world)
        cache.assert_current()
        proof = dict(complete=True, passed=True, rank=stage.rank, world=stage.world, mode=stage.mode,
            actual_device_type=stage.device.type, actual_device_index=stage.device.index,
            actual_research_parameters=stage.view.actor.counts()['actual_unique'],
            actual_native_parameters=stage.view.actor.base.counts()['total_unique'],
            actual_source_prefix_sha256=digest(stage._entries[:3]),
            actual_complete_prefix_presentations=sum(len(e['original_local_positions']) for e in stage._entries[:3]),
            actual_prefix_families=[e['family'] for e in stage._entries[:3]],
            actual_separate_physical_capacity_updates=6, actual_logical_completed_after_restoration=0,
            all_three_objectives_and_Adam_input_gradients_and_weights_Adam_metrics_private_RNG_exact=True,
            actual_Adam_input_gradient_observations=[e['actual_Adam_input_gradients'] for e in expected],
            exact_complete_uncached_observations_sha256=value_digest(expected),
            exact_durable_initializer_restored_before_full_capacity=True,
            external_original_initializer_pointer_sha256=pointer,
            cache_inventory=cache.inventory(), full_size_GPU_equivalence=stage.mode == 'capacity',
            uncached_capacity_probe_seconds=uncached_seconds, cached_capacity_probe_seconds=cached_seconds,
            timings_include_warmup_synchronization_and_instrumentation=True,
            timings_are_not_a_production_throughput_or_GPU_speedup_benchmark=True,
            new_production_optimizer_updates=0, goal_achieved=False)
        return cache, proof
    except BaseException:
        stage.failed = True
        if cache is not None:
            cache.close()
        raise
    finally:
        hook.remove()
