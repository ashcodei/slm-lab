"""Actual all-nine-family own-small capacity controller; scope controls are synthetic."""
import copy
from datetime import timedelta
import json
import os
from pathlib import Path
import random

import numpy as np
import pytest
import torch
from torch import distributed as dist
from torch.multiprocessing import spawn

from slm_lab.joint_visual_controller_v1 import run_capacity, require_full_capacity, FORMAT, FAMILIES
from slm_lab.joint_visual_training_stage_v1 import JointVisualStage, make_plan
from slm_lab.joint_visual_private_checkpoint_v2 import FixedGuard, digest
from test_joint_visual_private_checkpoint_v2 import fixture, adam, sources, prepare, OPTIONS
from test_joint_visual_cold_overlay_v1 import inputs


class NineFamilyMechanics:
    def __init__(self, batch):
        self.batch = batch

    def prepare(self, entry, *, rank, world, device):
        kind = FAMILIES[entry['family']]
        tail = kind == 'word' and len(entry['original_local_positions']) == 1
        prepared, source = prepare(self.batch, kind, rank, world, tail)
        source['family'] = entry['family']
        if tail:
            source['global_positions'] = [29]
        return prepared, source


def nine_stage(rank=0, world=1, bf16=False):
    view, batch = fixture()
    optimizer = adam(view)
    fixed = FixedGuard(view, optimizer)
    provider = NineFamilyMechanics(batch)
    native_families = [n for n in FAMILIES if FAMILIES[n] == 'native']
    order = [native_families[0], 'images:additional_photo_captions', 'words', *native_families[1:], 'words']
    entries = []
    for step, family in enumerate(order, 1):
        kind = FAMILIES[family]
        _, source = prepare(batch, kind, rank, world, tail=step == len(order))
        if step == len(order):
            source['global_positions'] = [29]
        positions = source['global_positions']
        entries.append(dict(step=step, family=family, original_local_positions=positions,
            complete_context_tokens=sum(source['global_context_lengths']),
            response_EOS_targets=sum(source['global_response_EOS_targets']),
            proposed_rank_source_counts=[len(positions[r::world]) for r in range(world)]))
    bound = sources(fixed)
    plan = make_plan(view, optimizer, fixed, sources={k: v for k, v in bound.items() if k != 'run_plan_sha256'},
        entries=entries, options=dict(OPTIONS, precision='bf16' if bf16 else 'fp32'), mode='mechanics')
    bound['run_plan_sha256'] = digest(plan)
    stage = JointVisualStage(view, optimizer, provider, plan=plan, sources=bound, entries=entries, rank=rank, world=world, fixed=fixed)
    # Explicit fixture plan; it lacks all full-source production populations.
    capacity_plan = dict(capacity_entries=entries, actual_original_epoch_entries=len(entries),
        actual_original_epoch_source_views=sum(len(e['original_local_positions']) for e in entries))
    return stage, batch, capacity_plan


def independent():
    with torch.random.fork_rng(devices=[]):
        view, _ = fixture()
    return view.actor.base


@pytest.mark.parametrize('bf16', [False, True])
def test_actual_all_nine_family_controller_recovery_final_commit_and_independent_native_cold(bf16, tmp_path):
    stage, batch, capacity_plan = nine_stage(bf16=bf16)
    result = run_capacity(stage, tmp_path, capacity_plan=capacity_plan, independent_native=independent,
        cold_inputs=inputs(batch), maximum_bytes=4 * 1024**2, minimum_free_bytes=0, allow_mechanics=True)
    assert result['actual_complete_capacity_entries'] == 10 and result['actual_physical_optimizer_calls'] == 11
    assert result['actual_capacity_source_presentations'] == 13 and result['actual_complete_source_families'] == sorted(FAMILIES)
    assert result['actual_final_sparse_group_clocks'] == dict(vision=10, pointing=1, word=2)
    assert result['actual_selected_Adam_leaves'] == result['actual_selected_parameter_leaves']
    assert result['independent_native_cold']['actual_text_photo_logits_and_word_readout_bit_exact']
    assert not result['full_size_GPU_capacity_passed'] and result['production_updates'] == 0
    with pytest.raises(ValueError):
        require_full_capacity([result] * 4, expected_bindings=result['original_source_bindings'], expected_capacity_plan_sha256=digest(capacity_plan))
    assert not torch.cuda.is_initialized()


def reports():
    # Pure admission-schema controls. These are not actual capacity receipts.
    rows = []
    for rank in range(4):
        rows.append(dict(format=FORMAT, complete=True, passed=True, rank=rank, world=4, mode='capacity',
            actual_device_type='cuda', actual_device_index=rank, actual_actor_parameters=999706442,
            actual_native_parameters=984399690, original_source_bindings=dict(source='a' * 64), capacity_plan_sha256='b' * 64,
            actual_complete_capacity_entries=10, actual_capacity_source_presentations=510, actual_physical_optimizer_calls=11,
            actual_complete_source_families=sorted(FAMILIES), actual_selected_Adam_leaves=130, actual_selected_parameter_leaves=130,
            actual_final_sparse_group_clocks=dict(vision=10, pointing=1, word=2), full_size_GPU_capacity_passed=True,
            historical_different_word_and_prefix_numeric_failures_not_waived=True,
            all_original_worst_cases_and_real_empty_tail_observed=True, production_updates=0,
            recovery=dict(exact_next_source_weights_Adam_metrics_and_private_RNG=True, valid_foreign_rank_RNG_rejected_before_mutation=True,
                invalid_source_active_moment_and_nonfinite_saved_values_rejected_before_mutation=True),
            independent_native_cold=dict(mode='capacity', mechanics_only=False, actual_text_context_tokens=4096,
                actual_photo_side=448, actual_word_shape=[64, 448], actual_word_readout_dtype='torch.float32',
                every_original_fixed_and_selected_native_value_equal=True, actual_text_photo_logits_and_word_readout_bit_exact=True,
                independent_native_storage_without_aliasing=True), final_actual_shared_checkpoint_sha256='c' * 64))
    return rows


def test_full_capacity_schema_requires_all_original_scopes_without_completing_the_goal():
    result = require_full_capacity(reports(), expected_bindings=dict(source='a' * 64), expected_capacity_plan_sha256='b' * 64)
    assert result['complete'] and result['production_updates'] == 0 and result['goal_achieved'] is False
    assert result['final_complete_native_export_and_full_assessments_still_required']


@pytest.mark.parametrize('case', ['missing_rank', 'wrong_physical_GPU', 'changed_source', 'missing_family',
    'partial_views', 'short_text_cold', 'missing_word_Adam', 'foreign_RNG_not_checked', 'CPU_mechanics', 'different_shared_state'])
def test_partial_or_nonmatching_capacity_cannot_unlock_full_training(case):
    rows = reports()
    if case == 'missing_rank':
        rows.pop()
    elif case == 'wrong_physical_GPU':
        rows[2]['actual_device_index'] = 1
    elif case == 'changed_source':
        rows[2]['original_source_bindings']['source'] = 'd' * 64
    elif case == 'missing_family':
        rows[2]['actual_complete_source_families'].pop()
    elif case == 'partial_views':
        rows[2]['actual_capacity_source_presentations'] = 12
    elif case == 'short_text_cold':
        rows[2]['independent_native_cold']['actual_text_context_tokens'] = 128
    elif case == 'missing_word_Adam':
        rows[2]['actual_selected_Adam_leaves'] = 112
    elif case == 'foreign_RNG_not_checked':
        rows[2]['recovery']['valid_foreign_rank_RNG_rejected_before_mutation'] = False
    elif case == 'CPU_mechanics':
        rows[2]['mode'] = 'mechanics'
    else:
        rows[2]['final_actual_shared_checkpoint_sha256'] = 'd' * 64
    with pytest.raises(ValueError):
        require_full_capacity(rows, expected_bindings=dict(source='a' * 64), expected_capacity_plan_sha256='b' * 64)


def worker(rank, folder):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and os.nice(0) >= 19
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    dist.init_process_group('gloo', init_method='file://' + str(Path(folder) / 'rendezvous'), rank=rank,
        world_size=4, timeout=timedelta(minutes=5))
    try:
        stage, batch, plan = nine_stage(rank=rank, world=4)
        random.seed(9000 + rank)
        np.random.seed(10000 + rank)
        torch.manual_seed(11000 + rank)
        result = run_capacity(stage, Path(folder) / 'shared', capacity_plan=plan, independent_native=independent,
            cold_inputs=inputs(batch), maximum_bytes=4 * 1024**2, minimum_free_bytes=0, allow_mechanics=True)
        assert result['actual_complete_capacity_entries'] == 10 and result['actual_physical_optimizer_calls'] == 11
        assert result['recovery']['valid_foreign_rank_RNG_rejected_before_mutation']
        assert result['actual_final_sparse_group_clocks'] == dict(vision=10, pointing=1, word=2)
        assert stage.metrics[-1]['local_objective_record']['actual_native_forward_calls'] == (1 if rank == 0 else 0)
        assert result['independent_native_cold']['actual_text_photo_logits_and_word_readout_bit_exact']
        assert not result['full_size_GPU_capacity_passed'] and not torch.cuda.is_initialized()
        (Path(folder) / (str(rank) + '.json')).write_text(json.dumps(result, indent=2) + '\n')
    finally:
        dist.destroy_process_group()


def test_four_actual_own_small_nine_family_capacity_controllers_and_independent_native_cold(tmp_path):
    spawn(worker, args=(str(tmp_path),), nprocs=4, join=True)
    proofs = [json.loads((tmp_path / (str(r) + '.json')).read_text()) for r in range(4)]
    assert all(p['complete'] and p['actual_complete_capacity_entries'] == 10 for p in proofs)
    assert len({p['final_actual_shared_checkpoint_sha256'] for p in proofs}) == 1
