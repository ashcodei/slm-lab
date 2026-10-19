"""Actual indexed four-worker capacity, full fixed fixture horizon and cold readback.

The fixture repeats one human program across nine source presentations. Its
explicit five-action CPU horizon never grants full native GPU admission.
"""
import copy
from datetime import timedelta
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch
from torch import distributed as dist
from torch.multiprocessing import spawn

from slm_lab.corpus import file_hash
from slm_lab.native_suffix_batched_history_stage_v5 import NativeSuffixBatchedHistoryStageV5
from slm_lab.native_suffix_indexed_stage_v4 import NativeSuffixIndexedStageV4
from slm_lab.native_suffix_capacity_controller_v2 import run_capacity, live
from slm_lab.native_suffix_capacity_evidence_v2 import run_stress_cases, require_capacity
from slm_lab.native_suffix_capacity_readback_v2 import read_capacity, check_readbacks
from slm_lab.native_suffix_train_data_v1 import NativeSuffixTrainData
from slm_lab.parallel_repair_durable_state_v1 import read_latest
from test_native_suffix_batched_history_stage_v5 import sources as previous_sources
from test_native_suffix_capacity_controller_v2 import student, inputs
from test_native_suffix_indexed_stage_v4 import BUDGETS
from test_public_reasoning_group_gradients_v1 import optimizer
import test_parallel_native_suffix_stage_v1 as original

ROOT = Path(__file__).resolve().parents[1]


def sources():
    result = previous_sources()
    for relative in ['src/slm_lab/native_history_summary_v1.py', 'src/slm_lab/native_allocated_history_guard_v1.py',
            'src/slm_lab/native_suffix_capacity_controller_v2.py', 'src/slm_lab/native_suffix_capacity_evidence_v2.py',
            'src/slm_lab/native_suffix_capacity_readback_v2.py', 'src/slm_lab/native_suffix_full_horizon_capacity_v2.py',
            'tests/test_native_suffix_capacity_indexed_v2.py']:
        path = ROOT / relative
        result[str(path)] = file_hash(path)
    return result


def setup(public, profile, rank, control):
    own = student(public)
    random.seed(989731 + rank); np.random.seed(989731 + rank); torch.manual_seed(989731 + rank)
    return NativeSuffixBatchedHistoryStageV5(own, optimizer(own), public,
        [[0, 1, 2, 3], [4, 5, 6, 7], [8]], profile, original.OPTIONS, sources(), mode='mechanics',
        rank=rank, world=4, control_group=control, gradient_group=None, history_budgets=BUDGETS)


@pytest.mark.parametrize('cls', [NativeSuffixIndexedStageV4, NativeSuffixBatchedHistoryStageV5])
def test_legacy_or_partial_single_replica_cannot_admit_indexed_capacity_or_create_output(tmp_path, cls):
    # Structural API rejection only, before constructing any model or source
    # data. Positive actual four-worker neural/cold checks are separate.
    stage = cls.__new__(cls)
    stage.rank, stage.world, stage.control_group = 0, 1, None
    stage.completed, stage.failed, stage.failure_observations = 0, False, []
    with pytest.raises(ValueError):
        run_capacity(stage, tmp_path / 'private', capacity_plan=dict(capacity_entries=[]), independent_actor=lambda: None,
            cold_inputs={}, maximum_bytes=32 * 1024**2, minimum_free_bytes=0,
            history_directory=tmp_path / 'history', allow_mechanics=True)
    assert not (tmp_path / 'private').exists() and not (tmp_path / 'history').exists()
    assert stage.completed == 0 and not stage.failed


def worker(rank, directory):
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    folder = Path(directory)
    dist.init_process_group('gloo', init_method='file://' + str(folder / 'rendezvous'), rank=rank,
        world_size=4, timeout=timedelta(minutes=15))
    control = dist.new_group(backend='gloo', timeout=timedelta(minutes=15))
    try:
        public = NativeSuffixTrainData.CPU_mechanics(original.ROOT, original.CONTROL)
        public._information[4]['native_execution_eligible'] = False
        cold_inputs = inputs(public)
        def independent():
            with torch.random.fork_rng(devices=[]): return student(public)
        for profile in ['anchor_CE', 'anchor_execution']:
            stage = setup(public, profile, rank, control)
            plan = dict(capacity_entries=copy.deepcopy(stage.entries), mechanics_only=True,
                actual_full_source_selection_not_used_by_repeated_one_program_fixture=True)
            before = live(stage)
            with pytest.raises(ValueError):
                run_capacity(stage, folder / 'forbidden', capacity_plan=plan, independent_actor=independent,
                    cold_inputs=cold_inputs, maximum_bytes=32 * 1024**2, minimum_free_bytes=0,
                    history_directory=folder / 'forbidden-history')
            assert live(stage) == before
            wrong = copy.deepcopy(plan)
            if rank == 2: wrong['mechanics_only'] = False
            with pytest.raises(ValueError):
                run_capacity(stage, folder / 'wrong', capacity_plan=wrong, independent_actor=independent,
                    cold_inputs=cold_inputs, maximum_bytes=32 * 1024**2, minimum_free_bytes=0,
                    history_directory=folder / 'wrong-history', allow_mechanics=True)
            assert live(stage) == before
            private = folder / profile / 'private'
            result = run_capacity(stage, private, capacity_plan=plan, independent_actor=independent,
                cold_inputs=cold_inputs, maximum_bytes=32 * 1024**2, minimum_free_bytes=0,
                history_directory=folder / profile / 'history', allow_mechanics=True)
            assert result['actual_complete_capacity_updates'] == 3 and result['actual_physical_optimizer_calls'] == 4
            assert result['actual_execution_eligible_presentations'] == 8 and result['actual_complete_CE_only_presentations'] == 1
            assert result['actual_unconditional_original_CE_targets'] == 612
            assert result['actual_committed_prefix_trajectory_count'] == 64
            assert result['actual_single_source_three_empty_rank_steps'] == 1
            assert result['no_complete_neural_history_materialized_by_capacity_controller']
            assert len(result['actual_durable_indexed_history_inventory']) == 2
            assert result['actual_allocated_block_guard_reports'][0]['actual_sealed_record_count'] == 2
            assert result['independent_complete_actor_cold']['complete_text_photo_word_outputs_bit_exact']
            horizons = run_stress_cases(stage, allow_mechanics=True, mechanics_positions=[0, 8])
            assert horizons['new_optimizer_calls'] == 0 and horizons['new_generated_actions'] == 0
            reports, stresses = stage.gather(result), stage.gather(horizons)
            require_capacity(stage, reports, stresses, plan, allow_mechanics=True)
            final_actor = result['actual_per_rank_intervals'][rank]['actor_sha256']
            fresh = setup(public, profile, rank, control)
            restored = read_capacity(fresh, private, reports=reports, horizon_reports=stresses,
                capacity_plan=plan, independent_actor=independent, cold_inputs=cold_inputs, allow_mechanics=True)
            readbacks = fresh.gather(restored)
            check_readbacks(fresh, reports, stresses, readbacks, plan, True)
            assert restored['new_optimizer_calls'] == 0 and restored['actual_actor_sha256'] == final_actor
            assert restored['independent_complete_actor_cold']['complete_text_photo_word_outputs_bit_exact']
            saved, receipt, _ = read_latest(private)
            assert saved['completed'] == 3 and receipt['checkpoint_sha256'] == result['final_actual_shared_checkpoint_sha256']
            assert len(json.dumps(saved['metrics'])) < 16 * 1024
            assert not result['full_size_GPU_capacity_passed'] and not result['production_training_admitted']
            assert not torch.cuda.is_initialized()
            (folder / (profile + '-rank-' + str(rank) + '.json')).write_text(json.dumps(dict(
                capacity=result, full_fixed_fixture_horizon=horizons, independent_fresh_readback=restored), indent=2, allow_nan=False))
    finally:
        dist.destroy_process_group()


def test_actual_four_replicas_both_profiles_indexed_capacity_horizon_fresh_readback_and_cold(tmp_path):
    spawn(worker, args=(str(tmp_path),), nprocs=4, join=True)
    actual = []
    for profile in ['anchor_CE', 'anchor_execution']:
        reports = [json.loads((tmp_path / (profile + '-rank-' + str(rank) + '.json')).read_text()) for rank in range(4)]
        assert len({r['capacity']['final_actual_shared_checkpoint_sha256'] for r in reports}) == 1
        actual.extend(reports)
    assert sum(r['capacity']['actual_physical_optimizer_calls'] for r in actual) == 32
    (tmp_path / 'complete.json').write_text(json.dumps(dict(
        actual_physical_small_CPU_AdamW_calls=32, actual_replicas=4, actual_profiles=2,
        each_source_context_CE_fallback_empty_tail_and_private_replay_preserved=True,
        actual_fixed_fixture_horizon_cases_per_profile=2, actual_independent_complete_cold_checks_per_profile=2,
        actual_fresh_capacity_readback_optimizer_calls=0, complete_dynamic_history_not_materialized=True,
        full_size_GPUcapacity_or_quality_gain=False, production_updates=0), indent=2))
