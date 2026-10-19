"""Actual native workers, durable private recovery and complete own-small cold.

The nine-presentation fixture repeats one original human program. It is not
full native-data admission, full-size GPU capacity or learned code quality.
"""
import copy
from datetime import timedelta
import json
from pathlib import Path

import pytest
import torch
from torch import distributed as dist
from torch.multiprocessing import spawn

from slm_lab.native_suffix_capacity_controller_v1 import run_capacity, live, FORMAT
from slm_lab.native_suffix_train_data_v1 import NativeSuffixTrainData
from slm_lab.native_suffix_validation_stage_v2 import NativeSuffixValidationStageV2
from slm_lab.parallel_repair_durable_state_v1 import read_latest
import test_parallel_native_suffix_stage_v1 as original


def setup(public, profile='anchor_execution', rank=0, world=1, control=None):
    # Reuse the original fixture's real native stage constructor arguments;
    # select the V2 class only in this new test process, never an active worker.
    prior = original.ParallelNativeSuffixStage
    original.ParallelNativeSuffixStage = NativeSuffixValidationStageV2
    try: return original.setup(public, profile, rank, world, control)
    finally: original.ParallelNativeSuffixStage = prior


def inputs(public):
    ids, _ = public.corpus.sequence(public.information(0)['original_source_position'])
    text = torch.tensor([ids[:32]], dtype=torch.long)
    photo = torch.tensor([ids[:8]], dtype=torch.long)
    return dict(text=dict(input_ids=text, attention_mask=torch.ones_like(text, dtype=torch.bool)),
        photo=dict(input_ids=photo, attention_mask=torch.ones_like(photo, dtype=torch.bool),
            pixel_values=torch.linspace(-1., 1., 3 * 16 * 16).reshape(1, 3, 16, 16)),
        word=dict(word_pixels=torch.linspace(-1., 1., 3 * 64 * 64).reshape(1, 3, 64, 64)))


def test_single_replica_cannot_enter_actual_four_replica_native_capacity(tmp_path):
    public = NativeSuffixTrainData.CPU_mechanics(original.ROOT, original.CONTROL)
    stage = setup(public)
    before = live(stage)
    with pytest.raises(ValueError):
        run_capacity(stage, tmp_path / 'forbidden', capacity_plan=dict(capacity_entries=stage.entries, mechanics_only=True),
            independent_actor=lambda: original.student(public), cold_inputs=inputs(public), maximum_bytes=32 * 1024**2,
            minimum_free_bytes=0, allow_mechanics=True)
    assert live(stage) == before and not stage.optimizer.state and not (tmp_path / 'forbidden').exists()


def worker(rank, folder):
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    folder = Path(folder)
    dist.init_process_group('gloo', init_method='file://' + str(folder / 'rendezvous'), rank=rank, world_size=4,
        timeout=timedelta(minutes=10))
    control = dist.new_group(backend='gloo', timeout=timedelta(minutes=10))
    try:
        public = NativeSuffixTrainData.CPU_mechanics(original.ROOT, original.CONTROL)
        # An explicit fixture disposition exercises unconditional complete CE
        # fallback. It does not relabel the real source's production eligibility.
        public._information[4]['native_execution_eligible'] = False
        cold_inputs = inputs(public)
        def independent():
            with torch.random.fork_rng(devices=[]): return original.student(public)
        for profile in ['anchor_CE', 'anchor_execution']:
            stage = setup(public, profile, rank, 4, control)
            plan = dict(capacity_entries=copy.deepcopy(stage.entries), mechanics_only=True,
                actual_full_source_capacity_plan_not_used_by_this_repeated_one_program_fixture=True)
            before = live(stage)
            with pytest.raises(ValueError):
                run_capacity(stage, folder / 'forbidden', capacity_plan=plan, independent_actor=independent,
                    cold_inputs=cold_inputs, maximum_bytes=32 * 1024**2, minimum_free_bytes=0)
            assert live(stage) == before
            wrong = copy.deepcopy(plan)
            if rank == 2: wrong['mechanics_only'] = False
            with pytest.raises(ValueError):
                run_capacity(stage, folder / 'wrong', capacity_plan=wrong, independent_actor=independent,
                    cold_inputs=cold_inputs, maximum_bytes=32 * 1024**2, minimum_free_bytes=0, allow_mechanics=True)
            assert live(stage) == before
            directory = folder / profile / 'durable'
            result = run_capacity(stage, directory, capacity_plan=plan, independent_actor=independent,
                cold_inputs=cold_inputs, maximum_bytes=32 * 1024**2, minimum_free_bytes=0, allow_mechanics=True)
            assert result['format'] == FORMAT and result['actual_complete_capacity_updates'] == 3
            assert result['actual_physical_optimizer_calls'] == 4 and result['actual_Adam_states'] == 17
            assert result['actual_source_budget']['source_presentations'] == 9
            assert result['actual_source_budget']['distinct_original_human_sources'] == 1
            assert result['actual_execution_eligible_presentations'] == 8 and result['actual_complete_CE_only_presentations'] == 1
            assert result['actual_unconditional_original_CE_targets'] == 612
            assert result['actual_single_source_three_empty_rank_steps'] == 1
            assert result['actual_committed_prefix_trajectory_count'] == 64
            assert result['actual_all_physical_global_generated_actions_including_replay'] > result['actual_committed_prefix_generated_action_count']
            assert result['actual_research_actor_parameters'] == 267334
            assert result['all_actual_next_actions_execution_semantics_unclipped_gradients_Adam_private_RNG_exact']
            assert result['invalid_peer_CPU_RNG_rejected_before_mutation'] and result['peer_only_valid_foreign_RNG_rejected_before_mutation']
            assert result['execution_timings_and_thread_IDs_retained_outside_replay_equality']
            assert not result['full_size_GPU_capacity_passed'] and not result['full_size_actual_selected_workload_passed']
            assert not result['production_training_admitted'] and not result['new_model_quality_gain']
            cold = result['independent_complete_actor_cold']
            assert cold['complete_text_photo_word_outputs_bit_exact'] and cold['all_private_RNGs_unchanged']
            assert cold['actual_text_positions'] == 32 and cold['actual_photo_side'] == 16 and cold['actual_word_shape'] == [64, 64]
            saved, receipt, _ = read_latest(directory)
            assert saved['completed'] == 3 and len(saved['private_workers']) == 4
            assert saved['metrics'][-1]['actual_rank_observations'][3]['kernel'] is None
            assert saved['metrics'][-1]['actual_rank_observations'][0]['actual_original_positions'] == [8]
            assert receipt['checkpoint_sha256'] == result['final_actual_shared_checkpoint_sha256']
            assert not torch.cuda.is_initialized()
            (folder / (profile + '-rank-' + str(rank) + '.json')).write_text(json.dumps(result, indent=2, allow_nan=False))
    finally:
        dist.destroy_process_group()


def test_actual_four_replicas_both_profiles_native_worker_replay_CE_fallback_empty_tail_and_independent_cold(tmp_path):
    spawn(worker, args=(str(tmp_path),), nprocs=4, join=True)
    for profile in ['anchor_CE', 'anchor_execution']:
        reports = [json.loads((tmp_path / (profile + '-rank-' + str(r) + '.json')).read_text()) for r in range(4)]
        assert len({r['final_actual_shared_checkpoint_sha256'] for r in reports}) == 1
        assert all(r['independent_complete_actor_cold']['complete_text_photo_word_outputs_bit_exact'] for r in reports)
        assert sum(r['actual_physical_optimizer_calls'] for r in reports) == 16
