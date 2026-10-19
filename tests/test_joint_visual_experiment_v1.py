"""Actual own-small visual replay and complete-loop integration controls."""
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

from slm_lab.joint_visual_durable_state_v2 import read_latest
from slm_lab.joint_visual_experiment_v1 import prove_next_update, train, live
from test_joint_visual_training_stage_v1 import stage_fixture, weights, equal_weights


@pytest.mark.parametrize('bf16', [False, True])
def test_actual_cross_objective_replay_then_whole_remaining_word_source(bf16, tmp_path):
    stage, _ = stage_fixture(tail=True, bf16=bf16)
    proof = prove_next_update(stage, tmp_path, maximum_bytes=4 * 1024**2, minimum_free_bytes=0)
    assert proof['actual_logical_completed_source_entries'] == stage.completed == 2
    assert proof['actual_physical_optimizer_calls'] == 3
    assert proof['exact_next_source_weights_Adam_metrics_and_private_RNG']
    assert proof['invalid_source_active_moment_and_nonfinite_saved_values_rejected_before_mutation']
    assert proof['mechanics_only'] and not proof['valid_foreign_rank_RNG_rejected_before_mutation']
    last = stage.next_update()
    assert last['family'] == 'words' and stage.completed == 3 and stage.next_entry is None
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('bf16', [False, True])
def test_complete_own_small_loop_commits_initializer_every_source_and_final_state(bf16, tmp_path):
    stage, _ = stage_fixture(tail=True, bf16=bf16)
    progress, preflights = [], []
    result = train(stage, tmp_path / 'shared', tmp_path / 'journal.jsonl', every=2,
        maximum_bytes=4 * 1024**2, minimum_free_bytes=0, maximum_journal_bytes=2 * 1024**2,
        preflight=lambda: preflights.append(stage.completed),
        progress=lambda row, elapsed: progress.append((row['family'], elapsed)), allow_mechanics=True)
    shared, receipt, _ = read_latest(tmp_path / 'shared')
    with (tmp_path / 'journal.jsonl').open('rb') as stream:
        rows = [json.loads(line) for line in stream]
    assert [row['step'] for row in rows] == [1, 2, 3] and preflights == [0, 1, 2]
    assert [row['actual_record'] for row in rows] == stage.metrics
    assert [name for name, _ in progress] == [r['family'] for r in stage.metrics]
    assert receipt['generation'] == 4 and shared['common']['cursor']['completed'] == 3
    assert shared['private_states'][0]['metrics'] == stage.metrics
    assert result['actual_completed_source_entries'] == 3
    assert result['actual_complete_source_presentations'] == 4
    assert result['actual_response_EOS_targets'] == sum(r['response_EOS_targets'] for r in stage.metrics)
    assert result['actual_initial_selected_baseline_committed'] and result['production_updates'] == 0
    assert not stage.failed and result['mechanics_only'] and not result['goal_achieved']


def test_small_mechanics_loop_requires_explicit_scope_and_preserves_numerical_state(tmp_path):
    stage, _ = stage_fixture(tail=True)
    before = live(stage)
    with pytest.raises(ValueError, match='explicit own-small mechanics loop'):
        train(stage, tmp_path / 'shared', tmp_path / 'journal.jsonl', every=2,
            maximum_bytes=4 * 1024**2, minimum_free_bytes=0, maximum_journal_bytes=2 * 1024**2,
            preflight=lambda: None, progress=lambda *args: None)
    assert live(stage) == before and not list(tmp_path.iterdir())


def worker(rank, folder):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and os.nice(0) >= 19
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    dist.init_process_group('gloo', init_method='file://' + str(Path(folder) / 'rendezvous'), rank=rank,
        world_size=4, timeout=timedelta(minutes=5))
    try:
        stage, _ = stage_fixture(world=4, rank=rank, tail=True)
        random.seed(3000 + rank)
        np.random.seed(4000 + rank)
        torch.manual_seed(5000 + rank)
        proof = prove_next_update(stage, Path(folder) / 'capacity', maximum_bytes=4 * 1024**2, minimum_free_bytes=0)
        assert proof['valid_foreign_rank_RNG_rejected_before_mutation']
        stage.next_update()
        assert stage.metrics[-1]['local_objective_record']['actual_native_forward_calls'] == (1 if rank == 0 else 0)
        fresh, _ = stage_fixture(world=4, rank=rank, tail=True)
        random.seed(6000 + rank)
        np.random.seed(7000 + rank)
        torch.manual_seed(8000 + rank)
        result = train(fresh, Path(folder) / 'train', Path(folder) / 'journal.jsonl', every=2,
            maximum_bytes=4 * 1024**2, minimum_free_bytes=0, maximum_journal_bytes=2 * 1024**2,
            preflight=lambda: None, progress=lambda *args: None, allow_mechanics=True)
        shared, receipt, pointer = read_latest(Path(folder) / 'train')
        assert shared['private_states'][rank]['metrics'] == fresh.metrics
        assert len({p['private_RNG_sha256'] for p in shared['private_states']}) == 4
        assert result['actual_completed_source_entries'] == 3 and result['production_updates'] == 0
        # A single worker's ownership/source preflight error stops every peer
        # before the next source performs any neural forward or optimizer step.
        broken, _ = stage_fixture(world=4, rank=rank, tail=True)
        before = weights(broken.view)
        calls = []
        hook = broken.view.native.vision.register_forward_hook(lambda *args: calls.append(1))
        def failure():
            if rank == 2:
                raise ValueError('deliberate own-small control preflight error')
        with pytest.raises(ValueError, match='Coordinated private visual checkpoint rejection'):
            train(broken, Path(folder) / 'preflight-control', Path(folder) / 'preflight-control-journal.jsonl', every=2,
                maximum_bytes=4 * 1024**2, minimum_free_bytes=0, maximum_journal_bytes=2 * 1024**2,
                preflight=failure, progress=lambda *args: None, allow_mechanics=True)
        hook.remove()
        assert not calls and not broken.optimizer.state and broken.completed == 0 and broken.failed
        equal_weights(before, broken.view)
        baseline, baseline_receipt, _ = read_latest(Path(folder) / 'preflight-control')
        assert baseline['common']['cursor']['completed'] == 0 and baseline_receipt['completed'] == 0
        assert not torch.cuda.is_initialized()
        (Path(folder) / (str(rank) + '.json')).write_text(json.dumps(dict(complete=True, rank=rank,
            actual_own_small_parameters=fresh.view.actor.counts()['actual_unique'],
            actual_capacity_logical_updates=2, actual_capacity_physical_optimizer_calls=3,
            exact_next_cross_objective_weights_Adam_metrics_private_RNG=True,
            source_missing_moment_nonfinite_saved_value_and_valid_peer_RNG_reject_before_mutation=True,
            complete_actual_three_source_loop_and_all_rank_private_metrics=True,
            all_four_actual_private_RNGs_distinct=True,
            actual_final_shared_checkpoint_sha256=receipt['checkpoint_sha256'],
            one_rank_preflight_failure_coordinated_before_any_forward=True,
            actual_initial_baseline_and_complete_final_cursor_committed=True,
            actual_singleton_tail_with_three_empty_ranks=True,
            production_updates=0, full_size_GPU_capacity_owned_controller_and_quality_pending=True), indent=2) + '\n')
    finally:
        dist.destroy_process_group()


def test_four_actual_small_replica_capacity_replay_full_loop_and_preflight_coordination(tmp_path):
    spawn(worker, args=(str(tmp_path),), nprocs=4, join=True)
    proofs = [json.loads((tmp_path / (str(rank) + '.json')).read_text()) for rank in range(4)]
    assert all(p['complete'] and p['complete_actual_three_source_loop_and_all_rank_private_metrics'] for p in proofs)
    assert len({p['actual_final_shared_checkpoint_sha256'] for p in proofs}) == 1
