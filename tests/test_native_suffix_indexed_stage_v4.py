"""Actual small-neural indexed history against the closed matched baseline.

Only newly added source-binding metadata is normalized for comparison. Every
original binding remains exact; actions, rewards, probabilities, gradients,
Adam and private RNG must match the already executed original baseline.
The repeated one-program/five-action CPU fixture cannot admit production.
"""
import copy
from datetime import timedelta
import gc
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch
from torch import distributed as dist
from torch.multiprocessing import spawn

from slm_lab.codenet_checkpoint_v1 import capture_rng
from slm_lab.corpus import file_hash
from slm_lab.factored_native_observation_stream_v1 import Reader as BaselineReader
from slm_lab.indexed_native_history_v2 import read_failure
from slm_lab.native_suffix_indexed_stage_v4 import NativeSuffixIndexedStageV4, FORMAT, HISTORY
from slm_lab.native_suffix_train_data_v1 import NativeSuffixTrainData
from slm_lab.parallel_function_repair_stage_v1 import manifest, value_digest
from slm_lab.parallel_native_suffix_stage_v1 import record_semantics
from slm_lab.parallel_repair_durable_state_v1 import commit, read_latest, restore_latest
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
from test_native_suffix_capacity_controller_v2 import student
from test_native_suffix_compact_stage_v3 import sources as baseline_sources
import test_parallel_native_suffix_stage_v1 as original
from test_public_reasoning_group_gradients_v1 import optimizer

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / 'reports/native-compact-neural-v3-proof-v1/pytest/test_actual_four_replicas_both0'
BUDGETS = dict(maximum_records=3, maximum_compact_record_bytes=16 * 1024**2,
    maximum_expanded_record_bytes=64 * 1024**2, maximum_logical_bytes=512 * 1024**2,
    maximum_physical_bytes=512 * 1024**2, minimum_free_bytes=0)


def sources():
    result = baseline_sources()
    for relative in ['src/slm_lab/indexed_native_history_v1.py', 'src/slm_lab/indexed_native_history_v2.py',
            'src/slm_lab/native_suffix_indexed_stage_v4.py', 'tests/test_native_suffix_indexed_stage_v4.py']:
        path = ROOT / relative
        result[str(path)] = file_hash(path)
    return result


def setup(public, profile='anchor_execution', rank=0, world=1, control=None):
    own = student(public)
    random.seed(989731 + rank); np.random.seed(989731 + rank); torch.manual_seed(989731 + rank)
    entries = [[0], [1], [2]] if world == 1 else [[0, 1, 2, 3], [4, 5, 6, 7], [8]]
    return NativeSuffixIndexedStageV4(own, optimizer(own), public, entries, profile, original.OPTIONS, sources(),
        mode='mechanics', rank=rank, world=world, control_group=control, gradient_group=None, history_budgets=BUDGETS)


def live(stage):
    return dict(actor=tensor_digest(stage.model.parts.named_parameters()), Adam=value_digest(stage.optimizer.state_dict()),
        RNG=value_digest(capture_rng('cpu')), completed=stage.completed, metrics=value_digest(stage.metrics), failed=stage.failed,
        active_directory=str(stage.history_reader.root) if stage.history_reader else None,
        writer_identity=id(stage.history_writer), tensor_identities=[(id(p),k,id(t),t.data_ptr(),t._version)
            for p, values in stage.optimizer.state.items() for k,t in values.items()])


def baseline(profile):
    wrapper = json.loads((ROOT / 'reports/native-compact-neural-v3-proof-v1/complete.json').read_text())
    assert wrapper['pytest_exit_code'] == 0 and wrapper['all_original_source_bytes_unchanged']
    reader = BaselineReader(BASELINE / profile / 'baseline.nativeobs.gz', maximum_logical_bytes=256 * 1024**2)
    rows = [row['actual_complete_update'] for _, row in reader]
    assert reader.complete and len(rows) == 3
    reference = rows[0]['actual_rank_observations'][0]['kernel']['own_reference_binding']
    proofs = [json.loads((BASELINE / (profile + '-rank-' + str(rank) + '.json')).read_text()) for rank in range(4)]
    state, receipt, _ = read_latest(BASELINE / profile / 'durable')
    assert receipt['checkpoint_sha256'] == proofs[0]['final_actual_checkpoint_sha256']
    return [value_digest(record_semantics(row)) for row in rows], reference, proofs, state


def compare_semantics(row, reference):
    normalized = copy.deepcopy(row)
    for observation in normalized['actual_rank_observations']:
        if observation['kernel'] is None: continue
        actual = observation['kernel']['own_reference_binding']
        assert {k:v for k,v in actual.items() if k != 'sources'} == {k:v for k,v in reference.items() if k != 'sources'}
        assert all(actual['sources'].get(k) == v for k,v in reference['sources'].items())
        assert all(Path(k).is_file() and file_hash(k) == v for k,v in actual['sources'].items() if k not in reference['sources'])
        observation['kernel']['own_reference_binding'] = copy.deepcopy(reference)
    return value_digest(record_semantics(normalized))


def test_actual_single_replica_pre_Adam_file_publication_snapshot_and_detached_recovery(tmp_path):
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    public = NativeSuffixTrainData.CPU_mechanics(original.ROOT, original.CONTROL)
    stage = setup(public)
    before = live(stage)
    with pytest.raises(ValueError, match='before native sampling'): stage.update()
    assert live(stage) == before and not stage.optimizer.state
    stage.open_history(tmp_path / 'attempt1')
    actual_step = stage.optimizer.step
    def checked_step(*args, **kwargs):
        receipt_path = stage.history_reader.root / 'record-000000000001.receipt.json'
        assert receipt_path.is_file()
        stage.history_reader.compact(json.loads(receipt_path.read_text()))
        return actual_step(*args, **kwargs)
    stage.optimizer.step = checked_step
    row = stage.update()
    assert stage.completed == 1 and stage.metrics[0]['format'] == HISTORY
    assert next(stage.expanded_metrics()) == row
    receipt = commit(stage, tmp_path / 'durable', maximum_bytes=32 * 1024**2, minimum_free_bytes=0)
    saved, external, _ = read_latest(tmp_path / 'durable')
    assert saved['format'] == FORMAT and receipt == external
    assert saved['complete_observations_are_durable_receipts'] and len(saved['metrics']) == 1
    stage.resume(saved, expected_manifest=external['semantic_manifest'])
    assert stage.history_reader is None and stage.history_writer is None
    stage.open_history(tmp_path / 'attempt2')
    assert stage.history_start == 1 and (tmp_path / 'attempt1/record-000000000001.json.gz').is_file()
    assert not torch.cuda.is_initialized()


def worker(rank, directory):
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    folder = Path(directory)
    dist.init_process_group('gloo', init_method='file://' + str(folder / 'rendezvous'),
        rank=rank, world_size=4, timeout=timedelta(minutes=15))
    control = dist.new_group(backend='gloo', timeout=timedelta(minutes=15))
    try:
        public = NativeSuffixTrainData.CPU_mechanics(original.ROOT, original.CONTROL)
        public._information[4]['native_execution_eligible'] = False
        for profile in ['anchor_CE', 'anchor_execution']:
            expected, reference, old_proofs, old_state = baseline(profile)
            stage = setup(public, profile, rank, 4, control)
            stage.open_history(folder / profile / 'attempt1')
            actual_step, before_Adam_checks = stage.optimizer.step, []
            def checked_step(*args, **kwargs):
                sequence = stage.completed + 1 - stage.history_start
                receipt = json.loads((stage.history_reader.root / ('record-' + str(sequence).zfill(12) + '.receipt.json')).read_text())
                assert receipt['step'] == stage.completed + 1
                stage.history_reader.compact(receipt)
                before_Adam_checks.append(receipt['step'])
                return actual_step(*args, **kwargs)
            stage.optimizer.step = checked_step
            for index in range(3):
                row = stage.update()
                assert compare_semantics(row, reference) == expected[index]
                if index == 1: commit(stage, folder / profile / 'durable', maximum_bytes=32 * 1024**2, minimum_free_bytes=0)
            expected_live = live(stage)
            assert expected_live['actor'] == old_proofs[rank]['final_complete_actor_sha256']
            assert expected_live['Adam'] == old_proofs[rank]['final_AdamW_sha256']
            assert expected_live['RNG'] == value_digest(old_state['private_workers'][rank]['rng'])
            old_tail = stage.history_reader.root / 'record-000000000003.json.gz'
            tail_sha = file_hash(old_tail)
            checkpoint, receipt, pointer = read_latest(folder / profile / 'durable')
            for fault in ['foreign_RNG', 'receipt_header', 'legacy_format']:
                bad, semantic = copy.deepcopy(checkpoint), receipt['semantic_manifest']
                if rank == 2:
                    if fault == 'foreign_RNG': bad['private_workers'][2]['rng']['torch'] = torch.Generator().manual_seed(923199).get_state()
                    elif fault == 'receipt_header':
                        bad['metrics'][0]['header_sha256'] = '0' * 64
                        semantic = manifest(bad)
                    else:
                        bad['format'] = 'own-native-code-compact-collectives-history-shared-AdamW-private-RNG-v3'
                        semantic = manifest(bad)
                before = live(stage)
                with pytest.raises(ValueError, match='before mutation'): stage.resume(bad, expected_manifest=semantic)
                assert live(stage) == before
            restore_latest(stage, folder / profile / 'durable', expected_pointer_sha256=pointer)
            assert stage.history_reader is None and stage.history_writer is None
            stage.open_history(folder / profile / 'attempt2')
            replay = stage.update()
            assert compare_semantics(replay, reference) == expected[-1]
            assert all(live(stage)[k] == expected_live[k] for k in ['actor', 'Adam', 'RNG', 'completed'])
            assert file_hash(old_tail) == tail_sha
            assert len(stage.metrics) == 3 and stage.metrics[-1]['record']['sequence'] == 1
            assert len({m['directory'] for m in stage.metrics}) == 2
            assert len(before_Adam_checks) == 4
            final = commit(stage, folder / profile / 'durable', maximum_bytes=32 * 1024**2, minimum_free_bytes=0)
            assert sum(r['original_response_EOS_targets'] for r in stage.expanded_metrics()) == 612
            assert len(json.dumps(stage.metrics)) < 16 * 1024
            assert not torch.cuda.is_initialized()
            (folder / (profile + '-rank-' + str(rank) + '.json')).write_text(json.dumps(dict(profile=profile, rank=rank,
                actual_CPU_physical_AdamW_calls=4, actual_research_actor_parameters=267718,
                closed_original_baseline_reused_without_new_baseline_optimizer_calls=True,
                only_verified_new_source_metadata_normalized_for_semantic_comparison=True,
                all_actions_rewards_probabilities_gradients_and_final_actor_Adam_private_RNG_exact=True,
                actual_pre_Adam_durable_readback_checks=len(before_Adam_checks),
                failed_or_uncommitted_prior_tail_file_preserved=True,
                exact_durable_replay_uses_fresh_attempt_at_saved_cursor=True,
                actual_resident_history_receipt_JSON_bytes=len(json.dumps(stage.metrics)),
                final_actual_checkpoint_sha256=final['checkpoint_sha256'], production_updates=0,
                full_size_GPUcapacity_or_quality_gain=False), indent=2))
            del stage, old_state, checkpoint, bad, row, replay
            gc.collect()
        # This injected failure occurs after real sampling, execution and
        # gradients on every source rank, before any Adam call can occur.
        import slm_lab.native_suffix_indexed_stage_v4 as module
        from slm_lab.batched_native_suffix_group_gradients_v1 import CodeGroupFailure
        stage = setup(public, 'anchor_execution', rank, 4, control)
        stage.open_history(folder / 'actual-peer-sampling-failure')
        original_kernel = module.group_gradients
        def failed_peer(*args, **kwargs):
            actual = original_kernel(*args, **kwargs)
            if rank == 2: raise CodeGroupFailure(dict(phase='explicit peer fault after real complete sampling', complete_actual_kernel=actual))
            return actual
        module.group_gradients = failed_peer
        before = live(stage)
        try:
            with pytest.raises(ValueError): stage.update()
        finally: module.group_gradients = original_kernel
        assert stage.failed and stage.completed == 0 and not stage.optimizer.state
        assert all(live(stage)[k] == before[k] for k in ['actor', 'Adam', 'RNG', 'completed'])
        failure = read_failure(stage.history_reader.root, stage.last_failure_receipt, maximum_bytes=64 * 1024**2)
        captured = failure['original_observation']
        assert len(captured['all_actual_rank_observations']) == 4
        assert all(len(r['complete_actual_local_kernel']['trajectories']) == 8
            for r in captured['all_actual_rank_observations'] if r['rank'] != 2)
        rejected = captured['complete_coordinated_failure_observations'][-1]['actual_rank_errors'][2]
        assert len(rejected['complete_actual_failure_observation']['complete_actual_kernel']['trajectories']) == 8
        assert not failure['successful_optimizer_commit_certified']
        (folder / ('failure-rank-' + str(rank) + '.json')).write_text(json.dumps(dict(rank=rank,
            all32_actual_sampled_trajectories_and_native_verdicts_preserved=True,
            no_Adam_actor_or_global_RNG_mutation=True, actual_failure_receipt=stage.last_failure_receipt), indent=2))
    finally:
        dist.destroy_process_group()


def test_actual_four_replicas_both_profiles_pre_Adam_durable_receipts_fresh_replay_and_complete_peer_failure(tmp_path):
    spawn(worker, args=(str(tmp_path),), nprocs=4, join=True)
    reports = [json.loads((tmp_path / (profile + '-rank-' + str(rank) + '.json')).read_text())
        for profile in ['anchor_CE', 'anchor_execution'] for rank in range(4)]
    assert sum(r['actual_CPU_physical_AdamW_calls'] for r in reports) == 32
    assert all((tmp_path / ('failure-rank-' + str(rank) + '.json')).exists() for rank in range(4))
    (tmp_path / 'complete.json').write_text(json.dumps(dict(actual_physical_small_CPU_AdamW_calls=32,
        actual_replicas=4, profiles=2, real_failure_source_groups_without_Adam=4,
        all_original_completed_baseline_math_and_recovery_equal=True,
        every_pre_Adam_observation_durable_before_actual_optimizer_call=True,
        all_complete_peer_failure_samples_and_verdicts_retained=True,
        full_size_GPUcapacity_or_quality=False, production_updates=0, ranks=reports), indent=2))
