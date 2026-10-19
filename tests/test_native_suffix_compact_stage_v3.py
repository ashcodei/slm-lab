"""Matched actual CPU neural updates, full-map history and durable replay.

The nine-presentation fixture repeats one human program. Its explicit five
action CPU limit is not the original 512-action horizon or a GPU admission.
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
from slm_lab.factored_native_observation_stream_v1 import Writer, Reader, canonical
from slm_lab.native_reference_records_v1 import PROTOCOL
from slm_lab.native_suffix_compact_stage_v3 import NativeSuffixCompactStageV3, FORMAT
from slm_lab.native_suffix_train_data_v1 import NativeSuffixTrainData
from slm_lab.native_suffix_validation_stage_v2 import NativeSuffixValidationStageV2
from slm_lab.parallel_function_repair_stage_v1 import manifest, value_digest
from slm_lab.parallel_native_suffix_stage_v1 import record_semantics, CE_ONLY
from slm_lab import parallel_repair_durable_state_v1 as durable
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
from test_factored_native_observation_stream_v1 import SOURCE
from test_native_suffix_capacity_controller_v2 import student
import test_parallel_native_suffix_stage_v1 as original
from test_public_reasoning_group_gradients_v1 import optimizer

ROOT = Path(__file__).resolve().parents[1]


def sources():
    result = json.loads(SOURCE.read_text())['source_files_sha256']
    for relative in ['src/slm_lab/native_reference_records_v1.py',
            'src/slm_lab/native_suffix_compact_stage_v3.py',
            'tests/test_native_suffix_compact_stage_v3.py']:
        path = ROOT / relative
        result[str(path)] = file_hash(path)
    return result


def setup(public, stage_type, profile, rank, control):
    own = student(public)
    random.seed(989731 + rank)
    np.random.seed(989731 + rank)
    torch.manual_seed(989731 + rank)
    return stage_type(own, optimizer(own), public, [[0, 1, 2, 3], [4, 5, 6, 7], [8]],
        profile, original.OPTIONS, sources(), mode='mechanics', rank=rank, world=4,
        control_group=control, gradient_group=None)


def live(stage):
    return dict(actor=tensor_digest(stage.model.parts.named_parameters()),
        optimizer=value_digest(stage.optimizer.state_dict()), RNG=value_digest(capture_rng('cpu')),
        completed=stage.completed, metrics=value_digest(stage.metrics), failed=stage.failed,
        tensor_identities=[(id(p), k, id(t), t.data_ptr(), t._version)
            for p, row in stage.optimizer.state.items() for k, t in row.items()])


def optimized(stage):
    state = live(stage)
    return {k: state[k] for k in ['actor', 'optimizer', 'RNG', 'completed']}


def journal(stage, path):
    if stage.rank != 0: return None
    path.parent.mkdir(parents=True, exist_ok=True)
    return Writer(path, root=path.parent, reference_binding=stage.reference_binding,
        maximum_logical_bytes=256 * 1024**2, maximum_physical_bytes=32 * 1024**2,
        minimum_free_bytes=0, guard=stage.data.guard.check)


def worker(rank, directory):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    folder = Path(directory)
    dist.init_process_group('gloo', init_method='file://' + str(folder / 'rendezvous'),
        rank=rank, world_size=4, timeout=timedelta(minutes=15))
    control = dist.new_group(backend='gloo', timeout=timedelta(minutes=15))
    try:
        public = NativeSuffixTrainData.CPU_mechanics(original.ROOT, original.CONTROL)
        public._information[4]['native_execution_eligible'] = False
        for profile in ['anchor_CE', 'anchor_execution']:
            baseline = setup(public, NativeSuffixValidationStageV2, profile, rank, control)
            baseline_weights, baseline_rows = [], []
            out = journal(baseline, folder / profile / 'baseline.nativeobs.gz')
            try:
                for _ in range(3):
                    row = baseline.update()
                    if out is not None: out.append(dict(actual_complete_update=row))
                    baseline_rows.append(value_digest(record_semantics(row)))
                    baseline_weights.append(optimized(baseline))
            finally:
                if out is not None: out.close()
            frozen_base = tensor_digest(baseline.model.base.named_parameters())
            del baseline, row
            gc.collect()
            stage = setup(public, NativeSuffixCompactStageV3, profile, rank, control)
            assert stage.reference_records.identity == value_digest(stage.reference_binding)
            before_base = tensor_digest(stage.model.base.named_parameters())
            assert before_base == frozen_base
            out = journal(stage, folder / profile / 'compact-and-replay.nativeobs.gz')
            try:
                for index in range(3):
                    row = stage.update()
                    if out is not None: out.append(dict(actual_complete_update=row))
                    assert value_digest(record_semantics(row)) == baseline_rows[index]
                    assert optimized(stage) == baseline_weights[index]
                    assert canonical(stage.reference_records.expand(stage.metrics[-1])) == canonical(row)
                    if index == 1:
                        assert row['actual_rank_observations'][0]['kernel']['format'] == CE_ONLY
                        receipt = durable.commit(stage, folder / profile / 'durable',
                            maximum_bytes=32 * 1024**2, minimum_free_bytes=0)
                        assert receipt['completed'] == 2
                    if index == 2:
                        assert [o['kernel'] is None for o in row['actual_rank_observations']] == [False, True, True, True]
                expected = optimized(stage)
                checkpoint, receipt, pointer = durable.read_latest(folder / profile / 'durable')
                assert checkpoint['format'] == FORMAT
                assert checkpoint['history_reference_protocol'] == PROTOCOL
                assert checkpoint['reference_binding_sha256'] == stage.reference_records.identity
                assert checkpoint['completed'] == 2
                faults = ['invalid_CPU_RNG', 'valid_foreign_CPU_RNG', 'reference_header', 'marker', 'old_format']
                for fault in faults:
                    bad = copy.deepcopy(checkpoint)
                    external = receipt['semantic_manifest']
                    if rank == 2:
                        if fault == 'invalid_CPU_RNG':
                            bad['private_workers'][2]['rng']['torch'] = torch.tensor([1], dtype=torch.uint8)
                        elif fault == 'valid_foreign_CPU_RNG':
                            bad['private_workers'][2]['rng']['torch'] = torch.Generator().manual_seed(179002).get_state()
                        elif fault == 'reference_header':
                            bad['reference_binding']['external_pretrained_teacher'] = 0
                            # The old generic manifest does not hash this field.
                            assert manifest(bad) == external
                        elif fault == 'marker':
                            bad['metrics'][0]['actual_rank_observations'][0]['kernel']['own_reference_binding']['sha256'] = '0' * 64
                            # Even a matching local manifest cannot admit an invalid reference.
                            external = manifest(bad)
                        else:
                            bad['format'] = 'own-parallel-native-code-suffix-shared-AdamW-private-RNG-v1'
                            external = manifest(bad)
                    before = live(stage)
                    with pytest.raises(ValueError, match='before mutation'):
                        stage.resume(bad, expected_manifest=external)
                    assert live(stage) == before
                durable.restore_latest(stage, folder / profile / 'durable', expected_pointer_sha256=pointer)
                assert stage.completed == 2
                replay = stage.update()
                if out is not None: out.append(dict(actual_complete_update=replay))
                assert value_digest(record_semantics(replay)) == baseline_rows[-1]
                assert optimized(stage) == expected
            finally:
                if out is not None: out.close()
            final_receipt = durable.commit(stage, folder / profile / 'durable',
                maximum_bytes=32 * 1024**2, minimum_free_bytes=0)
            assert final_receipt['completed'] == 3
            assert len(stage.optimizer.state) == 17
            assert all(float(v['step']) == 3 for v in stage.optimizer.state.values())
            assert tensor_digest(stage.model.base.named_parameters()) == before_base
            assert not torch.cuda.is_initialized()
            total_targets = sum(r['original_response_EOS_targets'] for r in stage.expanded_metrics())
            assert total_targets == 612
            proof = dict(profile=profile, rank=rank, actual_binding_count=len(stage.sources),
                actual_small_research_actor_parameters=stage.model.counts()['actual_unique'],
                physical_optimizer_calls=7, logical_source_updates_per_arm=3, source_presentations_per_arm=9,
                distinct_human_programs=1, actual_fixture_horizon=5, original_native_horizon=512,
                complete_unconditional_CE_targets_per_arm=total_targets,
                every_matched_action_execution_semantic_gradient_actor_AdamW_and_private_RNG_exact=True,
                all_complete_raw_timings_and_thread_IDs_retained_outside_semantic_equality=True,
                real_CE_only_source_and_three_empty_tail_ranks_preserved=True,
                durable_compact_next_update_replay_exact=True,
                peer_faults_rejected_before_any_mutation=faults,
                final_complete_actor_sha256=expected['actor'], final_AdamW_sha256=expected['optimizer'],
                final_actual_checkpoint_sha256=final_receipt['checkpoint_sha256'],
                actual_committed_semantic_row_sha256=baseline_rows,
                full_size_GPU_capacity_passed=False, production_updates=0, new_quality_gain=False, goal_achieved=False)
            (folder / (profile + '-rank-' + str(rank) + '.json')).write_text(json.dumps(proof, indent=2))
            del stage, checkpoint, bad, row, replay
            gc.collect()
    finally:
        dist.destroy_process_group()


def test_actual_four_replicas_both_profiles_match_original_neural_updates_and_durable_replay(tmp_path):
    spawn(worker, args=(str(tmp_path),), nprocs=4, join=True)
    all_proofs = []
    for profile in ['anchor_CE', 'anchor_execution']:
        reports = [json.loads((tmp_path / (profile + '-rank-' + str(rank) + '.json')).read_text()) for rank in range(4)]
        assert len({r['final_actual_checkpoint_sha256'] for r in reports}) == 1
        assert len({r['final_complete_actor_sha256'] for r in reports}) == 1
        assert all(r['actual_small_research_actor_parameters'] == 267718 for r in reports)
        for name, count in [('baseline', 3), ('compact-and-replay', 4)]:
            reader = Reader(tmp_path / profile / (name + '.nativeobs.gz'), maximum_logical_bytes=256 * 1024**2)
            rows = list(reader)
            assert reader.complete and len(rows) == count
        all_proofs.extend(reports)
    assert sum(r['physical_optimizer_calls'] for r in all_proofs) == 56
    (tmp_path / 'complete.json').write_text(json.dumps(dict(
        actual_physical_small_CPU_AdamW_calls=56, actual_profiles=2, actual_replicas=4,
        all_matched_neural_updates_and_durable_replay_exact=True, production_updates=0,
        full_size_GPU_capacity_or_quality_gain=False, ranks=all_proofs), indent=2))
