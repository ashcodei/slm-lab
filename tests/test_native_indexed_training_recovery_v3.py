"""Complete indexed training, post-Adam failure recovery and native export.

Reuse the closed four-replica capacity proof without repeating its updates.
The small CPU fixture repeats one human program; it cannot admit full-model
training, the original 512-action horizon or any capability improvement.
"""
import copy
from datetime import timedelta
import json
from pathlib import Path
import shutil

import pytest
import torch
from torch import distributed as dist
from torch.multiprocessing import spawn

from slm_lab.bounded_gzip_observation_stream_v1 import lines
from slm_lab.codenet_checkpoint_v1 import capture_rng
from slm_lab.corpus import file_hash
from slm_lab.indexed_native_history_v2 import read_failure
from slm_lab.native_suffix_capacity_controller_v2 import live
from slm_lab.native_suffix_standalone_export_v3 import export
from slm_lab.native_suffix_train_data_v1 import NativeSuffixTrainData
from slm_lab.native_suffix_training_controller_v3 import run_training
from slm_lab.own_dense_standalone_v1 import load
from slm_lab.parallel_function_repair_stage_v1 import value_digest
from slm_lab.parallel_native_suffix_stage_v1 import record_semantics
from slm_lab.parallel_repair_durable_state_v1 import read_latest
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
from test_native_suffix_capacity_controller_v2 import student, inputs
from test_native_suffix_capacity_indexed_v2 import setup, sources
import test_parallel_native_suffix_stage_v1 as original

ROOT = Path(__file__).resolve().parents[1]
CAPACITY = ROOT / 'reports/native-indexed-capacity-v2-proof-v1'
RUNTIME_FILES = ['src/slm_lab/native_training_runtime_bindings_v1.py',
    'src/slm_lab/native_indexed_history_inventory_v3.py',
    'tests/test_native_indexed_history_inventory_v3.py',
    'src/slm_lab/native_suffix_training_controller_v3.py',
    'src/slm_lab/native_suffix_standalone_export_v3.py',
    'tests/test_native_indexed_training_contract_v2.py',
    'tests/test_native_indexed_export_records_v2.py',
    'tests/test_native_indexed_training_recovery_v3.py',
    'scripts/check_native_indexed_training_CPU_v2.py']


def runtime_sources():
    return {str(ROOT / name): file_hash(ROOT / name) for name in RUNTIME_FILES}


def run(stage, folder, plan, reports, horizons, readbacks, *, progress=None, pointer=None, attempt='first', opt_in=True):
    return run_training(stage, folder / 'private', folder / (attempt + '-history'), folder / (attempt + '-events.jsonl.gz'),
        capacity_plan=plan, reports=reports, horizon_reports=horizons, readbacks=readbacks,
        runtime_sources=runtime_sources(), checkpoint_every=1, maximum_state_bytes=32 * 1024**2,
        minimum_free_bytes=0, maximum_event_raw_bytes=4096 * 3,
        preflight=lambda: stage.data.guard.check(), progress=progress,
        expected_resume_pointer_sha256=pointer, allow_mechanics=opt_in)


def worker(rank, directory):
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    folder = Path(directory)
    dist.init_process_group('gloo', init_method='file://' + str(folder / 'rendezvous'), rank=rank,
        world_size=4, timeout=timedelta(minutes=15))
    control = dist.new_group(backend='gloo', timeout=timedelta(minutes=15))
    try:
        public = NativeSuffixTrainData.CPU_mechanics(original.ROOT, original.CONTROL)
        public._information[4]['native_execution_eligible'] = False
        for profile in ['anchor_CE', 'anchor_execution']:
            def fresh(): return setup(public, profile, rank, control)
            stage = fresh()
            plan = dict(capacity_entries=copy.deepcopy(stage.entries), mechanics_only=True,
                actual_full_source_selection_not_used_by_repeated_one_program_fixture=True)
            artifacts = [json.loads((CAPACITY / 'pytest/test_actual_four_replicas_both0' /
                (profile + '-rank-' + str(r) + '.json')).read_text()) for r in range(4)]
            reports = [r['capacity'] for r in artifacts]
            horizons = [r['full_fixed_fixture_horizon'] for r in artifacts]
            readbacks = [r['independent_fresh_readback'] for r in artifacts]
            assert stage.sources == reports[rank]['original_source_bindings']
            before = live(stage)
            broken = copy.deepcopy(readbacks)
            if rank == 2: broken[1]['actual_private_RNG_sha256'] = 'c' * 64
            with pytest.raises(ValueError): run(stage, folder / profile / 'forbidden', plan, reports, horizons, broken)
            with pytest.raises(ValueError): run(stage, folder / profile / 'no-opt-in', plan, reports, horizons, readbacks, opt_in=False)
            assert live(stage) == before and not (folder / profile / 'forbidden').exists()
            result = run(stage, folder / profile / 'complete', plan, reports, horizons, readbacks)
            expected = live(stage)
            assert result['actual_complete_updates'] == result['actual_new_optimizer_calls'] == 3
            assert result['actual_controller_event_stream']['records'] == 3
            assert result['actual_controller_event_stream']['complete']
            assert result['actual_streamed_complete_history_summary']['actual_committed_prefix_trajectory_count'] == 64
            assert result['original_complete_training_budget']['original_response_EOS_targets'] == 612
            assert result['original_complete_training_budget']['CE_only_presentations'] == 1
            assert result['original_complete_training_budget']['actual_single_source_three_empty_rank_steps'] == 1

            interrupted = fresh()
            recovery = folder / profile / 'recovery'
            def stop_after_second(row, elapsed):
                if row['step'] == 2: raise OSError('explicit root failure after actual second Adam and fsynced event')
            with pytest.raises(ValueError, match='complete scalar timing'):
                run(interrupted, recovery, plan, reports, horizons, readbacks, progress=stop_after_second)
            assert interrupted.failed and interrupted.completed == 2
            assert all(p.grad is None for p in interrupted.model.parameters())
            saved, _, pointer = read_latest(recovery / 'private')
            assert saved['completed'] == 1
            old_events = [row for _, row in lines(recovery / 'first-events.jsonl.gz')]
            assert [row['actual_complete_update'] for row in old_events] == [1, 2]
            failed_receipt = interrupted.last_controller_failure_receipt
            failed = read_failure(recovery / 'first-history', failed_receipt, maximum_bytes=64 * 1024**2)
            failed_row = failed['original_observation']['complete_last_actual_neural_row']
            assert failed_row['step'] == 2 and not failed['successful_optimizer_commit_certified']
            retained_files = {str(p): file_hash(p) for p in (recovery / 'first-history').iterdir()
                if p.name != 'independent-failure.reserved'}

            resumed = fresh()
            before = live(resumed)
            with pytest.raises(ValueError):
                run(resumed, recovery, plan, reports, horizons, readbacks, pointer='d' * 64, attempt='bad-pin')
            assert live(resumed) == before and not (recovery / 'bad-pin-history').exists()
            resumed_result = run(resumed, recovery, plan, reports, horizons, readbacks, pointer=pointer, attempt='resumed')
            after = live(resumed)
            assert all(after[k] == expected[k] for k in ['actor', 'Adam', 'RNG', 'completed', 'failed'])
            assert result['actual_streamed_complete_history_summary'] == resumed_result['actual_streamed_complete_history_summary']
            assert resumed_result['actual_attempt_started_at_update'] == 1
            assert resumed_result['actual_new_optimizer_calls'] == 2
            assert len(resumed_result['actual_durable_indexed_history_inventory']) == 2
            second_events = [row for _, row in lines(recovery / 'resumed-events.jsonl.gz')]
            assert [row['actual_complete_update'] for row in second_events] == [2, 3]
            with resumed.checked_validation_sources():
                replayed = next(row for row in resumed.expanded_metrics() if row['step'] == 2)
            assert value_digest(record_semantics(failed_row)) == value_digest(record_semantics(replayed))
            assert all(file_hash(path) == digest for path, digest in retained_files.items())
            assert not torch.cuda.is_initialized()
            (folder / (profile + '-rank-' + str(rank) + '.json')).write_text(json.dumps(dict(
                rank=rank, profile=profile, actual_small_CPU_AdamW_calls=7,
                uninterrupted_training=result, interrupted_then_recovered_training=resumed_result,
                exact_final_actor_Adam_private_RNG_and_every_mathematical_history_record=True,
                post_Adam_root_failure_keeps_full_neural_row_timing_and_prior_checkpoint=True,
                failed_uncommitted_second_update_replayed_exactly=True,
                all_failed_attempt_files_unchanged_after_recovery=True,
                actual_failure_receipt=failed_receipt, production_updates=0,
                full_size_GPUcapacity_or_quality_gain=False), indent=2, allow_nan=False))
    finally:
        dist.destroy_process_group()


def export_and_cold(folder, profile, branch, public):
    state, _, _ = read_latest(folder / profile / branch / 'private')
    record = json.loads((folder / (profile + '-rank-0.json')).read_text())
    receipt = record['uninterrupted_training' if branch == 'complete' else 'interrupted_then_recovered_training']
    with torch.random.fork_rng(devices=[]): hot = student(public).eval()
    hot.import_export(state['model'], expected_sources=sources())
    before = tensor_digest(hot.named_parameters()); rng = value_digest(capture_rng('cpu'))
    native, payload, proof = export(hot, public, state, receipt, sources=sources(), runtime_sources=runtime_sources(),
        tokenizer_sha256=file_hash(public.corpus.tokenizer_path), working_budget_bytes=256 * 1024**2, allow_mechanics=True)
    assert tensor_digest(hot.named_parameters()) == before and value_digest(capture_rng('cpu')) == rng
    assert native.counts()['total_unique'] == 267430
    assert torch.equal(native.network.value_head.weight, hot.base.network.value_head.weight)
    path = folder / profile / branch / 'standalone'; path.mkdir()
    shutil.copyfile(public.corpus.tokenizer_path, path / 'tokenizer.json'); torch.save(payload, path / 'model.pt')
    artifact_sources = dict(sources()); artifact_sources.update(runtime_sources())
    cold, metadata = load(path / 'model.pt', expected_checkpoint_sha256=file_hash(path / 'model.pt'), expected_sources=artifact_sources)
    assert tensor_digest(native.named_parameters()) == tensor_digest(cold.named_parameters())
    assert metadata['no_old_base_actor_optimizer_or_training_reference_loaded']
    assert not ({p.data_ptr() for p in native.parameters()} & {p.data_ptr() for p in hot.parameters()})
    differences = {}
    with torch.no_grad(), torch.autocast('cpu', enabled=False):
        batches = inputs(public)
        for name in ['text', 'photo']:
            expected, loaded, unfused = native(**batches[name])['logits'], cold(**batches[name])['logits'], hot(**batches[name])['logits']
            assert torch.equal(expected, loaded) and bool(torch.isfinite(expected).all())
            differences[name] = float((expected - unfused).abs().max())
        assert torch.equal(native.word_logits(batches['word']['word_pixels']), cold.word_logits(batches['word']['word_pixels']))
    assert value_digest(capture_rng('cpu')) == rng
    return dict(complete=True, standalone_sha256=file_hash(path / 'model.pt'), native_parameter_sha256=tensor_digest(native.named_parameters()),
        complete_text_photo_word_cold_outputs_exact=True, source_actor_and_private_RNG_unchanged=True,
        hot_to_folded_FP32_logit_differences=differences, hot_or_BF16_generation_equivalence_not_assumed=True,
        all_original_sources_CE_masks_and_auxiliary_heads_preserved=True, provenance=proof,
        full_size_GPUcapacity_or_quality_gain=False)


def test_actual_four_replica_complete_training_post_Adam_failure_recovery_and_standalone_cold(tmp_path):
    spawn(worker, args=(str(tmp_path),), nprocs=4, join=True)
    public = NativeSuffixTrainData.CPU_mechanics(original.ROOT, original.CONTROL)
    public._information[4]['native_execution_eligible'] = False
    records, exports = [], []
    for profile in ['anchor_CE', 'anchor_execution']:
        reports = [json.loads((tmp_path / (profile + '-rank-' + str(rank) + '.json')).read_text()) for rank in range(4)]
        assert len({r['uninterrupted_training']['actual_checkpoint_sha256'] for r in reports}) == 1
        assert len({r['interrupted_then_recovered_training']['actual_checkpoint_sha256'] for r in reports}) == 1
        baseline = export_and_cold(tmp_path, profile, 'complete', public)
        recovered = export_and_cold(tmp_path, profile, 'recovery', public)
        assert baseline['native_parameter_sha256'] == recovered['native_parameter_sha256']
        records.extend(reports); exports.extend([baseline, recovered])
    assert sum(row['actual_small_CPU_AdamW_calls'] for row in records) == 56
    (tmp_path / 'complete.json').write_text(json.dumps(dict(
        actual_physical_small_CPU_AdamW_calls=56, replicas=4, profiles=2,
        reused_closed_capacity_without_new_capacity_optimizer_calls=True,
        exact_uninterrupted_and_post_Adam_failure_recovered_actor_Adam_private_RNG_and_complete_history=True,
        all_full_failed_neural_observations_and_previous_attempt_files_retained=True,
        actual_independent_complete_native_exports=exports,
        full_size_GPUcapacity_or_quality_gain=False, production_updates=0, goal_achieved=False), indent=2))
