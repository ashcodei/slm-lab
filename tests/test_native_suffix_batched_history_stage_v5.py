"""Every original record still validates; redundant source reads are batched."""
import copy
import gzip
import json
import os
from pathlib import Path
import threading

import pytest

from slm_lab.corpus import file_hash
from slm_lab.indexed_native_history_v2 import Writer
from slm_lab.native_suffix_batched_history_stage_v5 import NativeSuffixBatchedHistoryStageV5, PROTOCOL
from slm_lab.native_suffix_indexed_stage_v4 import NativeSuffixIndexedStageV4, HISTORY
from slm_lab.native_suffix_train_data_v1 import NativeSuffixTrainData
from slm_lab.parallel_function_repair_stage_v1 import value_digest
from test_native_suffix_capacity_controller_v2 import student
from test_native_suffix_indexed_stage_v4 import BUDGETS, sources as indexed_sources
from test_public_reasoning_group_gradients_v1 import optimizer
import test_parallel_native_suffix_stage_v1 as original

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'reports/native-indexed-stage-v4-smoke-pytest/test_actual_single_replica_pre0/attempt1'


def sources():
    result = indexed_sources()
    for relative in ['src/slm_lab/native_suffix_batched_history_stage_v5.py', 'tests/test_native_suffix_batched_history_stage_v5.py']:
        path = ROOT / relative
        result[str(path)] = file_hash(path)
    return result


@pytest.fixture(scope='module')
def public(): return NativeSuffixTrainData.CPU_mechanics(original.ROOT, original.CONTROL)


def stage(public, cls=NativeSuffixBatchedHistoryStageV5):
    own = student(public)
    return cls(own, optimizer(own), public, [[0], [1], [2]], 'anchor_execution', original.OPTIONS, sources(),
        mode='mechanics', history_budgets=BUDGETS)


def complete_record(out, root):
    with gzip.open(OLD / 'record-000000000001.json.gz', 'rt') as stream: compact = json.load(stream)
    with gzip.open(OLD / 'reference.json.gz', 'rt') as stream: old_header = json.load(stream)
    actual = copy.deepcopy(compact)
    for observation in actual['actual_rank_observations']:
        binding = old_header['reference_binding']
        assert value_digest({k:v for k,v in binding.items() if k != 'sources'}) == value_digest({k:v for k,v in out.reference_binding.items() if k != 'sources'})
        if observation['kernel'] is not None: observation['kernel']['own_reference_binding'] = copy.deepcopy(out.reference_binding)
    writer = Writer(root, reference_binding=out.reference_binding, starting_completed=0, budgets=BUDGETS, guard=lambda: None)
    writer.reserve_next(); receipt = writer.append(actual); writer.acknowledge_success(1)
    return actual, dict(format=HISTORY, directory=str(root), header_sha256=writer.header_sha256, record=receipt)


def test_every_original_field_is_revalidated_while_source_checks_move_to_actual_batch_boundaries(public, tmp_path):
    base, candidate = stage(public, NativeSuffixIndexedStageV4), stage(public)
    row, receipt = complete_record(candidate, tmp_path / 'source-only-normalized-history')
    assert base.reference_binding == candidate.reference_binding
    original_check = public.guard.check
    calls = []
    def observed_check(*args, **kwargs):
        calls.append(1); return original_check(*args, **kwargs)
    public.guard.check = observed_check
    try:
        base.validate_records([receipt]); base_calls = len(calls)
        calls.clear(); candidate.validate_records([receipt]); candidate_calls = len(calls)
        assert base_calls > candidate_calls == 2
        calls.clear(); candidate.validate_records([receipt]); assert len(calls) == 2
        public._information[0]['original_response_EOS_targets'] += 1
        try:
            with pytest.raises(ValueError): candidate.validate_records([receipt])
        finally: public._information[0]['original_response_EOS_targets'] -= 1
    finally: public.guard.check = original_check
    assert candidate._shared_history_batch is None and candidate._validation_owner is None
    assert not candidate.optimizer.state and candidate.completed == 0
    (tmp_path / 'actual-full-record-source-boundary-proof.json').write_text(json.dumps(dict(
        actual_original_record_validation_passed_in_both_stages=True,
        source_only_reference_metadata_rebound_to_identical_new_code_sources=True,
        actual_original_full_data_guard_calls=base_calls, actual_batched_full_data_guard_calls=candidate_calls,
        all_records_revalidated_on_every_call=True, changed_in_memory_source_metadata_still_rejected=True,
        no_history_or_semantic_verdict_cache=True, protocol=PROTOCOL,
        original_closed_small_CPU_observation_only=True, new_optimizer_calls=0,
        full_size_GPU_throughput_or_quality_not_measured=True), indent=2))


def test_same_size_same_mtime_source_change_is_rejected_at_batch_end(public, tmp_path):
    own = stage(public)
    path = tmp_path / 'additional-bound-source.txt';path.write_bytes(b'original bound bytes')
    public.guard.files[path] = file_hash(path);public.guard.states[path] = public.guard._state(path)
    old = path.stat()
    try:
        with pytest.raises(ValueError):
            with own.checked_validation_sources():
                path.write_bytes(b'changed  bound bytes')
                os.utime(path, ns=(old.st_atime_ns, old.st_mtime_ns))
                own.history_guard()
        assert own._shared_history_batch is None and own._validation_owner is None
        assert not own.optimizer.state
    finally:
        public.guard.files.pop(path);public.guard.states.pop(path)


def test_changed_source_binding_map_cannot_escape_an_existing_checked_batch(public, tmp_path):
    own = stage(public)
    path = tmp_path / 'additional-bound-source.txt';path.write_text('a real private input')
    public.guard.files[path] = file_hash(path);public.guard.states[path] = public.guard._state(path)
    digest = public.guard.files[path]
    try:
        with pytest.raises(ValueError):
            with own.checked_validation_sources(): public.guard.files.pop(path)
        assert own._shared_history_batch is None
    finally:
        public.guard.files.pop(path, None);public.guard.states.pop(path)


def test_independent_thread_cannot_borrow_checked_batch_and_outside_guard_stays_active(public):
    own = stage(public)
    errors = []
    def attempted():
        try: own.history_guard()
        except ValueError as exc: errors.append(str(exc))
    with own.checked_validation_sources():
        thread = threading.Thread(target=attempted);thread.start();thread.join()
        assert len(errors) == 1
    before = value_digest(own.reference_binding)
    own.history_guard()
    assert value_digest(own.reference_binding) == before and not own.optimizer.state
