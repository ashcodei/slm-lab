"""Funded real files preserve complete observations across owning-guard faults."""
import copy
import json
from pathlib import Path
import struct

import pytest
import torch

from slm_lab.codenet_checkpoint_v1 import capture_rng
from slm_lab.indexed_native_history_v2 import Writer, read_failure, observation_digest, file_sha
from slm_lab.indexed_native_history_v1 import Reader, canonical
from slm_lab.parallel_function_repair_stage_v1 import value_digest
from test_indexed_native_history_v1 import actual_rows, budgets
from test_native_reference_records_v1 import fixture


def writer(path, reference, guard=lambda: None, **changes):
    return Writer(path, reference_binding=reference, starting_completed=0, budgets=budgets(**changes), guard=guard)


def reader(out):
    return Reader(out.root, expected_header_sha256=out.header_sha256,
        maximum_compact_record_bytes=1024**2, maximum_expanded_record_bytes=8 * 1024**2, guard=lambda: None)


def failure_observation(row):
    nan = struct.unpack('>d', bytes.fromhex('7ff8000000000001'))[0]
    return dict(phase='actual four-rank gradient/verifier failure before commit',
        actual_rank_errors=[None, None, dict(complete_original_observation=row,
            raw_worker_output=b'original\x00\xffbytes', nonfinite=[nan, -0.0, float('inf'), -float('inf')],
            actual_CPU_tensor=torch.tensor([nan, -0.0, 3.], dtype=torch.float64, requires_grad=True),
            actual_bfloat_tensor=torch.tensor([1., -2.], dtype=torch.bfloat16),
            original_collection_types=(True, [1, 2], None)), None])


def test_complete_typed_nonfinite_failure_survives_rejecting_source_guard_and_RNG_is_unchanged(tmp_path):
    row, reference = fixture()
    valid = True
    def guard():
        if not valid: raise ValueError('Injected actual owning-source rejection')
    out = writer(tmp_path / 'failed', reference, guard)
    ticket = out.reserve_next()
    assert ticket['failure_allocated_bytes'] >= out.failure_slot['allocation_bytes']
    assert out.ticket['path'].stat().st_ino != out.failure_slot['path'].stat().st_ino
    evidence = failure_observation(row)
    before = value_digest(capture_rng('cpu'))
    valid = False
    with pytest.raises(ValueError, match='owning-source'): out.check()
    receipt = out.preserve_failure(evidence)
    actual = read_failure(out.root, receipt, maximum_bytes=8 * 1024**2)
    assert observation_digest(actual['original_observation']) == observation_digest(evidence)
    assert list(actual['original_observation']) == list(evidence)
    restored = actual['original_observation']['actual_rank_errors'][2]
    assert struct.pack('>d', restored['nonfinite'][0]).hex() == '7ff8000000000001'
    assert struct.pack('>d', restored['nonfinite'][1]).hex() == '8000000000000000'
    assert type(restored['original_collection_types']) is tuple
    assert restored['raw_worker_output'] == b'original\x00\xffbytes'
    assert any(m['requires_grad'] for m in actual['original_tensor_metadata'])
    assert all(m['device'] == 'cpu' for m in actual['original_tensor_metadata'])
    assert not actual['successful_optimizer_commit_certified'] and out.failed
    assert value_digest(capture_rng('cpu')) == before
    assert out.records == 0 and out.ticket['path'].exists()
    with pytest.raises(ValueError): out.reserve_next()
    with pytest.raises(ValueError): out.append(row)
    saved = file_sha(out.root / receipt['file'])
    with pytest.raises(ValueError): out.preserve_failure(evidence)
    assert file_sha(out.root / receipt['file']) == saved
    (tmp_path / 'actual-complete-typed-failure-proof.json').write_text(json.dumps(dict(
        actual_closed_source_binding_count=len(reference['sources']),
        complete_actual_small_CPU_observation_retained_with_real_source_map=True,
        deliberate_nonfinite_tensor_and_float_bits_collection_types_and_raw_bytes_exact=True,
        owning_source_guard_rejects_but_funded_evidence_remains_readable=True,
        one_independent_real_failure_slot_funded_before_sampling=True,
        all_actual_global_RNG_unchanged=True, actual_failure_receipt=receipt,
        new_neural_updates=0, production_GPU_capacity_or_quality_gain=False), indent=2))


def test_post_publication_guard_fault_cannot_overwrite_original_complete_record(tmp_path):
    row, reference = fixture()
    root = tmp_path / 'post-record'
    def guard():
        if (root / 'record-000000000001.json.gz').exists():
            raise ValueError('Injected guard rejection after complete record publication')
    out = writer(root, reference, guard)
    out.reserve_next()
    with pytest.raises(ValueError, match='after complete record'): out.append(row)
    record = root / 'record-000000000001.json.gz'
    original_sha = file_sha(record)
    receipt = out.preserve_failure(dict(actual_complete_observation=row, actual_error='source guard rejected'))
    assert file_sha(record) == original_sha
    assert out.failure_slot['path'].stat().st_ino != record.stat().st_ino
    actual = read_failure(root, receipt, maximum_bytes=8 * 1024**2)
    assert canonical(actual['original_observation']['actual_complete_observation']) == canonical(row)
    assert out.ticket['path'].stat().st_ino == record.stat().st_ino
    assert out.records == 0


def test_actual_three_success_records_require_model_acknowledgement_and_reuse_one_funded_slot(tmp_path):
    rows, reference = actual_rows()
    out = writer(tmp_path / 'complete', reference, maximum_records=len(rows))
    receipts, inodes = [], []
    for row in rows:
        out.reserve_next()
        inodes.append(out.failure_slot['path'].stat().st_ino)
        receipts.append(out.append(row))
        with pytest.raises(ValueError): out.reserve_next()
        with pytest.raises(ValueError): out.acknowledge_success(row['step'] + 1)
        out.acknowledge_success(row['step'])
    assert len(set(inodes)) == 1
    assert [canonical(r) for r in reader(out).records(receipts)] == [canonical(r) for r in rows]
    assert [o['kernel'] is None for o in rows[-1]['actual_rank_observations']] == [False, True, True, True]
    assert out.failure_slot['path'].stat().st_blocks * 512 >= out.failure_slot['allocation_bytes']
    assert not out.failed and out.failure_artifact is None and out.records == 3
    with pytest.raises(ValueError): out.reserve_next()


def test_success_observation_then_actual_Adam_rejection_preserves_both_original_artifacts(tmp_path):
    row, reference = fixture()
    out = writer(tmp_path / 'post-Adam-fault', reference)
    out.reserve_next()
    success = out.append(row)
    original_sha = file_sha(out.root / success['record'])
    failure = out.preserve_failure(dict(observation_boundary='after actual attempted Adam, before successful all-peer acknowledgement',
        complete_original_observation=row, actual_error='Injected invalid peer clock'))
    assert file_sha(out.root / success['record']) == original_sha
    assert canonical(next(reader(out).records([success]))) == canonical(row)
    assert read_failure(out.root, failure, maximum_bytes=8 * 1024**2)['successful_optimizer_commit_certified'] is False
    with pytest.raises(ValueError): out.acknowledge_success(1)


@pytest.mark.parametrize('quota', ['physical', 'logical'])
def test_both_success_and_failure_worst_cases_must_fit_before_any_source_sampling(tmp_path, quota):
    _, reference = fixture()
    options = dict(maximum_physical_bytes=16 * 1024**2) if quota == 'physical' else dict(maximum_logical_bytes=16 * 1024**2)
    out = writer(tmp_path / quota, reference, **options)
    with pytest.raises(ValueError, match='before sampling'): out.reserve_next()
    assert out.failure_slot is None and out.ticket is None and out.records == 0
    assert not list(out.root.glob('*.reserved'))


def test_corrupt_or_forged_failure_payload_receipt_and_boundary_cannot_admit_state(tmp_path):
    row, reference = fixture()
    out = writer(tmp_path / 'corrupt', reference)
    out.reserve_next()
    receipt = out.preserve_failure(dict(complete_actual_row=row))
    for field, value in [('successful_optimizer_commit_certified', True), ('file', '../other.pt'), ('actual_bytes', True)]:
        wrong = dict(receipt, **{field: value})
        with pytest.raises(ValueError): read_failure(out.root, wrong, maximum_bytes=8 * 1024**2)
    wrong = dict(receipt, original_typed_observation_sha256='0' * 64)
    with pytest.raises(ValueError): read_failure(out.root, wrong, maximum_bytes=8 * 1024**2)
    path = out.root / receipt['file']
    with path.open('r+b') as stream:
        first = stream.read(1);stream.seek(0);stream.write(bytes([first[0] ^ 1]))
    with pytest.raises(ValueError, match='payload byte'): read_failure(out.root, receipt, maximum_bytes=8 * 1024**2)
