"""Real allocated files and exact closed-record readback; no neural admission."""
import copy
import gzip
import json
from pathlib import Path

import pytest

from slm_lab.indexed_native_history_v1 import Writer, Reader, canonical, sha, RECEIPT_LIMIT
from test_native_reference_records_v1 import fixture, resident
from test_factored_native_observation_stream_v1 import JOURNAL


def actual_rows():
    _, reference = fixture()
    with gzip.open(JOURNAL, 'rt') as stream:
        rows = [json.loads(line)['actual_complete_update'] for line in stream]
    for row in rows:
        for observation in row['actual_rank_observations']:
            if observation['kernel'] is not None:
                observation['kernel']['own_reference_binding'] = copy.deepcopy(reference)
    return rows, reference


def budgets(**changes):
    value = dict(maximum_records=32, maximum_compact_record_bytes=1024**2,
        maximum_expanded_record_bytes=8 * 1024**2, maximum_logical_bytes=512 * 1024**2,
        maximum_physical_bytes=64 * 1024**2, minimum_free_bytes=0)
    value.update(changes)
    return value


def writer(path, reference, **options):
    return Writer(path, reference_binding=reference, starting_completed=0,
        budgets=options.pop('budgets', budgets()), guard=options.pop('guard', lambda: None), **options)


def reader(out):
    return Reader(out.root, expected_header_sha256=out.header_sha256,
        maximum_compact_record_bytes=1024**2, maximum_expanded_record_bytes=8 * 1024**2, guard=lambda: None)


def test_actual_complete_records_reserved_blocks_exact_streaming_and_checkpoint_prefix(tmp_path):
    rows, reference = actual_rows()
    out = writer(tmp_path / 'complete', reference)
    receipts = []
    for row in rows:
        allocation = out.reserve_next()
        reserved = Path(allocation['path'])
        assert reserved.stat().st_blocks * 512 >= allocation['allocation_bytes']
        assert out.records == len(receipts)
        receipts.append(out.append(row))
        assert not reserved.exists() and out.ticket is None
    assert out.records == 3
    assert out.logical_bytes == len(canonical(reference)) + sum(len(canonical(r)) for r in rows)
    assert [canonical(r) for r in reader(out).records(receipts)] == [canonical(r) for r in rows]
    assert [canonical(r) for r in reader(out).records(receipts[:2])] == [canonical(r) for r in rows[:2]]
    assert [o['kernel'] is None for o in list(reader(out).records(receipts))[-1]['actual_rank_observations']] == [False, True, True, True]
    shared_reader = reader(out)
    shared = next(shared_reader.records(receipts, share_reference=True))
    assert all(o['kernel']['own_reference_binding'] is shared_reader.codec.reference
        for o in shared['actual_rank_observations'] if o['kernel'] is not None)
    assert all(len(canonical(r)) < RECEIPT_LIMIT for r in receipts)


@pytest.mark.parametrize('quota', ['logical', 'physical'])
def test_complete_worst_next_record_quota_rejects_before_allocating_or_sampling(tmp_path, quota):
    _, reference = fixture()
    options = budgets(maximum_logical_bytes=len(canonical(reference)) + 8 * 1024**2 - 1) if quota == 'logical' else budgets(maximum_physical_bytes=2 * 1024**2)
    out = writer(tmp_path / quota, reference, budgets=options)
    with pytest.raises(ValueError, match='before sampling'): out.reserve_next()
    assert out.records == 0 and out.ticket is None
    assert list(out.root.glob('record-*')) == []


def test_actual_reservation_keeps_metadata_headroom_outside_allocated_file_and_rechecks_floor(tmp_path, monkeypatch):
    import slm_lab.indexed_native_history_v1 as module
    actual_wait = module.wait_for_reserve
    calls = []
    def observed_wait(root, **options):
        calls.append(options['minimum_bytes'])
        return actual_wait(root, **options)
    monkeypatch.setattr(module, 'wait_for_reserve', observed_wait)
    row, reference = fixture()
    out = writer(tmp_path / 'floor', reference, budgets=budgets(minimum_free_bytes=1024**2))
    ticket = out.reserve_next()
    assert calls == [1024**2 + ticket['allocation_bytes'] + 64 * 1024, 1024**2 + 64 * 1024]
    assert Path(ticket['path']).stat().st_blocks * 512 >= ticket['allocation_bytes']
    receipt = out.append(row)
    assert canonical(next(reader(out).records([receipt]))) == canonical(row)


def test_missing_reservation_incomplete_crash_and_existing_attempt_are_preserved(tmp_path):
    row, reference = fixture()
    out = writer(tmp_path / 'interrupted', reference)
    with pytest.raises(ValueError, match='pre-sampling reservation'): out.append(row)
    ticket = out.reserve_next()
    path = Path(ticket['path'])
    original_stat = path.stat()
    header = (out.root / 'reference.json.gz').read_bytes()
    with pytest.raises(FileExistsError): writer(out.root, reference)
    assert path.stat().st_ino == original_stat.st_ino and path.stat().st_blocks == original_stat.st_blocks
    assert (out.root / 'reference.json.gz').read_bytes() == header
    assert list(reader(out).records([])) == []
    with pytest.raises(ValueError): out.reserve_next()


def test_complete_observation_remains_on_disk_if_guard_fails_after_publication(tmp_path):
    row, reference = fixture()
    root = tmp_path / 'late-guard-fault'
    def guard():
        if list(root.glob('record-*.json.gz')):
            raise ValueError('Injected late owning-source rejection')
    out = writer(root, reference, guard=guard)
    ticket = out.reserve_next()
    with pytest.raises(ValueError, match='Injected late'): out.append(row)
    assert out.failed and out.records == 0
    reserved = Path(ticket['path'])
    published = root / 'record-000000000001.json.gz'
    assert reserved.exists() and reserved.stat().st_ino == published.stat().st_ino
    assert canonical(out.codec.pack(row)) == gzip.decompress(published.read_bytes())
    assert (root / 'record-000000000001.receipt.json').is_file()
    with pytest.raises(ValueError): out.reserve_next()
    with pytest.raises(ValueError): out.append(row)


def test_invalid_observation_type_or_reference_keeps_funded_attempt_and_never_admits_record(tmp_path):
    row, reference = fixture()
    for fault in ['tuple', 'changed_reference', 'wrong_step']:
        out = writer(tmp_path / fault, reference)
        ticket = out.reserve_next()
        bad = copy.deepcopy(row)
        if fault == 'tuple': bad['unsupported_original_tuple'] = (1, 2)
        elif fault == 'changed_reference':
            bad['actual_rank_observations'][2]['kernel']['own_reference_binding']['external_pretrained_teacher'] = 0
        else: bad['step'] = True
        with pytest.raises(ValueError): out.append(bad)
        assert out.failed and out.records == 0 and Path(ticket['path']).exists()
        assert not list(out.root.glob('*.receipt.json'))


def test_changed_header_payload_receipt_and_missing_or_reordered_prefix_fail_closed(tmp_path):
    rows, reference = actual_rows()
    out = writer(tmp_path / 'original', reference)
    receipts = []
    for row in rows:
        out.reserve_next()
        receipts.append(out.append(row))
    original = reader(out)
    for bad in [receipts[1:], receipts[::-1], [receipts[0], receipts[0]]]:
        with pytest.raises(ValueError, match='prefix record'): list(original.records(bad))
    wrong = copy.deepcopy(receipts[0])
    wrong['record'] = '../outside.json.gz'
    with pytest.raises(ValueError, match='location'): original.compact(wrong)
    wrong = copy.deepcopy(receipts[0])
    wrong['original_expanded_bytes'] += 1
    with pytest.raises(ValueError, match='receipt byte'): original.compact(wrong)
    record = out.root / receipts[0]['record']
    raw = record.read_bytes()
    record.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    with pytest.raises(ValueError, match='compressed observation'): original.compact(receipts[0])
    header = out.root / 'reference.json.gz'
    raw_header = header.read_bytes()
    header.write_bytes(raw_header[:-1] + bytes([raw_header[-1] ^ 1]))
    with pytest.raises(ValueError, match='header changed'): original.compact(receipts[1])
    with pytest.raises(ValueError, match='header SHA'): reader(out)


def test_source_shaped_history_retains_only_small_receipts_and_reads_one_record_at_a_time(tmp_path):
    row, reference = fixture()
    out = writer(tmp_path / 'source-only-stress', reference)
    receipts, expected = [], []
    first_resident = None
    for index in range(16):
        # Counter changes are explicitly transport stress, not new neural calls.
        row['step'] = index + 1
        out.reserve_next()
        receipts.append(out.append(row))
        expected.append(sha(canonical(row)))
        if first_resident is None: first_resident = resident((out.__dict__, out.codec.__dict__))
    final_resident = resident((out.__dict__, out.codec.__dict__))
    assert final_resident >= resident(out.codec.reference) + len(out.codec.original_bytes)
    assert final_resident - first_resident < 64 * 1024
    assert resident(receipts) < 16 * resident(out.codec.pack(row)) / 100
    assert [sha(canonical(r)) for r in reader(out).records(receipts)] == expected
    proof = dict(actual_closed_binding_count=len(reference['sources']), actual_individual_durable_records=16,
        reference_header_stored_once=True, actual_receipt_resident_bytes=resident(receipts),
        actual_writer_and_immutable_codec_fields_resident_bytes_after_first=first_resident,
        actual_writer_and_immutable_codec_fields_resident_bytes_after_16=final_resident,
        resident_measurement_scope='Recursive primitive/container size of writer and codec dictionaries; opaque Path/guard internals and interpreter/globals excluded',
        prior_control_dictionary_only_measurement_excluded_codec_fields=True,
        logical_original_bytes_including_every_repeated_reference=out.logical_bytes,
        compressed_and_receipt_file_bytes=out.physical_bytes,
        every_complete_original_byte_reconstructed=True, real_pre_sample_block_reservations_checked=True,
        source_shaped_transport_stress_only=True, source_counter_changes_not_new_neural_samples=True,
        original_closed_small_model_observations=True, full_size_GPUcapacity_or_quality=False, new_neural_updates=0)
    (tmp_path / 'actual-indexed-history-proof.json').write_text(json.dumps(proof, indent=2))
