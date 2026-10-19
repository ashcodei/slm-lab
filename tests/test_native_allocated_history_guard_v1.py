"""Actual filesystem blocks augment source guards before any new NN sampling."""
import copy
from pathlib import Path

import pytest

from slm_lab.corpus import file_hash
from slm_lab import native_allocated_history_guard_v1 as module
from slm_lab.native_allocated_history_guard_v1 import AllocatedHistoryGuard, inode
from slm_lab.indexed_native_history_v1 import Reader, RECEIPT_LIMIT, canonical
from slm_lab.indexed_native_history_v2 import Writer, read_failure
from test_indexed_native_history_v1 import actual_rows, budgets
from test_native_reference_records_v1 import fixture


def make(path, reference, guard=lambda: None, maximum=None):
    out = Writer(path, reference_binding=reference, starting_completed=0,
        budgets=budgets(maximum_records=3), guard=guard)
    added = AllocatedHistoryGuard(out, maximum_allocated_bytes=maximum or out.budgets['maximum_physical_bytes'],
        source_sha256=file_hash(Path(module.__file__).resolve()))
    out.guard = added
    return out, added


def actual_blocks(root):
    seen, total = set(), root.stat().st_blocks * 512
    for path in root.iterdir():
        identity, size = inode(path)
        if identity not in seen:
            seen.add(identity); total += size
    return total


def test_real_allocated_record_receipt_and_retained_failure_blocks_match_independent_scan(tmp_path):
    rows, reference = actual_rows()
    out, guard = make(tmp_path / 'owned', reference)
    receipts = []
    for row in rows:
        out.reserve_next()
        assert guard.maximum_observed_allocated_bytes <= guard.maximum
        receipts.append(out.append(row))
        out.acknowledge_success(row['step'])
        assert guard.maximum_observed_allocated_bytes >= actual_blocks(out.root)
    guard()
    assert guard.sealed_count == 3
    assert actual_blocks(out.root) == guard.sealed_allocated_bytes + inode(out.root / 'reference.json.gz')[1] + inode(out.failure_slot['path'])[1] + out.root.stat().st_blocks * 512
    assert guard.report()['maximum_observed_funded_peak_bytes'] <= guard.maximum
    assert guard.report()['no_observation_history_retained']
    reader = Reader(out.root, expected_header_sha256=out.header_sha256, maximum_compact_record_bytes=1024**2,
        maximum_expanded_record_bytes=8 * 1024**2, guard=lambda: None)
    assert [canonical(r) for r in reader.records(receipts)] == [canonical(r) for r in rows]


def test_actual_block_quota_rejects_before_source_sampling_even_when_raw_byte_quota_would_fit(tmp_path):
    _, reference = fixture()
    out = Writer(tmp_path / 'strict-physical', reference_binding=reference, starting_completed=0,
        budgets=budgets(maximum_records=3), guard=lambda: None)
    primary = 2 * out.budgets['maximum_compact_record_bytes'] + 2 * RECEIPT_LIMIT + 64 * 1024
    failure = 2 * out.budgets['maximum_expanded_record_bytes'] + 2 * RECEIPT_LIMIT + 64 * 1024
    maximum = actual_blocks(out.root) + primary + failure - 1
    assert out.physical_bytes + primary + failure <= out.budgets['maximum_physical_bytes']
    guard = AllocatedHistoryGuard(out, maximum_allocated_bytes=maximum,
        source_sha256=file_hash(Path(module.__file__).resolve()))
    out.guard = guard
    with pytest.raises(ValueError, match='before sampling'): out.reserve_next()
    assert out.records == 0 and out.ticket is None and out.failure_slot is None
    assert list(out.root.iterdir()) == [out.root / 'reference.json.gz']


def test_added_physical_guard_keeps_original_source_rejection_and_funded_failure_readable(tmp_path):
    row, reference = fixture()
    permitted = True
    def source():
        if not permitted: raise ValueError('The original source guard rejected')
    out, added = make(tmp_path / 'source-rejects', reference, source)
    out.reserve_next()
    permitted = False
    with pytest.raises(ValueError, match='original source'): out.append(row)
    failure = out.preserve_failure(dict(original_complete_observation=row, source_error='rejected before Adam'))
    readback = read_failure(out.root, failure, maximum_bytes=8 * 1024**2)
    assert canonical(readback['original_observation']['original_complete_observation']) == canonical(row)
    assert added.report()['no_original_source_guard_removed']


def test_guard_never_retargets_an_existing_writer_or_changes_original_binding(tmp_path):
    row, reference = fixture()
    out, added = make(tmp_path / 'original', reference)
    original = copy.deepcopy(out.codec.reference)
    out.reserve_next(); out.append(row); out.acknowledge_success(1)
    with pytest.raises(ValueError):
        AllocatedHistoryGuard(out, maximum_allocated_bytes=out.budgets['maximum_physical_bytes'],
            source_sha256=file_hash(Path(module.__file__).resolve()))
    assert out.codec.reference == original and out.guard is added
