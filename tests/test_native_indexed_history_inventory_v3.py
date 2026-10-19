"""Read the real post-Adam failure; preserve strict healthy-slot funding."""
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from slm_lab.corpus import file_hash
from slm_lab.indexed_native_history_v1 import Reader, canonical
from slm_lab.native_indexed_history_inventory_v3 import history_inventory
from slm_lab.native_history_summary_v1 import Summary
from slm_lab.parallel_function_repair_stage_v1 import value_digest
from slm_lab.parallel_repair_durable_state_v1 import read_latest
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
from test_indexed_native_history_v2 import writer
from test_native_reference_records_v1 import fixture

ROOT = Path(__file__).resolve().parents[1]
FAILED = ROOT / 'reports/native-indexed-training-v2-proof-v1/pytest/test_actual_four_replica_compl0/anchor_CE/recovery'


def adapter(envelopes, active=None):
    def reader(directory, header):
        return Reader(directory, expected_header_sha256=header, maximum_compact_record_bytes=16 * 1024**2,
            maximum_expanded_record_bytes=64 * 1024**2, guard=lambda: None)
    return SimpleNamespace(metrics=envelopes, history_reader=active, history_file_reader=reader)


def test_actual_complete_recovered_state_with_consumed_post_Adam_failure_preserves_every_file(tmp_path):
    state, _, _ = read_latest(FAILED / 'private')
    baseline, _, _ = read_latest(FAILED.parent / 'complete/private')
    assert state['completed'] == 3 and len({m['directory'] for m in state['metrics']}) == 2
    paths = [p for folder in {Path(m['directory']) for m in state['metrics']} for p in folder.iterdir()]
    before = {str(p): file_hash(p) for p in paths}
    inventory = history_inventory(adapter(state['metrics']))
    assert len(inventory) == 2
    consumed = next(row for row in inventory if row['actual_consumed_failure']['actual_attempted_step'] == 2)
    # The rejected final inventory itself also retained its complete last
    # neural row in the resumed attempt's slot. Both terminal attempts are
    # consumed; the separate uninterrupted attempt is still fully funded.
    assert sorted(row['actual_consumed_failure']['actual_attempted_step'] for row in inventory) == [2, 3]
    healthy = history_inventory(adapter(baseline['metrics']))[0]
    assert healthy['actual_consumed_failure'] is None
    assert consumed['actual_consumed_failure']['actual_attempted_step'] == 2
    assert consumed['actual_consumed_failure']['original_complete_failure_readback_passed']
    assert consumed['actual_consumed_failure']['successful_optimizer_commit_certified'] is False
    assert consumed['actual_independent_failure_slot_allocated_bytes'] < healthy['actual_independent_failure_slot_allocated_bytes']
    for row in inventory:
        directory = Path(row['directory']); seen = {}
        for path in directory.iterdir():
            stat = path.stat(); seen[(stat.st_dev, stat.st_ino)] = stat.st_blocks * 512
        assert row['actual_unique_inode_allocated_bytes'] == directory.stat().st_blocks * 512 + sum(seen.values())
    assert all(file_hash(path) == digest for path, digest in before.items())
    assert tensor_digest(state['model']['parts'].items()) == tensor_digest(baseline['model']['parts'].items())
    assert value_digest(state['optimizer']) == value_digest(baseline['optimizer'])
    assert [value_digest(w['rng']) for w in state['private_workers']] == [value_digest(w['rng']) for w in baseline['private_workers']]
    def summary(saved):
        actual = Summary(group_size=8); readers = {}
        for envelope in saved['metrics']:
            key = envelope['directory'], envelope['header_sha256']
            if key not in readers: readers[key] = adapter([]).history_file_reader(*key)
            reader = readers[key]
            actual.append(reader.codec.expand(reader.compact(envelope['record']), share_reference=True))
        return actual.result()
    assert summary(state) == summary(baseline)
    (tmp_path / 'actual-closed-consumed-failure-inventory-proof.json').write_text(json.dumps(dict(
        complete=True, actual_terminal_consumed_failure_steps=[2, 3], actual_healthy_attempts=1,
        actual_closed_CE_actor_Adam_all4_private_RNG_and_every_mathematical_history_record_equal=True,
        every_original_failure_and_record_byte_unchanged=True,
        previously_failed_controller_still_failed_not_relabelled=True,
        new_neural_optimizer_calls=0, full_size_GPUcapacity_or_quality_gain=False), indent=2))


@pytest.mark.parametrize('kind', ['healthy', 'depleted_unused', 'changed_payload', 'missing_receipt',
    'detached_original_link', 'foreign_header', 'unknown_file'])
def test_consumed_failure_never_weakens_live_funding_or_original_typed_evidence(tmp_path, kind):
    row, reference = fixture()
    out = writer(tmp_path / 'attempt', reference)
    out.reserve_next()
    if kind not in ['healthy', 'depleted_unused']:
        receipt = out.preserve_failure(dict(complete_original_neural_row=row, actual_error='controlled test failure'))
    if kind == 'depleted_unused':
        with out.failure_slot['path'].open('r+b') as stream: stream.truncate(4096)
    elif kind == 'changed_payload':
        with (out.root / receipt['file']).open('r+b') as stream:
            stream.seek(receipt['actual_bytes'] // 2); stream.write(b'\xff')
    elif kind == 'missing_receipt': (out.root / 'failure-step-000000000001.receipt.json').unlink()
    elif kind == 'detached_original_link':
        old = out.failure_slot['path']; replacement = tmp_path / 'separate-copy'
        shutil.copyfile(old, replacement); old.unlink(); replacement.rename(old)
    elif kind == 'foreign_header':
        receipt['expected_reference_header_sha256'] = 'e' * 64
        (out.root / 'failure-step-000000000001.receipt.json').write_bytes(canonical(receipt))
    elif kind == 'unknown_file': (out.root / 'foreign-owned-data').write_bytes(b'not part of this attempt')
    stage = adapter([], active=SimpleNamespace(root=out.root, header_sha256=out.header_sha256))
    if kind == 'healthy':
        actual = history_inventory(stage)
        assert actual[0]['actual_independent_failure_slot_disposition'] == 'unused-fully-funded'
        assert actual[0]['actual_independent_failure_slot_allocated_bytes'] >= out.failure_slot['allocation_bytes']
    else:
        with pytest.raises(ValueError): history_inventory(stage)
