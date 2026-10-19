"""Exact transport/history checks using closed small-neural observations.

The real production source map is used for storage geometry only. These
records are not new full-size model samples or evidence of model quality.
"""
import copy
import gzip
import json
import pickle
import sys

import pytest

from slm_lab.native_reference_records_v1 import NativeReferenceRecords, canonical
from slm_lab.parallel_function_repair_stage_v1 import value_digest
from test_factored_native_observation_stream_v1 import fixture as wrapped_fixture, JOURNAL


def fixture():
    envelope, reference = wrapped_fixture()
    return envelope['actual_complete_update'], reference


def resident(value, seen=None):
    seen = set() if seen is None else seen
    if id(value) in seen: return 0
    seen.add(id(value))
    total = sys.getsizeof(value)
    if isinstance(value, dict):
        total += sum(resident(k, seen) + resident(v, seen) for k, v in value.items())
    elif isinstance(value, (list, tuple)):
        total += sum(resident(v, seen) for v in value)
    return total


def test_every_original_record_byte_and_empty_rank_survives_factoring():
    row, reference = fixture()
    codec = NativeReferenceRecords(reference)
    with gzip.open(JOURNAL, 'rt') as stream:
        tail = list(map(json.loads, stream))[-1]['actual_complete_update']
    assert tail['actual_original_source_positions'] == [8]
    for observed in tail['actual_rank_observations']:
        if observed['kernel'] is not None:
            observed['kernel']['own_reference_binding'] = copy.deepcopy(reference)
    for original in [row, tail]:
        wire = pickle.loads(pickle.dumps(codec.pack(original), protocol=2))
        assert canonical(codec.expand(wire)) == canonical(original)
        assert value_digest(codec.expand(wire)) == value_digest(original)
        shared = codec.expand(wire, share_reference=True)
        assert all(o['kernel']['own_reference_binding'] is codec.reference
            for o in shared['actual_rank_observations'] if o['kernel'] is not None)
        defensive = codec.expand(wire)
        defensive['actual_rank_observations'][0]['kernel']['own_reference_binding']['sources'].clear()
        codec.check(reference)
        assert canonical(codec.expand(wire)) == canonical(original)
    assert [o['kernel'] is None for o in codec.expand(codec.pack(tail))['actual_rank_observations']] == [False, True, True, True]


def test_changed_reference_values_types_order_and_marker_are_rejected():
    row, reference = fixture()
    codec = NativeReferenceRecords(reference)
    for mutation in ['value', 'type', 'order']:
        wrong = copy.deepcopy(row)
        binding = wrong['actual_rank_observations'][2]['kernel']['own_reference_binding']
        if mutation == 'value': binding['sources'][next(iter(binding['sources']))] = '0' * 64
        elif mutation == 'type': binding['external_pretrained_teacher'] = 0
        else: binding['sources'] = dict(reversed(list(binding['sources'].items())))
        with pytest.raises(ValueError): codec.pack(wrong)
    compact = codec.pack(row)
    for mutation in ['sha', 'extra', 'missing']:
        wrong = copy.deepcopy(compact)
        marker = wrong['actual_rank_observations'][2]['kernel']['own_reference_binding']
        if mutation == 'sha': marker['sha256'] = '0' * 64
        elif mutation == 'extra': marker['silently_added'] = True
        else: marker.pop('format')
        with pytest.raises(ValueError): codec.expand(wrong)
    codec.reference['sources'].clear()
    with pytest.raises(ValueError): codec.check(reference)


@pytest.mark.parametrize('wrong', [
    dict(rank=0, actual_original_positions=[8], kernel=None),
    dict(rank=True, actual_original_positions=[], kernel=None),
    dict(rank=0, actual_original_positions=[], kernel=None, omitted_other_field=False),
    dict(rank=0, actual_original_positions=(), kernel=None),
])
def test_nonempty_sources_and_actual_rank_layout_cannot_disappear(wrong):
    _, reference = fixture()
    codec = NativeReferenceRecords(reference)
    with pytest.raises(ValueError): codec.pack_observation(wrong)
    with pytest.raises(ValueError): codec.expand_observation(wrong)


def test_independent_collective_roundtrips_do_not_replicate_full_map_per_history_row(tmp_path):
    row, reference = fixture()
    codec = NativeReferenceRecords(reference)
    def actual_rank_collective():
        result = copy.deepcopy(row)
        result['actual_rank_observations'] = [pickle.loads(pickle.dumps(o, protocol=2))
            for o in row['actual_rank_observations']]
        return result
    old_one = actual_rank_collective()
    old_two = [old_one, actual_rank_collective()]
    old_increment = resident(old_two) - resident([old_one])
    packed = codec.pack(old_one)
    records = [pickle.loads(pickle.dumps(packed, protocol=2)) for _ in range(32)]
    compact_increment = resident(records[:2]) - resident(records[:1])
    assert compact_increment < old_increment / 4
    assert len(pickle.dumps(packed, protocol=2)) < len(pickle.dumps(old_one, protocol=2)) / 4
    assert all(canonical(codec.expand(record)) == canonical(row) for record in records)
    proof = dict(actual_closed_binding_count=len(reference['sources']), actual_independent_history_records=32,
        old_actual_second_record_resident_bytes=old_increment,
        compact_actual_second_record_resident_bytes=compact_increment,
        original_actual_protocol2_record_bytes=len(pickle.dumps(old_one, protocol=2)),
        compact_actual_protocol2_record_bytes=len(pickle.dumps(packed, protocol=2)),
        one_shared_reference_resident_bytes=resident(codec.reference),
        actual_compact_32_record_resident_bytes=resident(records),
        every_reconstructed_original_byte_equal=True, original_fixture_only=True,
        source_map_substitution_for_transport_geometry_only=True,
        production_memory_or_capacity_measured=False, new_optimizer_updates=0, new_quality_gain=False)
    (tmp_path / 'actual-compact-history-geometry.json').write_text(json.dumps(proof, indent=2))
