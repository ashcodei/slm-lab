"""Actual public source records and full-source schedules; no model training."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

import pytest
import torch

from slm_lab.bounded_gzip_observation_stream_v1 import lines
from slm_lab.indexed_gzip_records_v1 import Writer, Records, ENTRY
from slm_lab.native_suffix_train_data_v1 import NativeSuffixTrainData, schedule_source_views

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / 'reports/native-suffix-CPU-mechanics-qualification-v1/closed-qualified-controls.json'


def write(folder, values, maximum=16 * 1024**2):
    calls = []
    def reserve(root, minimum_bytes, guard): calls.append(minimum_bytes); guard()
    with Writer(folder / 'pack.gz', folder / 'index', root=folder, maximum_raw_bytes=maximum,
            minimum_free_bytes=0, guard=lambda: None, reserve=reserve) as writer:
        for row in values: writer.append(row)
    return writer.receipt(), calls


def test_random_access_retains_complete_actual_public_control_proof_and_byte_LF_Unicode(tmp_path):
    proof = json.loads(CONTROL.read_text())
    rows = [dict(text='embedded\nline\u2028separator\u0000'), dict(actual_closed_public_source_proof=proof), dict(end=True)]
    receipt, reserves = write(tmp_path, rows)
    digest = hashlib.sha256()
    with Records(tmp_path / 'pack.gz', tmp_path / 'index', count=3) as records:
        for position in [2, 0, 1, 2]:
            raw, row = records.read(position)
            assert row == rows[position] and raw.endswith(b'\n') and raw.count(b'\n') == 1
        for position in range(3): digest.update(records.read(position)[0])
        with pytest.raises(ValueError): records.read(True)
    assert receipt['complete'] and receipt['records'] == 3 and receipt['complete_ordered_raw_sha256'] == digest.hexdigest()
    assert reserves == [16 * 1024**2] * 3
    (tmp_path / 'actual-public-proof-indexed-transport.json').write_text(json.dumps(dict(
        actual_closed_public_source_proof_retained_byte_LF_and_value_exact=True, actual_random_read_order=[2,0,1,2],
        actual_ordered_raw_sha256=digest.hexdigest(), actual_receipt=receipt, models_constructed=0, production_updates=0), indent=2))


@pytest.mark.parametrize('case', ['corrupt_member', 'index_gap', 'undersized_raw_bound'])
def test_actual_corrupt_or_out_of_bounds_indexed_members_reject(case, tmp_path):
    write(tmp_path, [dict(text='original source' * 1000)])
    pack, index = tmp_path / 'pack.gz', tmp_path / 'index'
    if case == 'corrupt_member':
        content = bytearray(pack.read_bytes()); content[len(content) // 2] ^= 127; pack.write_bytes(content)
    else:
        offset, packed, raw, digest = ENTRY.unpack(index.read_bytes())
        index.write_bytes(ENTRY.pack(offset + 1 if case == 'index_gap' else offset, packed,
            1 if case == 'undersized_raw_bound' else raw, digest))
    with pytest.raises(ValueError):
        with Records(pack, index, count=1) as records: records.read(0)


def test_raw_quota_and_declared_complete_population_cannot_be_bypassed(tmp_path):
    with Writer(tmp_path / 'pack.gz', tmp_path / 'index', root=tmp_path, maximum_raw_bytes=64,
            minimum_free_bytes=0, guard=lambda: None, reserve=lambda *a, **k: None) as writer:
        writer.append(dict(x='a' * 40))
        with pytest.raises(ValueError): writer.append(dict(x='b' * 100))
    assert writer.receipt()['records'] == 1
    with pytest.raises(ValueError): Records(tmp_path / 'pack.gz', tmp_path / 'index', count=2)
    with pytest.raises(ValueError):
        Writer(tmp_path / 'pack.gz', tmp_path / 'index', root=tmp_path, maximum_raw_bytes=64,
            minimum_free_bytes=0, guard=lambda: None)


def test_actual_one_human_source_CPU_fixture_retains_full_CE_and_cannot_admit_production(tmp_path):
    data = NativeSuffixTrainData.CPU_mechanics(ROOT, CONTROL)
    before = random.getstate(), torch.get_rng_state().clone()
    entries = data.complete_schedule(epochs=1, width=4, seed=989711)
    budget = data.budget(entries)
    assert len(data) == 9 and not data.full_source_data_admitted and data.admission_sha256 is None
    assert budget['updates'] == 3 and budget['source_presentations'] == 9 and budget['distinct_original_human_sources'] == 1
    assert budget['original_response_EOS_targets'] == 9 * 68 and budget['complete_original_context_tokens'] == 9 * 496
    assert budget['full_native_generation_horizon_sum'] == 9 * 512
    assert budget['actual_single_source_three_empty_rank_steps'] == 1
    assert sorted(p for e in entries for p in e) == list(range(9))
    with data.episode(0) as episode:
        batch = data.batch(0, 'cpu')
        assert episode.scope == 'CPU-mechanics' and episode.position == 188
        assert batch['input_ids'][0].tolist() == episode.rollout_view()['complete_original_anchor_ids']
        assert int((batch['labels'] != -100).sum()) == 68 and batch['labels'][0, -1] == 2
    assert random.getstate() == before[0] and torch.equal(torch.get_rng_state(), before[1])
    with pytest.raises(ValueError): NativeSuffixTrainData(tmp_path, purpose='optimization')
    (tmp_path / 'actual-one-source-CPU-data-adapter.json').write_text(json.dumps(dict(
        actual_budget=budget, repeated_original_human_source_position=188, full_original_native_horizon=512,
        all_original_context_ids_full_CE_and_true_EOS_retained=True, CPU_fixture_admission_forbidden=True,
        models_constructed=0, neural_forwards=0, production_updates=0), indent=2))


@pytest.mark.parametrize('epochs', [1, 3])
def test_complete_native_source_schedule_preserves_all_actual_views_levels_masks_and_real_tails(epochs, tmp_path):
    metadata = [r for _, r in lines(ROOT / 'reports/full-public-token-suffix-qualification-CPU-v1/all14884-native-view-descriptors.jsonl')]
    assert len(metadata) == 14884
    entries = schedule_source_views(epochs=epochs, width=4, seed=989711)
    assert len(entries) == 3724 * epochs
    assert Counter(p for e in entries for p in e) == Counter({p: epochs for p in range(14884)})
    assert sum(len(e) == 1 for e in entries) == 4 * epochs
    for epoch in range(epochs):
        for level in range(4):
            selected = entries[epoch * 3724 + level * 931:epoch * 3724 + (level + 1) * 931]
            assert {p % 4 for e in selected for p in e} == {level}
            assert sorted(p // 4 for e in selected for p in e) == list(range(3721))
            assert sum(metadata[p]['original_response_EOS_targets'] for e in selected for p in e) == 584218
            assert sum(metadata[p]['original_context_tokens'] for e in selected for p in e) == 2287522
            assert len(selected[-1]) == 1
    (tmp_path / 'actual-complete-source-view-schedule.json').write_text(json.dumps(dict(epochs=epochs,
        actual_complete_original_sources=3721, actual_complete_original_views_per_epoch=14884,
        actual_updates=len(entries), actual_source_presentations=14884 * epochs,
        actual_original_context_tokens=9150088 * epochs, actual_original_response_EOS_targets=2336872 * epochs,
        actual_full_generation_horizon_sum=sum(metadata[p]['full_native_action_budget'] for e in entries for p in e),
        every_original_source_and_full_CE_target_preserved_at_all_four_ordered_levels=True,
        actual_single_source_three_empty_rank_training_steps=4 * epochs,
        only_source_schedule_selection_not_native_profile_admission=True, models_constructed=0,
        neural_forwards=0, production_updates=0), indent=2))
