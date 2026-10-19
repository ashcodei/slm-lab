"""Deferred integration: actual complete native admission, never a CPU fixture.

Run only after the separate native producer, independent reader and data
publisher are complete. This verifies source transport and capacity selection,
not full-size GPU capacity or model learning.
"""
import copy
import gzip
import json
from pathlib import Path

import pytest

from slm_lab.native_suffix_capacity_cases_v1 import choose_admitted
from slm_lab.native_suffix_train_data_v1 import NativeSuffixTrainData, LEVELS
from slm_lab.recovered_native_suffix_train_data_v3 import load_recovered_native_suffix_data, check_admission

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def actual():
    data = load_recovered_native_suffix_data(ROOT, purpose='optimization')
    yield data
    data.records.close()


def test_actual_complete_recovered_native_pack_keeps_every_source_view_and_independent_disposition(actual, tmp_path):
    assert type(actual) is NativeSuffixTrainData and actual.full_source_data_admitted and actual.purpose == 'optimization'
    assert len(actual) == 14884 and actual.admission_sha256 and actual.manifest_sha256
    assert len({actual.information(p)['original_source_id'] for p in actual.slots}) == 3721
    for level in LEVELS:
        rows = [actual.information(p) for p in actual.slots if actual.information(p)['level'] == level]
        assert len(rows) == 3721
        assert sum(r['original_context_tokens'] for r in rows) == 2287522
        assert sum(r['original_response_EOS_targets'] for r in rows) == 584218
        assert all(r['full_source_data_admitted'] and type(r['native_execution_eligible']) is bool for r in rows)
    full = actual.complete_schedule(epochs=3, width=4, seed=989711)
    budget = actual.budget(full)
    assert budget['updates'] == 11172 and budget['source_presentations'] == 44652
    assert budget['original_response_EOS_targets'] == 7010616 and budget['complete_original_context_tokens'] == 27450264
    assert budget['actual_single_source_three_empty_rank_steps'] == 12
    (tmp_path / 'actual-recovered-complete-native-data.json').write_text(json.dumps(dict(
        complete_actual_sources=3721, complete_actual_views=14884, actual_full_training_budget=budget,
        actual_data_admission_sha256=actual.admission_sha256, actual_manifest_sha256=actual.manifest_sha256,
        CPU_mechanics_fixture_used=False, neural_forwards=0, optimizer_updates=0,
        full_size_GPUcapacity_passed=False, goal_achieved=False), indent=2))


def test_actual_capacity_loader_and_selected_batches_use_only_real_full_admitted_native_sources(actual, tmp_path):
    data = load_recovered_native_suffix_data(ROOT, purpose='capacity')
    try:
        assert type(data) is NativeSuffixTrainData and data.purpose == 'capacity'
        plan = choose_admitted(data)
        from types import SimpleNamespace
        from slm_lab.native_suffix_capacity_evidence_v1 import check_complete_capacity_plan, stress_positions
        check_complete_capacity_plan(SimpleNamespace(data=actual, options=dict(seed=989711, epochs=3)), plan)
        worst_horizon_positions = stress_positions(data)
        assert worst_horizon_positions and set(worst_horizon_positions) <= {p for e in plan['capacity_entries'] for p in e}
        assert plan['full_source_data_admitted'] and not plan['geometry_only']
        assert not plan['training_admitted'] and not plan['full_size_capacity_passed']
        assert plan['original_human_sources'] == 3721 and plan['original_source_views'] == 14884
        assert plan['all_12_real_single_source_three_empty_rank_training_tails_retained']
        assert plan['all_selected_sources_physically_rotated_to_each_replica']
        assert plan['complete_training_budget'] == data.budget(data.complete_schedule(epochs=3, width=4, seed=989711))
        assert plan['capacity_budget'] == data.budget(plan['capacity_entries'])
        selected = {p for e in plan['capacity_entries'] for p in e}
        for kind in ['documented', 'apps']:
            for level in LEVELS:
                rows = [data.information(p) for p in data.slots if data.information(p)['kind'] == kind and data.information(p)['level'] == level]
                for key in ['original_context_tokens', 'original_response_EOS_targets']:
                    maximum = max(r[key] for r in rows)
                    assert any(r['position'] in selected and r[key] == maximum for r in rows)
                eligible = [r for r in rows if r['native_execution_eligible']]
                if eligible:
                    for key in ['conditional_prefix_tokens', 'full_native_action_budget']:
                        maximum = max(r[key] for r in eligible)
                        assert any(r['position'] in selected and r[key] == maximum for r in eligible)
        packed = gzip.compress((json.dumps(plan, separators=(',', ':'), allow_nan=False) + '\n').encode(), compresslevel=3, mtime=0)
        assert len(packed) < 1024**2
        (tmp_path / 'actual-admitted-native-capacity-plan.json.gz').write_bytes(packed)
        (tmp_path / 'actual-admitted-native-capacity-selection.json').write_text(json.dumps(dict(
            complete_original_sources=3721, complete_native_views=14884, actual_capacity_budget=plan['capacity_budget'],
            actual_intact_selected_batches=len(plan['selected_intact_original_batches']),
            actual_data_admission_sha256=data.admission_sha256, actual_native_eligible_views=plan['actual_execution_eligible_views'],
            full_size_GPUcapacity_passed=False, neural_forwards=0, optimizer_updates=0, goal_achieved=False), indent=2))
    finally:
        data.records.close()


def test_actual_complete_admission_cannot_be_relabelled_held_partial_neural_or_a_different_loader(actual):
    folder = ROOT / 'data/native-human-code-suffix-v3'
    report = ROOT / 'reports/native-human-code-suffix-data-admission-CPU-v3'
    read = lambda p: json.loads(p.read_text())
    admitted, manifest, receipt, owner = (read(folder / 'admission.json'), read(folder / 'manifest.json'),
        read(report / 'complete.json'), read(report / 'plan.json')['owner'])
    settings = dict(admission_sha256=actual.admission_sha256, manifest_sha256=actual.manifest_sha256)
    check_admission(admitted, manifest, receipt, owner, **settings)
    for kind in ['role', 'scope', 'loader', 'partial', 'targets', 'readback', 'tail', 'neural', 'CE_only', 'manifest']:
        bad, md = copy.deepcopy(admitted), copy.deepcopy(manifest)
        if kind == 'role': bad['admitted_role'] = 'test'
        if kind == 'scope': bad['training_admitted'] = False
        if kind == 'loader': bad['source_loader'] = 'CPU-mechanics'
        if kind == 'partial': bad['original_sources'] -= 1
        if kind == 'targets': bad['each_level_original_response_EOS_targets'] -= 1
        if kind == 'readback': bad['actual_complete_indexed_records_freshly_read_back'] -= 1
        if kind == 'tail': bad['actual_single_source_three_empty_rank_training_steps_per_epoch'] = 0
        if kind == 'neural': bad['model_updates'] = 1
        if kind == 'CE_only': bad['CE_only_views'] -= 1
        if kind == 'manifest': md['separate_recovered_source_transport_v3'] = False
        with pytest.raises(ValueError): check_admission(bad, md, receipt, owner, **settings)
    for purpose in ['test', 'evaluation', 'CPU-mechanics']:
        with pytest.raises(ValueError): load_recovered_native_suffix_data(ROOT, purpose=purpose)
