"""Every actual human-source view and intact batch; no neural or RL admission."""
from collections import Counter
import copy
import gzip
import hashlib
import json
from pathlib import Path

import pytest

from slm_lab.expanded_execution_corpus_v1 import ExpandedExecutionCorpus
from slm_lab.native_suffix_capacity_cases_v1 import source_geometry, choose_geometry, choose_admitted, _select, KEYS
from slm_lab.native_suffix_train_data_v1 import NativeSuffixTrainData, LEVELS, schedule_source_views
from slm_lab.parallel_function_repair_stage_v1 import value_digest

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / 'reports/native-suffix-CPU-mechanics-qualification-v1/closed-qualified-controls.json'


@pytest.fixture(scope='module')
def actual():
    corpus = ExpandedExecutionCorpus(ROOT)
    geometry = source_geometry(corpus)
    return corpus, geometry, choose_geometry(geometry)


def test_every_actual_original_source_view_keeps_all_native_ids_masks_hints_and_full_horizons(actual, tmp_path):
    corpus, geometry, plan = actual
    contexts = Counter(); targets = Counter(); horizons = set()
    for source in range(3721):
        original = corpus.row(source)
        ids, labels = corpus.sequence(source)
        start = next(i for i,x in enumerate(labels) if x != -100)
        assert ids[0] == 1 and ids[-1] == 2 and labels == [-100] * start + ids[start:]
        original_response = original['original']['response']
        for j, level in enumerate(LEVELS):
            row = geometry['records'][4 * source + j]
            prefix = row['conditional_prefix_tokens']; supplied = row['supplied_public_reference_tokens']
            assert row['original_source_position'] == source and row['level'] == level
            assert row['original_context_tokens'] == len(ids) and row['original_response_EOS_targets'] == len(ids) - start
            assert prefix == start + supplied and prefix < len(ids)
            text = corpus.tokenizer.decode(ids[start:prefix], skip_special_tokens=False)
            assert original_response.startswith(text)
            assert ids[:prefix] + ids[prefix:] == ids and labels[prefix:] == ids[prefix:]
            if level:
                assert supplied <= max(0, len(ids) - start - 1 - level)
            else: assert supplied == 0
            expected = min(4096 - prefix, max(512, len(ids) - prefix + 64))
            assert row['full_native_action_budget'] == expected and len(ids) - prefix <= expected
            assert row['native_execution_eligible'] is None and row['full_source_data_admitted'] is False
            contexts[level] += len(ids); targets[level] += len(ids) - start; horizons.add(expected)
    assert set(contexts.values()) == {2287522} and set(targets.values()) == {584218}
    (tmp_path / 'actual-all-human-source-native-geometry.json').write_text(json.dumps(dict(
        all_original_sources=3721, all_native_views=14884, actual_context_tokens_per_level=dict(contexts),
        actual_response_EOS_targets_per_level=dict(targets), actual_original_native_horizons=sorted(horizons),
        all_actual_original_ids_and_masks_preserved=True, execution_eligibility_unknown=True,
        neural_forwards=0, optimizer_updates=0, training_admitted=False), indent=2))


def test_selected_intact_batches_cover_real_maxima_and_every_source_on_each_physical_replica(actual, tmp_path):
    corpus, geometry, plan = actual
    rows = geometry['records']
    full = schedule_source_views(epochs=3, width=4, seed=989711)
    selected = {p for e in plan['capacity_entries'] for p in e}
    for kind in ['documented', 'apps']:
        for level in LEVELS:
            population = [r for r in rows if r['kind'] == kind and r['level'] == level]
            for key in KEYS:
                value = lambda r: r['conditional_prefix_tokens'] + r['full_native_action_budget'] if key == 'prefix_plus_full_native_horizon' else r[key]
                maximum = max(map(value, population))
                assert any(r['position'] in selected and value(r) == maximum for r in population)
    exposure = {p:set() for p in selected}
    for mapping in plan['capacity_case_mapping']:
        original = full[mapping['original_global_step'] - 1]
        rotation = mapping['whole_batch_rank_rotation']
        actual_positions = mapping['actual_original_positions']
        assert original == mapping['original_intact_batch'] and actual_positions == original[rotation:] + original[:rotation]
        assert plan['capacity_entries'][mapping['capacity_step'] - 1] == actual_positions
        for rank,p in enumerate(actual_positions): exposure[p].add(rank)
    assert all(ranks == {0,1,2,3} for ranks in exposure.values())
    actual_tails = {i+1: e for i,e in enumerate(full) if len(e) == 1}
    assert len(actual_tails) == 12
    observed = {m['original_global_step']:m for m in plan['capacity_case_mapping'] if m['actual_original_training_tail']}
    assert set(observed) == set(actual_tails)
    assert all(m['actual_rank_source_counts'] == [1,0,0,0] for m in observed.values())
    for key in ['original_context_tokens', 'original_response_EOS_targets']:
        expected = max(sum(rows[p][key] for p in e) for e in full)
        assert any(sum(rows[p][key] for p in e) == expected for e in plan['capacity_entries'])
    output = (json.dumps(plan, separators=(',', ':'), allow_nan=False) + '\n').encode()
    packed = gzip.compress(output, compresslevel=3, mtime=0)
    assert len(packed) < 1024**2
    (tmp_path / 'actual-geometry-capacity-plan.json.gz').write_bytes(packed)
    (tmp_path / 'actual-full-source-capacity-selection.json').write_text(json.dumps(dict(
        original_human_sources=3721, original_source_views=14884, complete_training_budget=plan['complete_training_budget'],
        actual_selected_intact_batches=len(plan['selected_intact_original_batches']), capacity_groups=len(plan['capacity_entries']),
        actual_selected_sources=len(selected), actual_all_four_replica_exposure_checked=True,
        all_12_actual_training_tails_retained=True, actual_full_geometry_plan_sha256=hashlib.sha256(output).hexdigest(),
        geometry_only=True, full_native_execution_admission_not_granted=True, neural_forwards=0, optimizer_updates=0), indent=2))


def test_full_original_three_epoch_schedule_and_budgets_are_preserved(actual):
    corpus, geometry, plan = actual
    entries = schedule_source_views(epochs=3, width=4, seed=989711)
    assert Counter(p for e in entries for p in e) == Counter({p:3 for p in range(14884)})
    assert plan['complete_training_budget']['updates'] == 11172
    assert plan['complete_training_budget']['source_presentations'] == 44652
    assert plan['complete_training_budget']['complete_original_context_tokens'] == 27450264
    assert plan['complete_training_budget']['original_response_EOS_targets'] == 7010616
    assert plan['complete_training_budget']['execution_eligible_presentations'] is None
    assert plan['geometry_only'] and not plan['full_source_data_admitted'] and not plan['training_admitted']


def test_actual_CPU_fixture_or_unadmitted_geometry_cannot_grant_production_capacity():
    data = NativeSuffixTrainData.CPU_mechanics(ROOT, CONTROL)
    with pytest.raises(ValueError, match='admission'): choose_admitted(data)
    with pytest.raises(ValueError): choose_admitted(object())


def test_geometry_digest_population_roles_and_whole_source_masks_cannot_be_changed(actual):
    corpus, geometry, plan = actual
    for kind in ['stale_digest', 'dropped_source', 'role', 'anchor_targets', 'horizon', 'fake_eligibility']:
        bad = copy.deepcopy(geometry)
        if kind == 'stale_digest': bad['records'][0]['original_context_tokens'] -= 1
        if kind == 'dropped_source': bad['records'].pop()
        if kind == 'role': bad['records'][0]['original_role'] = 'test'
        if kind == 'anchor_targets': bad['records'][0]['original_response_EOS_targets'] -= 1
        if kind == 'horizon': bad['records'][0]['full_native_action_budget'] = 1
        if kind == 'fake_eligibility': bad['records'][0]['native_execution_eligible'] = True
        if kind != 'stale_digest': bad['source_geometry_sha256'] = value_digest(bad['records'])
        with pytest.raises(ValueError): choose_geometry(bad)
    for settings in [dict(epochs=1), dict(epochs=True), dict(seed=True), dict(seed=0)]:
        with pytest.raises(ValueError): choose_geometry(geometry, **settings)


def test_counterfactual_qualified_subset_selects_its_own_real_shape_maxima_without_claiming_admission(actual):
    # A pure selector unit control on actual source shapes. These eligibility
    # labels are deliberately counterfactual; they never enter choose_admitted,
    # source publication, neural training, or a production capacity proof.
    corpus, geometry, source_plan = actual
    rows = copy.deepcopy(geometry['records'])
    for r in rows:
        r['native_execution_eligible'] = r['original_source_position'] % 7 == 0
        r['full_source_data_admitted'] = True
    plan = _select(rows, seed=989711, epochs=3, admitted=True, identity='a'*64,
        admission_sha256='b'*64, manifest_sha256='c'*64)
    selected = {p for e in plan['capacity_entries'] for p in e}
    for kind in ['documented', 'apps']:
        for level in LEVELS:
            eligible = [r for r in rows if r['kind'] == kind and r['level'] == level and r['native_execution_eligible']]
            for key in ['conditional_prefix_tokens', 'full_native_action_budget']:
                maximum = max(r[key] for r in eligible)
                assert any(r['position'] in selected and r[key] == maximum for r in eligible)
    assert plan['training_admitted'] is False and plan['full_size_capacity_passed'] is False
