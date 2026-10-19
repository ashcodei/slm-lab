"""Closed actual reader observations and genuinely fresh source executions."""
from collections import Counter
import copy
import json
from pathlib import Path

import pytest

from slm_lab.bounded_gzip_observation_stream_v1 import Writer, RawBudget, lines
from slm_lab.expanded_execution_corpus_v1 import ExpandedExecutionCorpus
from slm_lab.independent_suffix_reader_recovery_v1 import prepare, validate_retained, assemble, fresh

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'reports/full-public-token-suffix-qualification-gzip-recovery-CPU-v2'
READER = ROOT / 'reports/full-public-token-suffix-readback-gzip-recovery-CPU-v2'


@pytest.fixture(scope='module')
def actual():
    corpus = ExpandedExecutionCorpus(ROOT)
    completed = json.loads((SOURCE / 'complete.json').read_text())
    selected = {0, 173, 3712, 3713}
    controls = {r['source_position']: r for _, r in lines(completed['complete_control_file']) if r['source_position'] in selected}
    metadata = {p: [] for p in selected}
    for _, r in lines(completed['actual_metadata_path']):
        if r['source_position'] in selected: metadata[r['source_position']].append(r)
    observed = {r['source_position']: (raw, r) for raw, r in lines(READER / 'all3721-fresh-reference-and14884-fresh-prefix-controls.jsonl.gz') if r['source_position'] in selected}
    catalogue = {p: [] for p in selected}
    for raw, r in lines(READER / 'all14884-view-dispositions.jsonl.gz'):
        if r['source_position'] in selected: catalogue[r['source_position']].append((raw, r))
    return corpus, controls, metadata, observed, catalogue


def test_real_documented_and_APPS_closed_reader_sources_reconstruct_all_masks_and_dispositions(actual):
    corpus, controls, metadata, observed, catalogue = actual
    counts = Counter()
    for p in [0, 173, 3712]:
        result = validate_retained(corpus, p, controls[p], metadata[p], observed[p][1], [r for _, r in catalogue[p]])
        counts.update(result['counts'])
        assert result['record'] == observed[p][1]
        assert all(v['original_response_EOS_targets'] == corpus.row(p)['response_EOS_targets'] for v in result['level_totals'].values())
    assert counts['complete_original_sources'] == 3 and counts['fresh_full_reference_runs'] == 3
    assert counts['fresh_prefix_control_runs'] == counts['all_native_views_rebuilt'] == 12


def test_actual_closed_observation_and_catalogue_bytes_survive_one_shared_quota(actual, tmp_path):
    corpus, controls, metadata, observed, catalogue = actual
    budget = RawBudget(1024**3)
    paths = [tmp_path / 'fresh.jsonl.gz', tmp_path / 'catalogue.jsonl.gz']
    raw_fresh, raw_catalogue = [], []
    with Writer(paths[0], root=tmp_path, budget=budget, reserve_bytes=0, guard=lambda:None,
            reserve=lambda *a, **k:None) as a, Writer(paths[1], root=tmp_path, budget=budget, reserve_bytes=0,
            guard=lambda:None, reserve=lambda *a, **k:None) as b:
        for p in [0, 173, 3712]:
            validate_retained(corpus, p, controls[p], metadata[p], observed[p][1], [r for _, r in catalogue[p]])
            raw_fresh.append(observed[p][0]); a.append_raw(raw_fresh[-1])
            for raw, _ in catalogue[p]: raw_catalogue.append(raw); b.append_raw(raw)
    assert b''.join(r for r, _ in lines(paths[0])) == b''.join(raw_fresh)
    assert b''.join(r for r, _ in lines(paths[1])) == b''.join(raw_catalogue)
    assert budget.used == sum(map(len, raw_fresh + raw_catalogue))


def test_changed_actual_source_role_masks_level_prefix_and_disposition_are_rejected(actual):
    corpus, controls, metadata, observed, catalogue = actual
    p = 173
    for kind in ['source', 'role', 'target', 'metadata', 'level', 'prefix', 'eligibility']:
        original, md = copy.deepcopy(controls[p]), copy.deepcopy(metadata[p])
        if kind == 'source': original['source_position'] += 1
        if kind == 'role': original['source_role'] = 'reserved'
        if kind == 'target': original['original_CE_response_EOS_targets'] -= 1
        if kind == 'metadata': md[0]['original_complete_input_sha256'] = '0' * 64
        if kind == 'level': original['views'][0]['nominal_missing_non_EOS_tokens'] = 16
        if kind == 'prefix': original['views'][0]['original_prefix_text'] += 'pass'
        if kind == 'eligibility': original['views'][0]['proposed_nontrivial_bounded_execution_view'] = not original['views'][0]['proposed_nontrivial_bounded_execution_view']
        with pytest.raises(ValueError): prepare(corpus, p, original, md)
    for kind in ['identity', 'missing', 'status', 'admission', 'catalogue', 'disagreement']:
        observation, cat = copy.deepcopy(observed[p][1]), copy.deepcopy([r for _, r in catalogue[p]])
        if kind == 'identity': observation['corpus_id'] += 'x'
        if kind == 'missing': observation['all_four_fresh_views'].pop()
        if kind == 'status': observation['all_four_fresh_views'][0]['fresh_prefix_execution']['status'] = 'invented'
        if kind == 'admission': cat[0]['source_training_admitted'] = True
        if kind == 'catalogue': cat.pop()
        if kind == 'disagreement': observation['full_reference_execution_disagreed'] = not observation['full_reference_execution_disagreed']
        with pytest.raises(ValueError): validate_retained(corpus, p, controls[p], metadata[p], observation, cat)


def test_missing_controls_and_historical_abstentions_or_disagreements_are_not_fabricated_rewards(actual):
    corpus, controls, metadata, observed, catalogue = actual
    p = 173; original = controls[p]
    old_ref = original['full_original_reference_execution']; old_prefix = [r['counterfactual_prefix_then_EOS_execution'] for r in original['views']]
    with pytest.raises(ValueError): assemble(corpus, p, original, metadata[p], old_ref, old_prefix[:3])
    abstention = dict(reward=None, status='preserved_controlled_adapter_abstention')
    result = assemble(corpus, p, original, metadata[p], abstention, [abstention] * 4)
    assert result['record']['fresh_full_reference_execution']['reward'] is None
    assert all(v['fresh_prefix_execution']['reward'] is None for v in result['record']['all_four_fresh_views'])
    assert not any(r['bounded_nontrivial_control_evidence'] for r in result['catalogue'])
    assert result['discrepancies']
    assert all(r['all_original_unconditional_CE_targets_retained'] for r in result['catalogue'])


def test_one_genuinely_unfinished_source_gets_five_fresh_worker_results_with_complete_raw_observations(actual, tmp_path):
    corpus, controls, metadata, observed, catalogue = actual
    p = 3713
    assert p not in observed and not catalogue[p]
    result = fresh(corpus, p, controls[p], metadata[p])
    assert result['counts']['fresh_full_reference_runs'] == 1 and result['counts']['fresh_prefix_control_runs'] == 4
    assert result['record']['source_position'] == p and len(result['record']['all_four_fresh_views']) == 4
    checked = validate_retained(corpus, p, controls[p], metadata[p], result['record'], result['catalogue'])
    assert checked == result
    assert result['record']['fresh_full_reference_execution'].get('actual_execution') is not None
    path = tmp_path / 'actual-new-source-worker-observations.json'
    path.write_text(json.dumps(dict(source_position=p, source_id=corpus.row(p)['id'], actual_fresh_reference_runs=1,
        actual_fresh_prefix_runs=4, complete_raw_result=result, neural_forwards=0, optimizer_updates=0,
        source_training_admitted=False, goal_achieved=False), indent=2))
