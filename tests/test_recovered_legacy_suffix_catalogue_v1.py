"""The entire actual recovered source scope, without native/GPU admission."""
import copy
import json
from pathlib import Path

import pytest

from slm_lab.expanded_execution_corpus_v1 import ExpandedExecutionCorpus
from slm_lab.native_code_execution_oracle_v2 import NativeCodeExecutionOracle
from slm_lab.qualified_native_token_suffix_episode_v1 import CompleteLegacySuffixCatalogue
from slm_lab.recovered_legacy_suffix_catalogue_v1 import load_recovered_legacy_catalogue, check_reports

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def actual():
    corpus = ExpandedExecutionCorpus(ROOT)
    oracle = NativeCodeExecutionOracle(corpus)
    return corpus, oracle, load_recovered_legacy_catalogue(oracle)


def test_entire_actual_recovered_catalogue_keeps_every_original_source_view_and_CE_target(actual, tmp_path):
    corpus, oracle, catalogue = actual
    assert type(catalogue) is CompleteLegacySuffixCatalogue and catalogue.oracle is oracle
    assert len(catalogue.rows) == 14884
    for level in [4, 16, 64, 0]:
        rows = [catalogue.rows[(source, level)] for source in range(3721)]
        assert sum(r['exact_native_descriptor']['original_context_tokens'] for r in rows) == 2287522
        assert sum(r['exact_native_descriptor']['original_response_EOS_targets'] for r in rows) == 584218
        for source, row in enumerate(rows):
            ids, labels = corpus.sequence(source)
            d = row['exact_native_descriptor']
            assert row['corpus_id'] == corpus.row(source)['id']
            assert d['original_context_tokens'] == len(ids) and d['original_response_EOS_targets'] == sum(t != -100 for t in labels)
            assert row['source_training_admitted'] is False and row['production_RL_admitted'] is False
    (tmp_path / 'actual-complete-recovered-legacy-catalogue.json').write_text(json.dumps(dict(
        original_sources=3721, original_native_views=14884, context_tokens_each_level=2287522,
        original_response_EOS_targets_each_level=584218, actual_catalogue_digest=catalogue.digest,
        original_failed_reader_not_relabelled_successful=True, new_native_profile_admission=False,
        neural_forwards=0, optimizer_updates=0, goal_achieved=False), indent=2))


def test_actual_qualified_and_CE_only_dispositions_keep_the_exact_unchanged_episode_protocol(actual):
    corpus, oracle, catalogue = actual
    checked = set()
    for (source, level), row in catalogue.rows.items():
        stratum = corpus.row(source)['kind'], level, row['bounded_nontrivial_control_evidence']
        if stratum in checked: continue
        checked.add(stratum)
        if row['bounded_nontrivial_control_evidence']:
            catalogue.check_view(source, level, row['exact_native_descriptor'])
        else:
            with pytest.raises(ValueError): catalogue.check_view(source, level, row['exact_native_descriptor'])
    assert {(kind, level) for kind, level, _ in checked} == {(kind, level)
        for kind in ['documented', 'apps'] for level in [4, 16, 64, 0]}
    assert {eligible for _, _, eligible in checked} == {True, False}
    with pytest.raises(ValueError): load_recovered_legacy_catalogue(object())


def test_recovered_counts_roles_partial_scopes_and_original_failures_cannot_be_waived():
    def read(p): return json.loads((ROOT / 'reports' / p).read_text())
    source = read('full-public-token-suffix-qualification-gzip-recovery-CPU-v2/complete.json')
    reader = read('full-public-token-suffix-readback-gzip-recovery-CPU-v4/complete.json')
    plan = read('full-public-token-suffix-readback-gzip-recovery-CPU-v4/plan.json')
    failure = read('full-public-token-suffix-readback-gzip-recovery-CPU-v2/failure.json')
    check_reports(source, reader, plan, failure)
    for kind in ['scope', 'role', 'failure', 'count', 'retained', 'new', 'targets', 'neural', 'origin']:
        bad = copy.deepcopy(reader)
        if kind == 'scope': bad['complete'] = False
        if kind == 'role': bad['source_training_admitted'] = True
        if kind == 'failure': bad['original_failure_preserved']['error'] = 'waived'
        if kind == 'count': bad['actual_counts']['complete_original_sources'] -= 1
        if kind == 'retained': bad['actual_retained_counts']['fresh_prefix_control_runs'] -= 1
        if kind == 'new': bad['actual_new_counts']['complete_original_sources'] = 0
        if kind == 'targets': bad['all_four_level_metadata_totals']['4']['original_response_EOS_targets'] -= 1
        if kind == 'neural': bad['neural_forwards'] = 1
        if kind == 'origin': bad['retained_controls_not_relabelled_as_new_executions'] = False
        with pytest.raises(ValueError): check_reports(source, bad, plan, failure)
