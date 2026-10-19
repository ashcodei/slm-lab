"""All actual retained native rows, independent semantics and exact byte copy."""
import copy
import json
from pathlib import Path

import pytest

from slm_lab.bounded_gzip_observation_stream_v1 import lines, Writer, RawBudget
from slm_lab.corpus import file_hash
from slm_lab.expanded_execution_corpus_v1 import ExpandedExecutionCorpus
from slm_lab.native_code_execution_oracle_v2 import NativeCodeExecutionOracle
from slm_lab.qualified_native_token_suffix_episode_v1 import QualifiedNativeTokenSuffixEpisode
from slm_lab.recovered_legacy_suffix_catalogue_v1 import load_recovered_legacy_catalogue
from slm_lab.recovered_native_source_controls_v1 import validate_retained

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'reports/full-native-token-suffix-qualification-CPU-v2'


@pytest.fixture(scope='module')
def actual():
    corpus = ExpandedExecutionCorpus(ROOT)
    oracle = NativeCodeExecutionOracle(corpus)
    catalogue = load_recovered_legacy_catalogue(oracle)
    source = json.loads((ROOT / 'reports/full-public-token-suffix-qualification-gzip-recovery-CPU-v2/complete.json').read_text())
    reader = json.loads((ROOT / 'reports/full-public-token-suffix-readback-gzip-recovery-CPU-v4/complete.json').read_text())
    old = [row for _, row in lines(source['complete_control_file'])]
    fresh = [row for _, row in lines(reader['actual_fresh_execution_control_file'])]
    retained = list(lines(OLD / 'all14884-native-profile-view-controls.jsonl.gz'))
    proof_files = {str(OLD / 'closed-source-bindings.json'): file_hash(OLD / 'closed-source-bindings.json')}
    assert len(retained) == 2371 and len(old) == len(fresh) == 3721
    def check(index, observed=None):
        position, level_index = divmod(index, 4)
        return validate_retained(corpus, oracle, catalogue, retained[index][1] if observed is None else observed, index,
            old[position]['views'][level_index], fresh[position]['all_four_fresh_views'][level_index], expected_proof_files=proof_files)
    return corpus, oracle, catalogue, retained, check


def test_all_2371_actual_retained_native_views_reconstruct_with_zero_new_executions(actual, tmp_path):
    corpus, oracle, catalogue, retained, check = actual
    results = []
    with oracle.checked_source_batch():
        for index in range(len(retained)): results.append(check(index))
    assert sum(r['qualified'] for r in results) == 2067
    assert sum(r['actual_retained_native_gold_control_invocations'] for r in results) == 4134
    assert sum(r['actual_retained_native_prefix_control_invocations'] for r in results) == 4134
    assert all(r['new_executions'] == 0 for r in results)
    assert retained[-1][1]['source_position'] == 592 and retained[-1][1]['level'] == 64
    (tmp_path / 'actual-complete-retained-native-prefix-validation.json').write_text(json.dumps(dict(
        actual_retained_closed_views=2371, actual_whole_source_prefix=592, actual_partial_last_source_views=3,
        actual_qualified_native_views=2067, actual_old_gold_control_invocations=4134, actual_old_prefix_control_invocations=4134,
        independently_rebuilt_complete_native_programs_masks_budgets_outcomes_and_legacy_provenance=True,
        actual_new_executions=0, whole_3721source_14884view_native_scope_still_incomplete=True,
        source_training_admitted=False, production_RL_admitted=False, goal_achieved=False), indent=2))


def test_changed_source_targets_control_program_or_retained_CE_disposition_rejects(actual):
    _, oracle, _, retained, check = actual
    qualified = next(i for i, (_, r) in enumerate(retained) if r['eligible_for_later_native_profile'])
    failed = next(i for i, (_, r) in enumerate(retained) if not r['eligible_for_later_native_profile'])
    with oracle.checked_source_batch():
        for kind in ['source', 'targets', 'budget', 'program', 'binding', 'legacy', 'upgrade']:
            index = failed if kind == 'upgrade' else qualified
            row = copy.deepcopy(retained[index][1]); proof = row['native_profile_control_proof']
            if kind == 'source': row['source_position'] += 1
            if kind == 'targets': proof['native_view']['original_response_EOS_targets'] -= 1
            if kind == 'budget': proof['native_view']['full_native_action_budget'] = 1
            if kind == 'program': proof['actual_controls'][0]['native_gold']['rendered_response']['exact_execution_program_sha256'] = 'e'*64
            if kind == 'binding': proof['source_files_sha256'] = {}
            if kind == 'legacy': proof['actual_controls'][0]['legacy_prefix_then_EOS']['reward'] = 1.
            if kind == 'upgrade': row['eligible_for_later_native_profile'] = True
            with pytest.raises(ValueError): check(index, row)


def test_actual_qualified_retained_proof_keeps_exact_downstream_type_and_all_raw_byte_LF_records(actual, tmp_path):
    _, oracle, catalogue, retained, _ = actual
    qualified = next(r for _, r in retained if r['eligible_for_later_native_profile'])
    proof = tmp_path / 'actual-retained-proof.json'
    proof.write_text(json.dumps(qualified['native_profile_control_proof'], ensure_ascii=True, allow_nan=False)+'\n')
    with oracle.checked_source_batch():
        episode = QualifiedNativeTokenSuffixEpisode(oracle, proof, catalogue=catalogue)
        episode.check()
    path = tmp_path / 'retained-records.jsonl.gz'
    with Writer(path, root=ROOT, budget=RawBudget(128*1024**2), reserve_bytes=40*1024**3, guard=lambda: None) as writer:
        for raw, _ in retained: writer.append_raw(raw)
    assert [raw for raw, _ in lines(path)] == [raw for raw, _ in retained]
