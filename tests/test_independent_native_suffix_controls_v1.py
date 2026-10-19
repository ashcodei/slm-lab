"""Actual public source/control readback; constructs no neural model."""
import copy
import json
from pathlib import Path

import pytest

from slm_lab.bounded_gzip_observation_stream_v1 import lines
from slm_lab.expanded_execution_corpus_v1 import ExpandedExecutionCorpus
from slm_lab.independent_native_suffix_controls_v1 import check, agrees
from slm_lab.independent_token_suffix_readback_v1 import rebuild
from slm_lab.native_code_execution_oracle_v2 import NativeCodeExecutionOracle, safe_observation

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def source():
    corpus = ExpandedExecutionCorpus(ROOT)
    return corpus, NativeCodeExecutionOracle(corpus)


def apps():
    p = ROOT / 'reports/native-suffix-CPU-mechanics-qualification-v1/closed-qualified-controls.json'
    row = json.loads(p.read_text())
    return row['source_position'], row['nominal_missing_non_EOS_tokens'], row['native_view'], row['actual_controls']


@pytest.mark.parametrize('kind', ['apps', 'documented'])
def test_actual_native_gold_prefix_programs_full_masks_budgets_and_fresh_worker_verdicts(source, kind, tmp_path):
    corpus, oracle = source
    with oracle.checked_source_batch():
        if kind == 'apps':
            position, level, metadata, controls = apps()
        else:
            position, level = 0, 16
            metadata = next(r for _, r in lines(ROOT / 'reports/full-public-token-suffix-qualification-CPU-v1/all14884-native-view-descriptors.jsonl')
                if r['source_position'] == position and r['nominal_missing_non_EOS_tokens'] == level)
            metadata = {k: v for k, v in metadata.items() if k not in ['source_position', 'corpus_id', 'kind']}
            rebuilt = rebuild(corpus.tokenizer, corpus.row(position)['original'], *corpus.sequence(position), level, metadata)
            controls = []
            for repeat in range(2):
                gold = oracle.delivery(position, rebuilt['original_reference_prefix_ids'], rebuilt['original_missing_suffix_and_EOS_ids'],
                    delivered_action_budget=metadata['full_native_action_budget'], origin='original_reference_control')
                prefix = oracle.delivery(position, rebuilt['original_reference_prefix_ids'], [2],
                    delivered_action_budget=metadata['full_native_action_budget'], origin='test_fixture')
                legacy = corpus.score(position, rebuilt['original_prefix_text'], stopped_on_eos=True)
                controls.append(dict(repeat=repeat, native_gold=gold, native_prefix_then_EOS=prefix, legacy_prefix_then_EOS=safe_observation(legacy)))
        native, qualified = check(corpus, oracle, position, level, metadata, controls)
        gold = oracle.delivery(position, native['original_reference_prefix_ids'], native['original_missing_suffix_and_EOS_ids'],
            delivered_action_budget=metadata['full_native_action_budget'], origin='original_reference_control')
        prefix = oracle.delivery(position, native['original_reference_prefix_ids'], [2],
            delivered_action_budget=metadata['full_native_action_budget'], origin='test_fixture')
        assert gold['reward'] == 1 and prefix['reward'] == 0 and agrees(controls, gold, prefix)
        assert native['metadata']['original_response_EOS_targets'] == sum(v != -100 for v in corpus.sequence(position)[1])
        assert native['metadata']['full_native_action_budget'] >= len(native['original_missing_suffix_and_EOS_ids'])
        if kind == 'documented': assert prefix['rendered_response']['valid'] is False
    (tmp_path / ('actual-independent-' + kind + '-native-source-controls.json')).write_text(json.dumps(dict(
        source_position=position, kind=kind, level=level, independently_rebuilt_native_metadata=metadata,
        actual_original_controls=controls, actual_fresh_native_gold=gold, actual_fresh_native_prefix=prefix,
        all_actual_program_tests_outputs_and_verdict_semantics_agree=True, conditional_source_qualified=qualified,
        source_only_not_complete_population_admission=True, models_constructed=0, neural_forwards=0,
        production_updates=0, goal_achieved=False), indent=2, allow_nan=False))


@pytest.mark.parametrize('case', ['shortened_budget', 'changed_worker_program', 'dropped_CE_target'])
def test_actual_closed_source_control_corruption_rejects_without_reexecution_or_source_mutation(source, case):
    corpus, oracle = source
    position, level, metadata, controls = apps()
    metadata, controls = copy.deepcopy(metadata), copy.deepcopy(controls)
    if case == 'shortened_budget': controls[0]['native_gold']['actual_delivered_action_budget'] = 5
    elif case == 'changed_worker_program':
        controls[0]['native_gold']['actual_original_oracle_observation']['actual_execution']['actual_program_sha256'] = 'c' * 64
    else: metadata['original_response_EOS_targets'] -= 1
    with oracle.checked_source_batch():
        with pytest.raises(ValueError): check(corpus, oracle, position, level, metadata, controls)


def test_preserved_adapter_missing_verdict_never_becomes_a_zero_reward_or_source_qualification(source):
    corpus, oracle = source
    position, level, metadata, controls = apps()
    controls = copy.deepcopy(controls)
    native = rebuild(corpus.tokenizer, corpus.row(position)['original'], *corpus.sequence(position), level, metadata)
    controls[0]['native_prefix_then_EOS'] = dict(reward=None, status='preserved_native_control_adapter_exception',
        error='controlled unavailable worker observation', requested_source_position=position,
        requested_public_reference_ids=native['original_reference_prefix_ids'], requested_native_actions=[2],
        requested_action_budget=metadata['full_native_action_budget'], requested_origin='test_fixture')
    with oracle.checked_source_batch():
        _, qualified = check(corpus, oracle, position, level, metadata, controls)
    assert not qualified and controls[0]['native_prefix_then_EOS']['reward'] is None
