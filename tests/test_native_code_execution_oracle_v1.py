"""Actual complete source formats and bounded workers; no neural model quality."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import pytest
import torch

from slm_lab.apps_published_cases_v2 import contract
from slm_lab.codenet_checkpoint_v1 import capture_rng
from slm_lab.code_quality import instruction_code
from slm_lab.expanded_execution_corpus_v1 import ExpandedExecutionCorpus
from slm_lab.native_code_execution_oracle_v1 import NativeCodeExecutionOracle, identity
from slm_lab.parallel_function_repair_stage_v1 import value_digest


@pytest.fixture(scope='module')
def corpus(): return ExpandedExecutionCorpus(Path(__file__).resolve().parents[1])


@pytest.fixture(scope='module')
def oracle(corpus): return NativeCodeExecutionOracle(corpus)


def selected(oracle, kind):
    corpus = oracle.corpus
    if kind == 'documented':
        return next(p for p in range(173) if corpus.documented_reward.rows[corpus.row(p)['source_id']]['qualified_for_independent_reader'])
    for p in range(173, len(corpus)):
        row = corpus.row(p)
        if not corpus.apps_controls[row['source_id']]['informative']: continue
        path = corpus.root / 'data/public-apps-original-inventory-v1/original-train' / row['original']['source_problem_id'] / 'input_output.json'
        if contract(path.read_bytes())['mode'] == 'function': return p
    raise AssertionError('The actual admitted APPS corpus must contain a function-mode informative source')


def test_all3721_original_formats_masks_targets_and_human_program_bytes_reconcile(oracle, tmp_path):
    counts = Counter()
    contexts = targets = 0
    for position in range(len(oracle.corpus)):
        info = oracle.source_information(position)
        text = oracle.corpus.row(position)['original']['response']
        rendered = oracle.render(position, text)
        assert rendered['valid'] and rendered['exact_execution_program_sha256'] == info['original_execution_program_sha256']
        counts[info['kind'] + ':' + info['source_format']] += 1
        contexts += info['complete_original_context_tokens']
        targets += info['complete_original_response_EOS_targets']
    assert sum(counts.values()) == 3721 and contexts == 2287522 and targets == 584218
    assert counts['documented:one_exact_Python_fence'] == 173
    assert sum(n for k, n in counts.items() if k.startswith('apps:')) == 3548
    assert all(f.__globals__['instruction_code'] is instruction_code for f in oracle.parents)
    (tmp_path / 'actual-all3721-source-format-readback.json').write_text(json.dumps(dict(
        actual_source_format_counts=dict(counts), actual_original_sources=3721, original_context_tokens=contexts,
        original_positive_response_EOS_targets=targets, every_original_human_program_and_native_mask_reconstructed=True,
        original_parent_scorer_globals_unchanged=True, models_constructed=0, neural_forwards=0,
        source_episode_full_qualification_and_GPUcapacity_and_model_quality_still_required=True), indent=2))


@pytest.mark.parametrize('kind', ['documented', 'apps'])
def test_real_gold_program_workers_reject_extracted_prose_native_controls_and_missing_EOS(oracle, tmp_path, kind):
    position = selected(oracle, kind)
    corpus = oracle.corpus
    original = corpus.row(position)['original']['response']
    rendered = oracle.render(position, original)
    complete = oracle.score_response(position, original, stopped_on_EOS=True, origin='original_reference_control')
    assert complete['reward'] == 1. and complete['all_observed_worker_program_bytes_equal_complete_native_program']
    malformed = 'Here is the code:\n```python\n' + rendered['exact_execution_program'] + '\n```\nThis finishes the answer.'
    old = corpus.score(position, malformed, stopped_on_eos=True)
    rejected = oracle.score_response(position, malformed, stopped_on_EOS=True, origin='test_fixture')
    assert old['reward'] == 1. and rejected['reward'] == 0.
    ids, _ = corpus.sequence(position)
    actions = ids[corpus.row(position)['prefix_positions']:]
    without_EOS = oracle.delivery(position, [], actions[:-1], delivered_action_budget=len(actions) - 1,
        origin='original_reference_control')
    assert without_EOS['reward'] == .9 and not without_EOS['actual_stopped_on_EOS']
    invalid = oracle.delivery(position, [], [0, *actions], delivered_action_budget=len(actions) + 1, origin='test_fixture')
    assert invalid['reward'] == 0. and invalid['status'] == 'invalid_generated_native_control_token'
    assert invalid['actual_generated_suffix_ids'] == [0, *actions]
    assert '<pad>' in invalid['actual_response_with_native_control_tokens_preserved']
    assert invalid['actual_original_oracle_observation'] is None
    # Preserve an actual legacy-style decode/score comparison, not a claim
    # that these deliberately assembled reference controls were model samples.
    legacy_control = corpus.score(position, corpus.tokenizer.decode([0, *actions]), stopped_on_eos=True)
    assert legacy_control['reward'] == 1.
    value = dict(actual_source_position=position, actual_source_id=corpus.row(position)['id'], kind=kind,
        actual_original_gold_control=complete, actual_prose_response=malformed,
        actual_original_extracting_scorer=old, actual_native_scorer_rejection=rejected,
        actual_gold_without_EOS=without_EOS, actual_injected_native_control_delivery=invalid,
        actual_legacy_style_decode_then_score=legacy_control,
        test_responses_are_not_generated_student_samples=True, models_constructed=0, production_updates=0)
    (tmp_path / ('actual-' + kind + '-native-program-controls.json')).write_text(json.dumps(value, indent=2, allow_nan=False))


def test_parallel_real_workers_preserve_parent_globals_and_all_private_global_RNGs(oracle, tmp_path):
    positions = [selected(oracle, k) for k in ['documented', 'apps']]
    original = [oracle.corpus.row(p)['original']['response'] for p in positions]
    before = value_digest(capture_rng('cpu'))
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(oracle.score_response, p, text, stopped_on_EOS=True, origin='original_reference_control')
            for p, text in zip(positions, original)]
        actual = [f.result() for f in futures]
    assert [r['reward'] for r in actual] == [1., 1.] and value_digest(capture_rng('cpu')) == before
    assert all(f.__globals__['instruction_code'] is instruction_code for f in oracle.parents)
    assert all(f.__globals__['instruction_code'] is identity for f in oracle.functions)
    assert not torch.cuda.is_initialized()
    (tmp_path / 'actual-parallel-native-worker-controls.json').write_text(json.dumps(dict(
        actual_original_reference_controls=actual, all_parent_scorer_globals_and_private_RNGs_preserved=True,
        actual_parallel_source_workers=2, models_constructed=0, production_updates=0, GPU_speedup_not_measured=True), indent=2, allow_nan=False))


def test_invalid_delivery_foreign_prefix_source_roles_and_changed_private_binding_reject(oracle, monkeypatch):
    position = selected(oracle, 'apps')
    for actions, horizon in [([2, 2], 2), ([7], 2), ([], 1), ([True, 2], 2)]:
        with pytest.raises(ValueError):
            oracle.delivery(position, [], actions, delivered_action_budget=horizon, origin='test_fixture')
    with pytest.raises(ValueError): oracle.delivery(position, [12345], [2], delivered_action_budget=1, origin='test_fixture')
    with pytest.raises(ValueError): oracle.score_response(position, 'pass', stopped_on_EOS=True, origin='unknown')
    with monkeypatch.context() as changed:
        changed.setattr(oracle.corpus, 'role', 'reserved')
        with pytest.raises(ValueError): oracle.source_information(position)
    with monkeypatch.context() as changed:
        changed.setitem(oracle.functions[0].__globals__, 'instruction_code', lambda text: 'pass')
        with pytest.raises(ValueError): oracle.source_information(position)
    oracle.assert_bound()


def test_actual_nonfinite_candidate_observation_is_preserved_as_JSON_without_reward_repair(oracle, tmp_path):
    position = selected(oracle, 'apps')
    row = oracle.corpus.row(position)
    path = oracle.corpus.root / 'data/public-apps-original-inventory-v1/original-train' / row['original']['source_problem_id'] / 'input_output.json'
    function = contract(path.read_bytes())['function']
    program = 'def ' + function + "(*args, **kwargs):\n    return float('nan')\n"
    text = '```python\n' + program + '\n```' if oracle.source_information(position)['source_format'] == 'one_exact_Python_fence' else program
    actual = oracle.score_response(position, text, stopped_on_EOS=True, origin='test_fixture')
    assert actual['reward'] == 0.
    json.dumps(actual, allow_nan=False)
    (tmp_path / 'actual-nonfinite-candidate-control.json').write_text(json.dumps(actual, indent=2, allow_nan=False))
