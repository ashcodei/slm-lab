import copy

import pytest

from slm_lab.ai2d_native_eval_inputs_v1 import PROTOCOL, native_prompt
from slm_lab.ai2d_exposure_patterns_v1 import KINDS, SCOPES, reconcile, registry, signatures
from slm_lab.held_question_scanner_v1 import QuestionScanner


def example(identity, choices):
    question = 'What is a producer?'
    return dict(id='unit:' + identity, original_question_id=identity, image_id=int(identity), original_ABC_labels=False,
        split='test', purpose='evaluation_only', admitted_to_training=False, protocol=PROTOCOL,
        question=question, options=choices, whole_native_prompt=native_prompt(question, choices), answer_index=0)


def scan(cases, text):
    scanner = QuestionScanner(registry(cases)); scanner.text(text, dict(scope='current_native', family='unit-only', position=17))
    return {name: dict(scope=name, all_declared_scope_inputs_scanned=True,
        matches=scanner.matches if name == 'current_native' else {}) for name in SCOPES}


def test_generic_shared_question_does_not_claim_same_options_or_image():
    a = example('1', ['grass', 'fox']); b = example('2', ['algae', 'fish'])
    results, counts = reconcile([a, b], scan([a, b], signatures(a)['question_with_ordered_options']))
    assert counts['original_question']['any_inspected_scope'] == 2
    assert counts['question_with_ordered_options']['any_inspected_scope'] == 1
    assert counts['whole_native_prompt']['any_inspected_scope'] == 0
    assert results[0]['literal_text_patterns']['question_with_ordered_options']['found']
    assert not results[1]['literal_text_patterns']['question_with_ordered_options']['found']
    assert all(r['generic_question_match_is_not_same_diagram_options_or_answer_exposure'] for r in results)
    results, counts = reconcile([a, b], scan([a, b], a['whole_native_prompt']))
    assert counts['whole_native_prompt']['any_inspected_scope'] == 1
    changed = copy.deepcopy(a); changed['answer_index'] = 1
    assert signatures(a) == signatures(changed)


def test_source_role_partial_scopes_and_unknown_matches_cannot_claim_complete_audit():
    a = example('1', ['grass', 'fox']); scans = scan([a], 'No relevant question.')
    _, counts = reconcile([a], scans); assert all(counts[k]['any_inspected_scope'] == 0 for k in KINDS)
    missing = dict(scans); missing.pop(SCOPES[-1])
    with pytest.raises(ValueError): reconcile([a], missing)
    bad = copy.deepcopy(scans); bad['current_native']['matches']['unknown:case'] = {}
    with pytest.raises(ValueError): reconcile([a], bad)
    bad = copy.deepcopy(a); bad['purpose'] = 'optimization'
    with pytest.raises(ValueError): registry([bad])
