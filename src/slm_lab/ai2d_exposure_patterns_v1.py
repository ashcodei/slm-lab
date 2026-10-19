"""Separate generic diagram questions from stronger literal text signatures.

An occurrence of 'What is a producer?' is possible text exposure, not evidence
that the same diagram/choices/gold occurred in training. Even no whole-prompt
match cannot certify semantic, image or complete historical independence.
"""
from collections import Counter

from .ai2d_native_eval_inputs_v1 import PROTOCOL, native_prompt

KINDS = ('original_question', 'question_with_ordered_options', 'whole_native_prompt')
SCOPES = ('current_native', 'future_human', 'future_code', 'wikipedia-expanded-v1', 'code-search-expanded-v1', 'broad-mixture-v1')


def signatures(case):
    if (case['split'] != 'test' or case['purpose'] != 'evaluation_only' or case['admitted_to_training'] is not False
            or case['protocol'] != PROTOCOL): raise ValueError('Original evaluation-only publisher case required')
    prompt = native_prompt(case['question'], case['options'])
    if case['whole_native_prompt'] != prompt: raise ValueError('Complete original native prompt must reconstruct')
    return dict(original_question=case['question'],
        question_with_ordered_options=case['question'] + '\n' + '\n'.join(chr(65 + i) + '. ' + option for i, option in enumerate(case['options'])),
        whole_native_prompt=prompt)


def registry(cases):
    if not cases or len({c['id'] for c in cases}) != len(cases): raise ValueError('Every unique original question required')
    rows = []
    for case in cases:
        for kind, text in signatures(case).items():
            rows.append(dict(audit_id=case['id'] + ':' + kind, benchmark='AI2D:' + kind, id=case['original_question_id'],
                pattern_kind=kind, question=text, previously_sampled_for_local_evaluation=False))
    return rows


def reconcile(cases, scans):
    if set(scans) != set(SCOPES): raise ValueError('All six complete declared source scopes required')
    expected = {r['audit_id'] for r in registry(cases)}
    for name, value in scans.items():
        if (value['scope'] != name or value['all_declared_scope_inputs_scanned'] is not True
                or not set(value['matches']) <= expected): raise ValueError('Complete exact source scope and known original pattern identities required')
    results = []
    for case in cases:
        patterns = {}
        for kind in KINDS:
            identity = case['id'] + ':' + kind
            matches = {scope: scans[scope]['matches'][identity] for scope in SCOPES if identity in scans[scope]['matches']}
            patterns[kind] = dict(inspected_exact_match_scopes=list(matches), matches=matches, found=bool(matches))
        results.append(dict(id=case['id'], original_question_id=case['original_question_id'], image_id=case['image_id'],
            original_ABC_labels=case['original_ABC_labels'], literal_text_patterns=patterns,
            generic_question_match_is_not_same_diagram_options_or_answer_exposure=True,
            no_match_is_not_semantic_image_or_historical_independence=True))
    counts = {kind: dict(any_inspected_scope=sum(r['literal_text_patterns'][kind]['found'] for r in results),
        by_scope={scope: sum(scope in r['literal_text_patterns'][kind]['inspected_exact_match_scopes'] for r in results) for scope in SCOPES}) for kind in KINDS}
    return results, counts
