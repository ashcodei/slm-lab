import copy

import pytest

from slm_lab.nq_open_evaluation import correct, compare_reports, summarize


def reports():
    rows = [{'id': str(i), 'group_id': str(i), 'prompt_sha256': 'q' + str(i), 'prompt_tokens': 12,
             'reference_answers': ['Iceland', 'Republic of Iceland'], 'generated': 'Iceland' if i == 0 else 'wrong',
             'exact': i == 0, 'potential_exact_question_exposure': i == 0} for i in range(3610)]
    before = {key: 'fixed' for key in ['evaluation_provenance', 'data_sha256', 'manifest_sha256', 'tokenizer_sha256',
                                      'exposure_audit_sha256', 'input_audit_sha256', 'inference_precision', 'inference_device_type', 'protocol']}
    before.update(checkpoint_sha256='base', generation_budget=32, results=rows, overall=summarize(rows))
    after = copy.deepcopy(before); after['checkpoint_sha256'] = 'candidate'
    return before, after


def test_any_original_alias_can_match_and_empty_output_cannot():
    assert correct(' REPUBLIC of Iceland  ', ['Iceland', 'Republic of Iceland'])
    assert correct('Ｒｅｙｋｊａｖｉｋ', ['Reykjavik'])
    assert not correct('', ['Iceland'])
    assert not correct('The answer is Iceland', ['Iceland'])


def test_exact_exposure_does_not_disappear_from_full_original_benchmark_metrics():
    before, _ = reports(); stats = before['overall']
    assert stats['all_original_development']['examples'] == 3610
    assert stats['potential_exact_question_exposure']['examples'] == 1
    assert stats['potential_exact_question_exposure']['correct'] == 1
    assert stats['no_exact_question_match_in_audited_sources']['examples'] == 3609
    assert stats['no_exact_question_match_in_audited_sources']['correct'] == 0


def test_same_total_correctness_cannot_hide_lost_original_answer():
    before, after = reports()
    after['results'][0].update(generated='wrong', exact=False)
    after['results'][1].update(generated='Iceland', exact=True)
    after['overall'] = summarize(after['results'])
    guard = compare_reports(before, after)
    assert not guard['passed'] and guard['previously_correct_questions_lost'] == ['0']


@pytest.mark.parametrize('mutation', ['subset', 'reference', 'exposure', 'false_score', 'aggregate'])
def test_mismatched_evidence_cannot_admit_policy(mutation):
    before, after = reports()
    if mutation == 'subset': after['results'].pop(); after['overall'] = summarize(after['results'])
    if mutation == 'reference': after['results'][1]['reference_answers'] = ['different original source']
    if mutation == 'exposure': after['results'][1]['potential_exact_question_exposure'] = True; after['overall'] = summarize(after['results'])
    if mutation == 'false_score': after['results'][1]['exact'] = True; after['overall'] = summarize(after['results'])
    if mutation == 'aggregate': after['overall']['all_original_development']['correct'] = 99
    with pytest.raises(ValueError): compare_reports(before, after)
