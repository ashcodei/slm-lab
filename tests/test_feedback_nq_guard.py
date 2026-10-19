import copy

import pytest

from slm_lab.feedback_nq_guard import enforce, validate_coverage
from slm_lab.nq_open_evaluation import PROTOCOL, summarize


def reports():
    rows = [{'id': str(i), 'group_id': str(i), 'prompt_sha256': 'q' + str(i), 'prompt_tokens': 12,
             'reference_answers': ['Iceland', 'Republic of Iceland'], 'generated': 'Iceland' if i == 0 else 'wrong',
             'exact': i == 0, 'potential_exact_question_exposure': i == 0} for i in range(3610)]
    before = {key: 'fixed' for key in ['evaluation_provenance', 'data_sha256', 'manifest_sha256', 'tokenizer_sha256',
                                      'exposure_audit_sha256', 'input_audit_sha256', 'inference_precision', 'inference_device_type']}
    before.update(checkpoint_sha256='base', protocol=PROTOCOL, generation_budget=32, results=rows, overall=summarize(rows))
    after = copy.deepcopy(before); after['checkpoint_sha256'] = 'candidate'
    result = {'base_checkpoint_sha256': 'base', 'candidate_checkpoint_sha256': 'candidate',
              'eligible_for_selection': True, 'failed_checks': [], 'checks': {'code': {'passed': True}}}
    return result, before, after


def test_positive_preference_cannot_hide_factual_loss_even_when_aggregate_accuracy_holds():
    result, before, after = reports()
    after['results'][0].update(generated='wrong', exact=False)
    after['results'][1].update(generated='Iceland', exact=True)
    after['overall'] = summarize(after['results'])
    guarded = enforce(result, before, after)
    assert not guarded['eligible_for_selection']
    assert guarded['failed_checks'] == ['human_fact_question_retention']
    assert guarded['human_fact_question_guard']['previously_correct_questions_lost'] == ['0']
    assert result['eligible_for_selection'] and not result['failed_checks']


def test_factual_retention_cannot_override_existing_code_or_reading_failure():
    result, before, after = reports()
    result['eligible_for_selection'] = False; result['failed_checks'] = ['code', 'own_context_reading_retention']
    guarded = enforce(result, before, after)
    assert not guarded['eligible_for_selection'] and guarded['failed_checks'] == result['failed_checks']
    assert guarded['checks']['code'] == result['checks']['code']


@pytest.mark.parametrize('change', ['policy', 'subset', 'budget', 'protocol', 'exposure'])
def test_different_branch_partial_benchmark_or_relabelled_exposure_cannot_approve_feedback(change):
    result, before, after = reports()
    if change == 'policy': result['candidate_checkpoint_sha256'] = 'different branch'
    if change == 'subset': after['results'].pop(); after['overall'] = summarize(after['results'])
    if change == 'budget': after['generation_budget'] = 8
    if change == 'protocol': after['protocol'] = 'multiple choice'
    if change == 'exposure':
        after['results'][1]['potential_exact_question_exposure'] = True
        after['overall'] = summarize(after['results'])
    with pytest.raises(ValueError): enforce(result, before, after)


def test_full_matched_factual_case_evidence_can_admit_otherwise_eligible_branch():
    result, before, after = reports(); validate_coverage(before)
    guarded = enforce(result, before, after)
    assert guarded['eligible_for_selection'] and not guarded['failed_checks']
    assert guarded['checks']['human_fact_question_retention']['passed']
