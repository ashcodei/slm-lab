from copy import deepcopy

import pytest

from slm_lab.instruction_trial_comparison import compare_code, knowledge_summary


def report():
    rows = [{'language': 'Python', 'task': name, 'specification': 'Original contract ' + name,
             'intended_scaling': 'linear', 'code': 'def original(): return 1',
             'candidate': {'syntax_valid': True, 'functional_passed': good},
             'efficiency_stress_passed': good} for name, good in [('first', True), ('second', False)]]
    return {'evaluation_provenance': {'files_sha256': {'original.py': 'abc'}}, 'inference_device_type': 'cuda',
            'inference_precision': 'bf16', 'tokenizer_sha256': 'tokenizer', 'suite_sha256': 'original',
            'task_count': 2, 'generation_budget': 512, 'decoding': 'greedy', 'prompt_mode': 'instruction',
            'syntax_pass_rate': 1., 'functional_pass_at_1': .5, 'efficiency_stress_pass_rate': .5,
            'by_language': {'Python': {'tasks': 2, 'syntax_passed': 2, 'functional_passed': 1, 'efficiency_passed': 1}},
            'results': rows}


def test_unchanged_aggregate_cannot_hide_lost_original_correct_program():
    before = report(); after = deepcopy(before)
    for row in after['results']:
        row['candidate']['functional_passed'] = not row['candidate']['functional_passed']
        row['efficiency_stress_passed'] = not row['efficiency_stress_passed']
    result = compare_code(before, after, 'code_instruction', 2)
    assert result['before']['functional'] == result['after']['functional'] == 1
    assert result['case_changes']['functional'] == {'gained': ['Python:second'], 'lost': ['Python:first']}


def test_different_generation_budget_is_not_compared():
    before = report(); after = deepcopy(before); after['generation_budget'] = 128
    with pytest.raises(ValueError, match='generation_budget'):
        compare_code(before, after, 'code_instruction', 2)


def test_forged_aggregate_and_duplicate_cases_are_rejected():
    before = report(); after = deepcopy(before); after['functional_pass_at_1'] = 1.
    with pytest.raises(ValueError, match='actual cases'):
        compare_code(before, after, 'code_instruction', 2)
    after = deepcopy(before); after['results'][1]['task'] = 'first'; after['results'][1]['specification'] = 'Original contract first'
    with pytest.raises(ValueError, match='identities'):
        compare_code(before, after, 'code_instruction', 2)


def test_empty_extracted_code_is_visible_separately_from_syntax_score():
    before = report(); before['results'][1]['code'] = ''; after = deepcopy(before)
    result = compare_code(before, after, 'code_instruction', 2)
    assert result['before']['syntax'] == 2 and result['before']['syntax_passed_with_empty_extracted_code'] == 1


def test_strict_letter_metric_does_not_credit_a_prose_response():
    rows = [{'id': str(i), 'domain': 'test', 'answer_index': 0, 'generated_index': 0 if i < 2 else None,
             'likelihood_index': 0, 'generated_correct': i < 2, 'likelihood_correct': True,
             'generated_response': 'A' if i == 0 else 'A is mentioned in this prose' if i == 1 else '',
             'chance_accuracy': .1} for i in range(559)]
    group = {'questions': 559, 'generated_accuracy': 2 / 559, 'likelihood_accuracy': 1., 'format_compliance': 2 / 559}
    result = knowledge_summary({'predictions': rows, 'skipped': [], 'overall': group, 'by_domain': {'test': group}})
    assert result['generated_correct'] == 2
    assert result['derived_strict_letter_only_responses'] == result['derived_strict_letter_only_correct'] == 1
