import copy

import pytest

from slm_lab.exercise_evaluation import compare_reports


def report():
    return {'evaluation_provenance': {'fixture': True}, 'data_sha256': 'data', 'manifest_sha256': 'manifest',
            'tokenizer_sha256': 'own', 'inference_precision': 'bf16', 'inference_device_type': 'cuda',
            'declared_generation_budget': 2048, 'protocol': 'fixture', 'examples': 2,
            'checkpoint_sha256': 'before', 'python_syntax_accuracy': 1., 'module_literal_test_pass_rate': .5,
            'results': [{'id': name, 'group_id': name, 'prompt_sha256': name, 'generation_budget': 2048,
                         'checks': [{'function': name, 'published_cases': [{'args_literal': '(3,)', 'expected_literal': '9'}]}],
                         'all_supported_literal_tests_passed': name == 'first'} for name in ['first', 'second']]}


def test_aggregate_improvement_cannot_hide_loss_of_previously_correct_module():
    before = report(); after = copy.deepcopy(before)
    after['checkpoint_sha256'] = 'after'
    after['results'][0]['all_supported_literal_tests_passed'] = False
    after['results'][1]['all_supported_literal_tests_passed'] = True
    guard = compare_reports(before, after)
    assert not guard['passed'] and guard['previously_correct_modules_lost'] == ['first']


@pytest.mark.parametrize('changed', ['budget', 'case', 'precision'])
def test_changed_evaluation_contract_cannot_be_compared(changed):
    before = report(); after = copy.deepcopy(before)
    if changed == 'budget': after['results'][0]['generation_budget'] = 512
    elif changed == 'case': after['results'][0]['checks'][0]['published_cases'][0]['expected_literal'] = '10'
    else: after['inference_precision'] = 'fp32'
    with pytest.raises(ValueError, match='Unmatched'): compare_reports(before, after)
