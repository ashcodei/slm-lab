import copy

import pytest

from slm_lab.exercise_cpp_selection import compare_reports


def report():
    return {'evaluation_provenance': {'fixture': True}, 'data_sha256': 'data', 'manifest_sha256': 'manifest',
            'tokenizer_sha256': 'own', 'inference_precision': 'bf16', 'inference_device_type': 'cuda',
            'protocol': 'fixture', 'toolchain': {'fixture': True}, 'examples': 2, 'reference_report_sha256': 'reference',
            'checkpoint_sha256': 'before', 'compilation_success_rate': 1., 'published_suite_pass_rate': .5,
            'results': [{'id': name, 'group_id': name, 'prompt_sha256': name, 'context_tokens': 128,
                         'generation_budget': 2048, 'solution_files': [name + '.cpp'], 'test_sha256': {'suite': name},
                         'reference_inventory': {'assertions': 2, 'cases': [['global', name]], 'all_passed': True},
                         'candidate': {'compilation_succeeded': True, 'published_tests_passed': name == 'first'}}
                        for name in ['first', 'second']]}


def test_aggregate_improvement_cannot_hide_losing_a_correct_cpp_module():
    before = report(); after = copy.deepcopy(before)
    after['checkpoint_sha256'] = 'after'
    after['results'][0]['candidate']['published_tests_passed'] = False
    after['results'][1]['candidate']['published_tests_passed'] = True
    guard = compare_reports(before, after)
    assert not guard['passed'] and guard['previously_correct_modules_lost'] == ['first']


def test_complete_module_improvement_with_retained_existing_passes_is_admitted():
    before = report(); after = copy.deepcopy(before)
    after['checkpoint_sha256'] = 'after'
    after['results'][1]['candidate']['published_tests_passed'] = True
    after['published_suite_pass_rate'] = 1.
    assert compare_reports(before, after)['passed']


@pytest.mark.parametrize('change', ['budget', 'tests', 'case_inventory', 'precision', 'compiler', 'duplicate', 'aggregate'])
def test_changed_contract_or_inconsistent_evidence_cannot_be_compared(change):
    before = report(); after = copy.deepcopy(before)
    if change == 'budget': after['results'][0]['generation_budget'] = 512
    elif change == 'tests': after['results'][0]['test_sha256']['suite'] = 'other'
    elif change == 'case_inventory': after['results'][0]['reference_inventory']['assertions'] = 1
    elif change == 'precision': after['inference_precision'] = 'fp32'
    elif change == 'compiler': after['toolchain'] = {'other': True}
    elif change == 'duplicate': after['results'][1]['id'] = after['results'][0]['id']
    else: after['published_suite_pass_rate'] = 1.
    with pytest.raises(ValueError): compare_reports(before, after)
