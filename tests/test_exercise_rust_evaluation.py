from copy import deepcopy

import pytest

from slm_lab.exercise_rust_evaluation import compare_reports


def report():
    keys = ['evaluation_provenance', 'data_sha256', 'manifest_sha256', 'tokenizer_sha256',
            'source_archive_sha256', 'toolchain_manifest_sha256', 'reference_report_sha256',
            'kernel_report_sha256', 'declared_generation_budget', 'protocol',
            'inference_precision', 'inference_device_type']
    value = {key: key for key in keys}
    value.update(checkpoint_sha256='own', full_file_delivery_rate=0., compilation_success_rate=0.,
                 published_module_pass_rate=0.)
    value['results'] = [{'id': str(i), 'group_id': str(i), 'prompt_sha256': str(i),
                         'generation_budget': 2048, 'original_test_sha256': {'test.rs': str(i)},
                         'expected_solution_files': ['src/lib.rs'], 'full_file_delivery': False,
                         'compilation_passed': False, 'published_tests_passed': False} for i in range(24)]
    return value


def success(value, index):
    row = value['results'][index]
    row.update(full_file_delivery=True, compilation_passed=True, published_tests_passed=True,
        execution={'published_tests_passed': True, 'reference_inventory_matched': True,
            'inventory': {'cases': ['original::test'], 'passed_cases': ['original::test'],
                          'skipped': [], 'all_passed': True},
            'execution': [{'returncode': 0, 'limit': None}]})
    for key in ('full_file_delivery_rate', 'compilation_success_rate', 'published_module_pass_rate'):
        value[key] = sum(row['published_tests_passed'] for row in value['results']) / 24


def test_complete_failed_policy_reports_are_valid_without_implying_capability():
    before = report(); after = deepcopy(before); after['checkpoint_sha256'] = 'candidate'
    assert compare_reports(before, after)['passed']
    assert compare_reports(before, after)['after_module_pass_rate'] == 0.


def test_aggregate_tie_cannot_hide_a_lost_original_correct_module():
    before = report(); after = report(); success(before, 0); success(after, 1)
    comparison = compare_reports(before, after)
    assert not comparison['passed'] and comparison['previously_correct_modules_lost'] == ['0']


@pytest.mark.parametrize('field', ['kernel_report_sha256', 'toolchain_manifest_sha256',
                                 'reference_report_sha256', 'declared_generation_budget'])
def test_changed_native_runtime_or_generation_protocol_rejects_comparison(field):
    before = report(); after = deepcopy(before); after[field] = 'changed'
    with pytest.raises(ValueError): compare_reports(before, after)


def test_all_twenty_four_original_modules_required():
    before = report(); after = deepcopy(before); after['results'].pop()
    with pytest.raises(ValueError): compare_reports(before, after)


def test_success_requires_full_native_test_execution_instead_of_summary_claim():
    before = report(); after = deepcopy(before); success(after, 0)
    after['results'][0]['execution']['inventory']['skipped'] = ['original::test']
    with pytest.raises(ValueError): compare_reports(before, after)


def test_reported_aggregate_must_match_per_module_behavior():
    before = report(); after = deepcopy(before); after['published_module_pass_rate'] = 1.
    with pytest.raises(ValueError): compare_reports(before, after)


def test_changed_original_source_interface_cannot_be_compared():
    before = report(); after = deepcopy(before); after['results'][0]['expected_solution_files'] = ['src/other.rs']
    with pytest.raises(ValueError): compare_reports(before, after)
