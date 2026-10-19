import copy

import pytest

from slm_lab.exercise_go_evaluation import compare_reports
from slm_lab.exercise_go_runner import extract_files, inventory


def test_complete_go_delivery_rejects_paths_and_test_file_overrides():
    raw = 'File: solution.go\n```go\npackage solution\nfunc Answer() int { return 42 }\n```'
    assert set(extract_files(raw, ['solution.go'])) == {'solution.go'}
    for name in ('../outside.go', 'suite_test.go', '/tmp/code.go'):
        with pytest.raises(ValueError): extract_files(raw.replace('solution.go', name))
    with pytest.raises(ValueError): extract_files(raw, ['other.go'])


def test_package_exit_success_cannot_hide_missing_or_skipped_original_test_cases():
    with pytest.raises(ValueError): inventory([{'Action': 'pass', 'Package': 'published'}])
    events = [{'Action': 'run', 'Test': 'TestOriginal'}, {'Action': 'skip', 'Test': 'TestOriginal'}, {'Action': 'pass'}]
    assert not inventory(events)['all_passed']


def reports():
    rows = [{'id': str(i), 'group_id': str(i), 'prompt_sha256': str(i), 'generation_budget': 2048,
        'original_test_sha256': {'original_test.go': 'source'}, 'expected_solution_files': ['solution.go'],
        'full_file_delivery': True, 'compilation_passed': True, 'published_tests_passed': i == 0} for i in range(42)]
    before = {key: 'fixed' for key in ['evaluation_provenance', 'data_sha256', 'manifest_sha256', 'tokenizer_sha256',
        'source_archive_sha256', 'toolchain_manifest_sha256', 'reference_report_sha256', 'protocol',
        'inference_precision', 'inference_device_type']}
    before.update(checkpoint_sha256='base', declared_generation_budget=2048, results=rows,
        full_file_delivery_rate=1., compilation_success_rate=1., published_module_pass_rate=1 / 42)
    after = copy.deepcopy(before); after['checkpoint_sha256'] = 'candidate'
    return before, after


def test_equal_aggregate_pass_rate_cannot_hide_lost_correct_original_go_library():
    before, after = reports()
    after['results'][0]['published_tests_passed'] = False
    after['results'][1]['published_tests_passed'] = True
    guard = compare_reports(before, after)
    assert not guard['passed'] and guard['previously_correct_modules_lost'] == ['0']


@pytest.mark.parametrize('change', ['subset', 'interface', 'test_source', 'budget', 'aggregate'])
def test_incomplete_or_changed_go_development_protocol_is_rejected(change):
    before, after = reports()
    if change == 'subset': after['results'].pop()
    if change == 'interface': after['results'][0]['expected_solution_files'] = ['other.go']
    if change == 'test_source': after['results'][0]['original_test_sha256'] = {'original_test.go': 'changed'}
    if change == 'budget': after['results'][0]['generation_budget'] = 100
    if change == 'aggregate': after['published_module_pass_rate'] = 1.
    with pytest.raises(ValueError): compare_reports(before, after)
