import copy

import pytest

from slm_lab.exercise_js_evaluation import compare_reports
from slm_lab.exercise_js_runner import extract_files, inventory


def test_file_delivery_preserves_named_es_modules_and_rejects_path_or_test_overrides():
    source = 'File: solution.js\n```javascript\nexport const answer = () => 42;\n```'
    assert extract_files(source, ['solution.js']) == {'solution.js': 'export const answer = () => 42;\n'}
    for name in ['../outside.js', 'suite.spec.js', '/tmp/code.js']:
        with pytest.raises(ValueError): extract_files(source.replace('solution.js', name))
    with pytest.raises(ValueError): extract_files(source, ['other.js'])


def test_original_skipped_cases_prevent_claiming_complete_published_success():
    report = {'success': True, 'numTotalTests': 2, 'numPassedTests': 1,
              'testResults': [{'assertionResults': [{'fullName': 'first', 'status': 'passed', 'numPassingAsserts': 1},
                                                   {'fullName': 'second', 'status': 'pending', 'numPassingAsserts': 0}]}]}
    result = inventory(report)
    assert result['cases'] == ['first', 'second'] and result['skipped'] == 1 and not result['all_passed']


def reports():
    rows = [{'id': str(i), 'group_id': str(i), 'prompt_sha256': str(i), 'generation_budget': 2048,
             'original_test_sha256': {'suite.js': 'source'}, 'expected_solution_files': ['solution.js'],
             'full_file_delivery': True, 'syntax_valid': True, 'published_tests_passed': i == 0} for i in range(37)]
    before = {key: 'fixed' for key in ['evaluation_provenance', 'data_sha256', 'manifest_sha256', 'tokenizer_sha256',
                                      'source_archive_sha256', 'toolchain_manifest_sha256', 'reference_report_sha256',
                                      'protocol', 'inference_precision', 'inference_device_type']}
    before.update(checkpoint_sha256='base', declared_generation_budget=2048, results=rows,
                  full_file_delivery_rate=1., javascript_syntax_rate=1., published_module_pass_rate=1 / 37)
    after = copy.deepcopy(before); after['checkpoint_sha256'] = 'candidate'
    return before, after


def test_same_net_module_score_cannot_hide_lost_original_correct_library():
    before, after = reports()
    after['results'][0]['published_tests_passed'] = False; after['results'][1]['published_tests_passed'] = True
    result = compare_reports(before, after)
    assert not result['passed'] and result['previously_correct_modules_lost'] == ['0']


@pytest.mark.parametrize('mutation', ['subset', 'interface', 'source', 'budget', 'aggregate'])
def test_incomplete_or_unmatched_original_module_evidence_is_rejected(mutation):
    before, after = reports()
    if mutation == 'subset': after['results'].pop()
    if mutation == 'interface': after['results'][0]['expected_solution_files'] = ['other.js']
    if mutation == 'source': after['results'][0]['original_test_sha256'] = {'suite.js': 'changed test'}
    if mutation == 'budget': after['results'][0]['generation_budget'] = 512
    if mutation == 'aggregate': after['published_module_pass_rate'] = 1.
    with pytest.raises(ValueError): compare_reports(before, after)
