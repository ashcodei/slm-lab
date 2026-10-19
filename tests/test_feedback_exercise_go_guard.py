import copy

import pytest

from slm_lab.exercise_go_evaluation import PROTOCOL
from slm_lab.feedback_exercise_go_guard import enforce, validate_coverage


def fixtures():
    inventory = {'cases': ['TestOriginal'], 'passed_cases': ['TestOriginal'], 'skipped': [], 'all_passed': True}
    execution = {'published_tests_passed': True, 'compilation_passed': True, 'returncode': 0, 'limit': None,
        'execution': {'returncode': 0, 'limit': None}, 'reference_inventory_matched': True, 'inventory': inventory}
    rows = [{'id': str(i), 'group_id': str(i), 'prompt_sha256': str(i), 'generation_budget': 2048,
        'original_test_sha256': {'original_test.go': 'source', 'go.mod': 'module'}, 'expected_solution_files': ['solution.go'],
        'full_file_delivery': True, 'compilation_passed': True, 'published_tests_passed': i == 0,
        'execution': copy.deepcopy(execution)} for i in range(42)]
    before = {key: 'fixed' for key in ['evaluation_provenance', 'data_sha256', 'manifest_sha256', 'tokenizer_sha256',
        'source_archive_sha256', 'toolchain_manifest_sha256', 'reference_report_sha256',
        'inference_precision', 'inference_device_type']}
    before.update(checkpoint_sha256='base', protocol=PROTOCOL, declared_generation_budget=2048, results=rows,
        full_file_delivery_rate=1., compilation_success_rate=1., published_module_pass_rate=1 / 42)
    after = copy.deepcopy(before); after['checkpoint_sha256'] = 'candidate'
    result = {'base_checkpoint_sha256': 'base', 'candidate_checkpoint_sha256': 'candidate',
        'eligible_for_selection': True, 'failed_checks': [], 'checks': {'code': {'passed': True}}}
    references = [{'id': str(i), 'group_id': str(i), 'original_test_sha256': {'original_test.go': 'source', 'go.mod': 'module'},
        'original_reference_file_sha256': {'solution.go': 'body'}, 'result': {'inventory': copy.deepcopy(inventory)}} for i in range(42)]
    return result, before, after, references


def test_equal_net_go_score_cannot_hide_lost_original_correct_module_after_feedback():
    result, before, after, references = fixtures()
    after['results'][0]['published_tests_passed'] = False; after['results'][1]['published_tests_passed'] = True
    guarded = enforce(result, before, after, references)
    assert not guarded['eligible_for_selection'] and guarded['failed_checks'] == ['go_module_retention']
    assert guarded['go_module_guard']['previously_correct_modules_lost'] == ['0']
    assert result['eligible_for_selection'] and not result['failed_checks']


def test_go_pass_cannot_override_existing_reading_factual_or_javascript_failure():
    result, before, after, references = fixtures()
    result.update(eligible_for_selection=False, failed_checks=['own_context_reading_retention', 'javascript_module_retention'])
    guarded = enforce(result, before, after, references)
    assert not guarded['eligible_for_selection'] and guarded['failed_checks'] == result['failed_checks']
    assert guarded['checks']['code'] == result['checks']['code']


@pytest.mark.parametrize('change', ['policy', 'subset', 'protocol', 'budget', 'unknown_id', 'group', 'test_bytes',
    'interface', 'compile_error', 'runtime_error', 'missing_cases', 'skipped', 'non_boolean'])
def test_partial_or_mismatched_native_go_evidence_cannot_approve_feedback(change):
    result, before, after, references = fixtures()
    if change == 'policy': result['candidate_checkpoint_sha256'] = 'another policy'
    if change == 'subset': after['results'].pop()
    if change == 'protocol': after['protocol'] = 'source-parse only'
    if change == 'budget': after['declared_generation_budget'] = 128
    if change == 'unknown_id': after['results'][0]['id'] = 'another module'
    if change == 'group': after['results'][0]['group_id'] = 'another group'
    if change == 'test_bytes': after['results'][0]['original_test_sha256'] = {'original_test.go': 'changed'}
    if change == 'interface': after['results'][0]['expected_solution_files'] = ['different.go']
    if change == 'compile_error': after['results'][0]['execution']['returncode'] = 1
    if change == 'runtime_error': after['results'][0]['execution']['execution']['returncode'] = 1
    if change == 'missing_cases':
        after['results'][0]['execution']['inventory'].update(cases=[], passed_cases=[])
    if change == 'skipped': after['results'][0]['execution']['inventory']['skipped'] = ['TestOriginal']
    if change == 'non_boolean': after['results'][0]['published_tests_passed'] = 1
    with pytest.raises(ValueError): enforce(result, before, after, references)


def test_full_matched_go_evidence_keeps_an_otherwise_eligible_feedback_branch():
    result, before, after, references = fixtures(); validate_coverage(before, references)
    guarded = enforce(result, before, after, references)
    assert guarded['eligible_for_selection'] and not guarded['failed_checks']
    assert guarded['checks']['go_module_retention']['passed']
