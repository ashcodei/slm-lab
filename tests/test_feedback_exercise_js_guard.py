import copy

import pytest

from slm_lab.exercise_js_evaluation import PROTOCOL
from slm_lab.feedback_exercise_js_guard import enforce, validate_coverage


def fixtures():
    inventory = {'cases': ['original behavior'], 'assertions': 2, 'all_passed': True, 'skipped': 0}
    execution = {'published_tests_passed': True, 'returncode': 0, 'reference_inventory_matched': True,
                 'inventory': inventory}
    rows = [{'id': str(i), 'group_id': str(i), 'prompt_sha256': str(i), 'generation_budget': 2048,
             'original_test_sha256': {'solution.spec.js': 'original'}, 'expected_solution_files': ['solution.js'],
             'full_file_delivery': True, 'syntax_valid': True, 'published_tests_passed': i == 0,
             'execution': copy.deepcopy(execution)} for i in range(37)]
    before = {key: 'fixed' for key in ['evaluation_provenance', 'data_sha256', 'manifest_sha256', 'tokenizer_sha256',
                                      'source_archive_sha256', 'toolchain_manifest_sha256', 'reference_report_sha256',
                                      'inference_precision', 'inference_device_type']}
    before.update(checkpoint_sha256='base', protocol=PROTOCOL, declared_generation_budget=2048, results=rows,
                  full_file_delivery_rate=1., javascript_syntax_rate=1., published_module_pass_rate=1 / 37)
    after = copy.deepcopy(before); after['checkpoint_sha256'] = 'candidate'
    result = {'base_checkpoint_sha256': 'base', 'candidate_checkpoint_sha256': 'candidate',
              'eligible_for_selection': True, 'failed_checks': [], 'checks': {'code': {'passed': True}}}
    references = [{'id': str(i), 'group_id': str(i), 'original_test_sha256': {'solution.spec.js': 'original'},
                   'original_reference_file_sha256': {'solution.js': 'body'},
                   'result': {'inventory': copy.deepcopy(inventory)}} for i in range(37)]
    return result, before, after, references


def test_equal_net_score_with_lost_original_library_blocks_positive_preference_selection():
    result, before, after, references = fixtures()
    after['results'][0]['published_tests_passed'] = False
    after['results'][1]['published_tests_passed'] = True
    guarded = enforce(result, before, after, references)
    assert not guarded['eligible_for_selection']
    assert guarded['failed_checks'] == ['javascript_module_retention']
    assert guarded['javascript_module_guard']['previously_correct_modules_lost'] == ['0']
    assert result['eligible_for_selection'] and not result['failed_checks']


def test_passing_modules_cannot_override_preexisting_code_or_factual_qa_failure():
    result, before, after, references = fixtures()
    result.update(eligible_for_selection=False, failed_checks=['code', 'human_fact_question_retention'])
    guarded = enforce(result, before, after, references)
    assert not guarded['eligible_for_selection'] and guarded['failed_checks'] == result['failed_checks']
    assert guarded['checks']['code'] == result['checks']['code']


@pytest.mark.parametrize('change', ['policy', 'subset', 'protocol', 'budget', 'unknown_id', 'group',
                                    'test_bytes', 'interface', 'failed_execution', 'missing_cases',
                                    'fewer_assertions', 'skipped', 'non_boolean'])
def test_unmatched_or_incomplete_native_publisher_evidence_cannot_approve_feedback(change):
    result, before, after, references = fixtures()
    if change == 'policy': result['candidate_checkpoint_sha256'] = 'another branch'
    if change == 'subset': after['results'].pop()
    if change == 'protocol': after['protocol'] = 'isolated function exercise'
    if change == 'budget': after['declared_generation_budget'] = 128
    if change == 'unknown_id': after['results'][0]['id'] = 'other module'
    if change == 'group': after['results'][0]['group_id'] = 'other source'
    if change == 'test_bytes': after['results'][0]['original_test_sha256'] = {'solution.spec.js': 'changed'}
    if change == 'interface': after['results'][0]['expected_solution_files'] = ['different.js']
    if change == 'failed_execution': after['results'][0]['execution']['returncode'] = 1
    if change == 'missing_cases': after['results'][0]['execution']['inventory']['cases'] = []
    if change == 'fewer_assertions': after['results'][0]['execution']['inventory']['assertions'] = 1
    if change == 'skipped': after['results'][0]['execution']['inventory']['skipped'] = 1
    if change == 'non_boolean': after['results'][0]['published_tests_passed'] = 1
    with pytest.raises(ValueError): enforce(result, before, after, references)


def test_full_matched_native_evidence_retains_an_otherwise_eligible_policy():
    result, before, after, references = fixtures(); validate_coverage(before, references)
    guarded = enforce(result, before, after, references)
    assert guarded['eligible_for_selection'] and not guarded['failed_checks']
    assert guarded['checks']['javascript_module_retention']['passed']
