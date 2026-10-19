"""Retain complete original JavaScript modules when selecting feedback policies."""
import copy

from .exercise_js_evaluation import PROTOCOL, compare_reports


def validate_coverage(report, references=None):
    if report['protocol'] != PROTOCOL or report['declared_generation_budget'] != 2048:
        raise ValueError('Feedback JavaScript requires the full37-module native publisher protocol')
    if compare_reports(report, report)['passed'] is not True:
        raise ValueError('JavaScript case and aggregate evidence is inconsistent')
    for row in report['results']:
        flags = [row[key] for key in ['full_file_delivery', 'syntax_valid', 'published_tests_passed']]
        if any(type(flag) is not bool for flag in flags) or flags[1] and not flags[0] or flags[2] and not flags[1]:
            raise ValueError('JavaScript delivery, parse and execution evidence is inconsistent')
        if row['published_tests_passed']:
            execution = row['execution']
            if (execution['published_tests_passed'] is not True or execution['returncode'] != 0
                    or execution['reference_inventory_matched'] is not True
                    or execution['inventory']['all_passed'] is not True or execution['inventory']['skipped'] != 0):
                raise ValueError('Successful JavaScript module requires complete passing native execution')
    if references is not None:
        original = {row['id']: row for row in references}
        if len(references) != 37 or len(original) != 37 or set(original) != {row['id'] for row in report['results']}:
            raise ValueError('Feedback JavaScript must retain all original held module identities')
        for row in report['results']:
            source = original[row['id']]
            if (row['group_id'] != source['group_id'] or row['original_test_sha256'] != source['original_test_sha256']
                    or row['expected_solution_files'] != sorted(source['original_reference_file_sha256'])):
                raise ValueError('Feedback JavaScript interface or original publisher tests changed')
            if row['published_tests_passed']:
                measured, expected = row['execution']['inventory'], source['result']['inventory']
                if measured['cases'] != expected['cases'] or measured['assertions'] != expected['assertions']:
                    raise ValueError('Successful JavaScript module lacks the full original test inventory')


def enforce(result, before, after, references=None):
    validate_coverage(before, references); validate_coverage(after, references)
    guard = compare_reports(before, after)
    if (guard['base_checkpoint_sha256'] != result['base_checkpoint_sha256']
            or guard['candidate_checkpoint_sha256'] != result['candidate_checkpoint_sha256']):
        raise ValueError('JavaScript and existing feedback selection compare different policies')
    merged = copy.deepcopy(result)
    merged['javascript_module_guard'] = guard
    merged['checks']['javascript_module_retention'] = {
        'passed': guard['passed'], 'before': guard['before_module_pass_rate'],
        'after': guard['after_module_pass_rate'],
        'previously_correct_modules_lost': guard['previously_correct_modules_lost']}
    if not guard['passed']: merged['failed_checks'].append('javascript_module_retention')
    merged['eligible_for_selection'] = result['eligible_for_selection'] and guard['passed']
    return merged
