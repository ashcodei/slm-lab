"""Retain complete original native Go modules when selecting feedback policies."""
import copy

from .exercise_go_evaluation import PROTOCOL, compare_reports


def validate_coverage(report, references=None):
    if report['protocol'] != PROTOCOL or report['declared_generation_budget'] != 2048:
        raise ValueError('Feedback Go requires the full42-module native publisher protocol')
    if compare_reports(report, report)['passed'] is not True:
        raise ValueError('Go case and aggregate evidence is inconsistent')
    for row in report['results']:
        flags = [row[key] for key in ('full_file_delivery', 'compilation_passed', 'published_tests_passed')]
        if any(type(value) is not bool for value in flags) or flags[1] and not flags[0] or flags[2] and not flags[1]:
            raise ValueError('Go delivery/compilation/native execution evidence is inconsistent')
        if row['published_tests_passed']:
            module = row['execution']; runtime = module['execution']; inventory = module['inventory']
            if (module['published_tests_passed'] is not True or module['compilation_passed'] is not True
                    or module['returncode'] != 0 or module['limit'] is not None
                    or runtime['returncode'] != 0 or runtime['limit'] is not None
                    or module['reference_inventory_matched'] is not True or inventory['all_passed'] is not True
                    or inventory['skipped'] or sorted(inventory['passed_cases']) != sorted(inventory['cases'])):
                raise ValueError('Successful Go module requires complete passing native compilation/tests')
    if references is not None:
        original = {row['id']: row for row in references}
        if len(references) != 42 or len(original) != 42 or set(original) != {row['id'] for row in report['results']}:
            raise ValueError('Feedback Go must retain all original held module identities')
        for row in report['results']:
            source = original[row['id']]
            if (row['group_id'] != source['group_id'] or row['original_test_sha256'] != source['original_test_sha256']
                    or row['expected_solution_files'] != sorted(source['original_reference_file_sha256'])):
                raise ValueError('Feedback Go interface or original test/module bytes changed')
            if row['published_tests_passed'] and row['execution']['inventory']['cases'] != source['result']['inventory']['cases']:
                raise ValueError('Successful Go module lacks the full original test/subtest inventory')


def enforce(result, before, after, references=None):
    validate_coverage(before, references); validate_coverage(after, references)
    guard = compare_reports(before, after)
    if (guard['base_checkpoint_sha256'] != result['base_checkpoint_sha256']
            or guard['candidate_checkpoint_sha256'] != result['candidate_checkpoint_sha256']):
        raise ValueError('Go and existing feedback selection compare different policies')
    merged = copy.deepcopy(result)
    merged['go_module_guard'] = guard
    merged['checks']['go_module_retention'] = {'passed': guard['passed'],
        'before': guard['before_module_pass_rate'], 'after': guard['after_module_pass_rate'],
        'previously_correct_modules_lost': guard['previously_correct_modules_lost']}
    if not guard['passed']: merged['failed_checks'].append('go_module_retention')
    merged['eligible_for_selection'] = result['eligible_for_selection'] and guard['passed']
    return merged
