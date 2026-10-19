"""Keep all original complete Rust publisher modules in later policy selection."""
import copy

from .exercise_rust_evaluation_v2 import PROTOCOL, compare_reports


def validate_coverage(report, references=None):
    if report['protocol'] != PROTOCOL or report['declared_generation_budget'] != 2048:
        raise ValueError('Rust retention requires all24 original modules and full native protocol')
    if compare_reports(report, report)['passed'] is not True:
        raise ValueError('Rust case and aggregate evidence is inconsistent')
    if references is not None:
        original = {row['id']: row for row in references}
        if len(references) != 24 or len(original) != 24 or set(original) != {row['id'] for row in report['results']}:
            raise ValueError('Rust retention requires every original held module identity')
        for row in report['results']:
            reference = original[row['id']]
            if (row['group_id'] != reference['group_id']
                    or row['original_test_sha256'] != reference['original_test_sha256']
                    or row['expected_solution_files'] != sorted(reference['original_reference_file_sha256'])):
                raise ValueError('Rust original package/interface/test identities changed')
            if row['published_tests_passed']:
                execution = row['execution']; expected = reference['result']['inventory']['cases']
                if (execution['inventory']['cases'] != expected
                        or execution.get('compilation_passed') is not True
                        or not execution.get('compilation')
                        or any(part['returncode'] != 0 or part['limit'] for part in execution['compilation'])):
                    raise ValueError('Successful Rust module lacks all original compiled tests/cases')


def enforce(result, before, after, references=None):
    validate_coverage(before, references); validate_coverage(after, references)
    guard = compare_reports(before, after)
    if (guard['base_checkpoint_sha256'] != result['base_checkpoint_sha256']
            or guard['candidate_checkpoint_sha256'] != result['candidate_checkpoint_sha256']):
        raise ValueError('Rust and previous capability checks refer to different policies')
    merged = copy.deepcopy(result)
    merged['rust_module_guard'] = guard
    merged['checks']['rust_module_retention'] = {'passed': guard['passed'],
        'before': guard['before_module_pass_rate'], 'after': guard['after_module_pass_rate'],
        'previously_correct_modules_lost': guard['previously_correct_modules_lost']}
    if not guard['passed']: merged['failed_checks'].append('rust_module_retention')
    merged['eligible_for_selection'] = result['eligible_for_selection'] and guard['passed']
    return merged
