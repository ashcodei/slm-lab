"""Extend existing feedback selection with matched native real-photo reading."""
import copy

from .textvqa_evaluation import compare_reports


def validate_coverage(report):
    rows = report['results']
    if (len(rows) != 128 or len({row['group_id'] for row in rows}) != 128
            or report['max_examples'] != 128 or report['max_new_tokens'] != 32 or report['image_size'] != 448):
        raise ValueError('Feedback reading guard requires all128 distinct declared real-photo cases')


def enforce(result, before, after):
    validate_coverage(before); validate_coverage(after)
    guard = compare_reports(before, after)
    if (guard['base_checkpoint_sha256'] != result['base_checkpoint_sha256']
            or guard['candidate_checkpoint_sha256'] != result['candidate_checkpoint_sha256']):
        raise ValueError('Reading and existing feedback selection compare different policies')
    merged = copy.deepcopy(result)
    merged['real_photo_reading_guard'] = guard
    merged['checks']['real_photo_reading_retention'] = {'passed': guard['passed'],
               'before_accuracy': guard['before_exact_accuracy'], 'after_accuracy': guard['after_exact_accuracy'],
               'previously_correct_questions_lost': guard['previously_correct_questions_lost']}
    if not guard['passed']:
        merged['failed_checks'].append('real_photo_reading_retention')
    merged['eligible_for_selection'] = result['eligible_for_selection'] and guard['passed']
    return merged
