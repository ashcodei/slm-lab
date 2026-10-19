"""Preserve own-reader-conditioned native answers during feedback selection."""
import copy

from .feedback_reading_guard import validate_coverage as validate_original_coverage
from .own_context_transfer_evaluation import PROTOCOL, compare_reports


def validate_coverage(report):
    validate_original_coverage(report)
    if (report['protocol'] != PROTOCOL or report['image_size_own_reader'] != 768
            or report['context_headroom_tokens'] != 256 or report['maximum_context_words'] != 64
            or not 0 < report['total_inference_parameters'] < 1_000_000_000):
        raise ValueError('Feedback own-context retention requires the complete fixed sub-1B conditioning protocol')
    if compare_reports(report, report)['passed'] is not True:
        raise ValueError('Own-context baseline case/aggregate evidence is inconsistent')


def enforce(result, before, after):
    validate_coverage(before); validate_coverage(after)
    guard = compare_reports(before, after)
    if (guard['base_checkpoint_sha256'] != result['base_checkpoint_sha256']
            or guard['candidate_checkpoint_sha256'] != result['candidate_checkpoint_sha256']):
        raise ValueError('Own-context and existing feedback selection compare different policies')
    merged = copy.deepcopy(result)
    merged['own_context_reading_guard'] = guard
    merged['checks']['own_context_reading_retention'] = {
        'passed': guard['passed'], 'before_accuracy': guard['before_exact_accuracy'],
        'after_accuracy': guard['after_exact_accuracy'],
        'previously_correct_questions_lost': guard['previously_correct_questions_lost']}
    if not guard['passed']: merged['failed_checks'].append('own_context_reading_retention')
    merged['eligible_for_selection'] = result['eligible_for_selection'] and guard['passed']
    return merged
