"""Keep original factual QA case correctness when selecting feedback policies."""
import copy

from .nq_open_evaluation import PROTOCOL, compare_reports


def validate_coverage(report):
    if report['protocol'] != PROTOCOL or report['generation_budget'] != 32:
        raise ValueError('Feedback factual QA requires the fixed full3610-question native protocol')
    if compare_reports(report, report)['passed'] is not True:
        raise ValueError('Original factual QA case/aggregate evidence is inconsistent')


def enforce(result, before, after):
    validate_coverage(before); validate_coverage(after)
    guard = compare_reports(before, after)
    if (guard['base_checkpoint_sha256'] != result['base_checkpoint_sha256']
            or guard['candidate_checkpoint_sha256'] != result['candidate_checkpoint_sha256']):
        raise ValueError('Factual QA and existing feedback selection compare different policies')
    merged = copy.deepcopy(result)
    merged['human_fact_question_guard'] = guard
    merged['checks']['human_fact_question_retention'] = {
        'passed': guard['passed'], 'before': guard['before'], 'after': guard['after'],
        'previously_correct_questions_lost': guard['previously_correct_questions_lost']}
    if not guard['passed']: merged['failed_checks'].append('human_fact_question_retention')
    merged['eligible_for_selection'] = result['eligible_for_selection'] and guard['passed']
    return merged
