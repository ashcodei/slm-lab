"""Finite public-train execution reward with an explicit no-op control.

Only cases that distinguish the reference from the no-op earn partial credit.
Full credit additionally requires every original case and a genuine delivered
EOS. This is machine feedback, not a correctness proof or a human preference.
"""
import hashlib
import math

from .apps_published_cases_v2 import compare, execute
from .code_quality import instruction_code


def require_train(source):
    if source.get('split') != 'train' or source.get('publisher_split') != 'train':
        raise ValueError('Only original admitted publisher-train sources may provide optimization rewards')


def noop(cases):
    if cases['mode'] == 'stdin':
        return 'pass\n'
    name = cases['function']
    if not isinstance(name, str) or not name.isidentifier() or name.startswith('__'):
        raise ValueError('Exact original function interface required')
    return f'def {name}(*args, **kwargs):\n    return None\n'


def flags(observation, cases, *, program_sha256=None):
    if (observation.get('original_tests_sha256') != cases['original_tests_sha256']
            or observation.get('mode') != cases['mode']
            or observation.get('total_cases') != cases['total_cases']
            or program_sha256 is not None and observation.get('original_program_sha256') != program_sha256):
        raise ValueError('Exact original case/program identity required')
    if observation.get('all_cases_observed') is not True:
        return None
    rows = observation.get('raw_results', {}).get('results', [])
    if (not observation.get('syntax_valid') or observation.get('execution', {}).get('status') != 'completed'
            or len(rows) != cases['total_cases']):
        raise ValueError('Every completed finite case must retain an actual raw result')
    result = []
    for row, expected in zip(rows, cases['outputs']):
        if (set(row) != {'value', 'seconds'} or type(row['seconds']) not in (int, float)
                or not math.isfinite(row['seconds']) or row['seconds'] < 0):
            raise ValueError('Finite raw result and bounded nonnegative timing required')
        result.append(compare(row['value'], expected, cases['mode']))
    if (result != observation.get('case_passes') or sum(result) != observation.get('passed_cases')
            or all(result) != observation.get('all_finite_published_cases_pass')
            or observation.get('status') != ('all_finite_published_cases_pass' if all(result)
                                            else 'completed_published_case_mismatch')):
        raise ValueError('Every reported verdict must reconstruct from retained raw values')
    return result


def control(source, cases, original, reference, no_op):
    require_train(source)
    if hashlib.sha256(original.encode()).hexdigest() != source['original_source_body_sha256']:
        raise ValueError('Unchanged original human program required')
    if cases['original_tests_sha256'] != source['original_tests_sha256']:
        raise ValueError('Unchanged original public training cases required')
    reference_flags = flags(reference, cases, program_sha256=source['original_source_body_sha256'])
    no_op_flags = flags(no_op, cases, program_sha256=hashlib.sha256(noop(cases).encode()).hexdigest())
    if reference_flags != [True] * cases['total_cases'] or no_op_flags is None:
        raise ValueError('All actual reference and no-op case results required')
    indices = [i for i, passed in enumerate(no_op_flags) if not passed]
    return dict(source_id=source['id'], original_program_sha256=source['original_source_body_sha256'],
        original_tests_sha256=cases['original_tests_sha256'], total_cases=cases['total_cases'],
        informative_case_indices=indices, informative=bool(indices), noop_case_passes=no_op_flags,
        machine_feedback_not_human_preference=True, partial_credit_ceiling=.9)


def reward(source, cases, admitted_control, response, *, stopped_on_eos, seconds=20):
    require_train(source)
    if (admitted_control['source_id'] != source['id']
            or admitted_control['original_tests_sha256'] != cases['original_tests_sha256']
            or admitted_control['original_program_sha256'] != source['original_source_body_sha256']
            or type(stopped_on_eos) is not bool or not isinstance(response, str)):
        raise ValueError('Exact admitted train control and actual delivery termination required')
    indices = admitted_control['informative_case_indices']
    if (indices != [i for i, passed in enumerate(admitted_control['noop_case_passes']) if not passed]
            or len(admitted_control['noop_case_passes']) != cases['total_cases']):
        raise ValueError('Complete no-op discrimination must reconstruct')
    result = dict(source_id=source['id'], reward=0., informative=bool(indices),
        complete_delivery_stopped_on_EOS=stopped_on_eos, human_preferences=False,
        genuine_human_RLHF=False, all_finite_original_cases_passed=False,
        general_correctness_or_efficiency_proven=False)
    if not indices:
        return dict(result, status='supervised_only_original_cases_do_not_distinguish_noop')
    code = instruction_code(response)
    try:
        actual = execute(code, cases, seconds=seconds)
    except ValueError as error:
        return dict(result, status='candidate_source_qualification', diagnostic=str(error))
    actual_flags = flags(actual, cases, program_sha256=hashlib.sha256(code.encode()).hexdigest())
    if actual_flags is None:
        return dict(result, status=actual['status'], actual_execution=actual)
    full = all(actual_flags)
    partial = sum(actual_flags[i] for i in indices) / len(indices)
    result.update(reward=1. if full and stopped_on_eos else .9 * partial,
        all_finite_original_cases_passed=full, informative_cases_passed=sum(actual_flags[i] for i in indices),
        informative_cases=len(indices), status='complete_original_stream_scored', actual_execution=actual)
    return result
