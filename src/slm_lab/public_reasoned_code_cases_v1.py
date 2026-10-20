"""Full published-case qualification with reference-fixed output conventions.

Reuses the existing confined stdlib runtime. It never places expected values
in the program's readable directory and never treats finite-case agreement as
proof of general correctness, an official benchmark score or an RL oracle.
"""
import hashlib
import json
import math

from .apps_published_cases_v2 import contract, execute, implementation_sources


def case_identity(cases):
    validated = contract(json.dumps(dict(inputs=cases['inputs'], outputs=cases['outputs'],
        fn_name=cases['function']), allow_nan=False, separators=(',', ':')).encode())
    if any(validated[k] != cases[k] for k in ('mode', 'function', 'inputs', 'outputs', 'total_cases')):
        raise ValueError('Complete original test contract required without clipping')
    return hashlib.sha256(json.dumps({k: cases[k] for k in ('mode', 'function', 'inputs', 'outputs', 'total_cases')},
        ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def values(cases, observation):
    case_identity(cases)
    if (observation.get('all_cases_observed') is not True
            or observation.get('total_cases') != cases['total_cases']
            or observation.get('mode') != cases['mode']
            or observation.get('original_tests_sha256') != cases['original_tests_sha256']
            or observation.get('execution', {}).get('status') != 'completed'):
        return None
    raw = observation.get('raw_results', {})
    rows = raw.get('results')
    if (raw.get('format') != 'own-APPS-raw-results-v2' or not isinstance(rows, list)
            or len(rows) != cases['total_cases']
            or any(set(row) != {'value', 'seconds'} or type(row['seconds']) not in (int, float)
                or not math.isfinite(row['seconds']) or row['seconds'] < 0 for row in rows)):
        raise ValueError('Every actual confined returned value and timing must be retained')
    return [row['value'] for row in rows]


def flags(actual, expected, comparison):
    if len(actual) != len(expected):
        raise ValueError('Every original finite test is required')
    if comparison == 'stdin_exact_ordered_tokens':
        return [isinstance(a, str) and a.encode().split() == b.encode().split() for a, b in zip(actual, expected)]
    if comparison == 'stdin_outside_whitespace_only':
        return [isinstance(a, str) and a.strip() == b.strip() for a, b in zip(actual, expected)]
    if comparison == 'function_exact_JSON_value':
        return [a == b for a, b in zip(actual, expected)]
    if comparison == 'function_original_singleton_wrapper':
        if not all(isinstance(b, list) and len(b) == 1 for b in expected):
            raise ValueError('One fixed original singleton wrapper required for every case')
        return [a == b[0] for a, b in zip(actual, expected)]
    raise ValueError('Unknown reference-fixed output convention')


def qualify_reference(cases, observation, *, stdin_comparison='stdin_outside_whitespace_only'):
    identity = case_identity(cases)
    result = dict(complete_case_identity_sha256=identity, total_cases=cases['total_cases'],
        reference_program_sha256=observation['original_program_sha256'], reference_qualified=False,
        comparison=None, output_convention_fixed_before_candidate_execution=True)
    actual = values(cases, observation)
    if actual is None:
        return dict(result, status='original_reference_incomplete_runtime_qualification')
    if cases['mode'] == 'stdin':
        if stdin_comparison not in ('stdin_outside_whitespace_only', 'stdin_exact_ordered_tokens'):
            raise ValueError('Explicit original-source stdin comparison required')
        comparison = stdin_comparison
    elif all(flags(actual, cases['outputs'], 'function_exact_JSON_value')):
        comparison = 'function_exact_JSON_value'
    elif all(isinstance(value, list) and len(value) == 1 for value in cases['outputs']):
        comparison = 'function_original_singleton_wrapper'
    else:
        return dict(result, status='original_reference_output_schema_or_value_qualification')
    passes = flags(actual, cases['outputs'], comparison)
    if not all(passes):
        return dict(result, status='original_reference_output_qualification', reference_case_passes=passes)
    return dict(result, reference_qualified=True, comparison=comparison,
        status='whole_original_reference_pass', reference_case_passes=passes)


def score_candidate(cases, observation, reference):
    identity = case_identity(cases)
    if (reference.get('reference_qualified') is not True
            or reference.get('complete_case_identity_sha256') != identity
            or reference.get('total_cases') != cases['total_cases']
            or reference.get('output_convention_fixed_before_candidate_execution') is not True):
        raise ValueError('Same complete original cases and prior successful reference qualification required')
    if (cases['mode'] == 'stdin') != reference['comparison'].startswith('stdin_'):
        raise ValueError('Original invocation mode cannot change after reference qualification')
    result = dict(complete_case_identity_sha256=identity, total_cases=cases['total_cases'],
        candidate_program_sha256=observation['original_program_sha256'],
        reference_program_sha256=reference['reference_program_sha256'], comparison=reference['comparison'],
        all_cases_observed=False, all_finite_cases_pass=False, independent_semantic_correctness_certified=False,
        RL_reward_admitted=False, SFT_admitted=False)
    actual = values(cases, observation)
    if actual is None:
        return dict(result, status='candidate_runtime_resource_or_incomplete_output_qualification')
    passes = flags(actual, cases['outputs'], reference['comparison'])
    return dict(result, status='all_complete_original_cases_pass' if all(passes) else 'completed_original_case_mismatch',
        all_cases_observed=True, all_finite_cases_pass=all(passes), case_passes=passes, passed_cases=sum(passes))


__all__ = ['contract', 'execute', 'implementation_sources', 'case_identity', 'qualify_reference', 'score_candidate']
