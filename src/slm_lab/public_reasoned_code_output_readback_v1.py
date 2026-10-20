"""Independently check retained confined values against whole source cases.

Does not import the execution producer or either of its comparators. This
checks saved source/value arithmetic; it does not rerun programs or establish
OS confinement independently of the original execution evidence.
"""
import hashlib
import json
import math


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def identity(cases):
    keys = ('mode', 'function', 'inputs', 'outputs', 'total_cases')
    if (cases['mode'] not in ('stdin', 'function') or not isinstance(cases['inputs'], list)
            or not isinstance(cases['outputs'], list) or not 1 <= cases['total_cases'] <= 1000
            or len(cases['inputs']) != len(cases['outputs']) or len(cases['inputs']) != cases['total_cases']):
        raise ValueError('Every complete original case and invocation mode required')
    return sha(json.dumps({k: cases[k] for k in keys}, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False))


def equal(actual, expected, mode):
    if mode == 'stdin_exact_ordered_tokens':
        return isinstance(actual, str) and actual.encode().split() == expected.encode().split()
    if mode == 'stdin_outside_whitespace_only':
        return isinstance(actual, str) and actual.strip() == expected.strip()
    if mode == 'function_exact_JSON_value':
        return actual == expected
    if mode == 'function_original_singleton_wrapper':
        if not isinstance(expected, list) or len(expected) != 1:
            raise ValueError('Uniform original singleton output wrapper required')
        return actual == expected[0]
    raise ValueError('Unknown fixed output convention')


def observations(cases, program, observed):
    if (observed['original_program_sha256'] != sha(program)
            or observed['original_tests_sha256'] != cases['original_tests_sha256']
            or observed['mode'] != cases['mode'] or observed['total_cases'] != cases['total_cases']):
        raise ValueError('Actual unchanged program and full original source cases differ')
    if observed.get('all_cases_observed') is not True:
        if observed.get('all_finite_published_cases_pass') is not False:
            raise ValueError('Incomplete runtime cannot claim all-case success')
        return None
    process = observed['execution']
    raw = observed['raw_results']
    rows = raw['results']
    if (process['status'] != 'completed' or process['returncode'] != 0
            or raw['format'] != 'own-APPS-raw-results-v2' or len(rows) != cases['total_cases']
            or observed['syntax_valid'] is not True):
        raise ValueError('Every actual confined return value and completed process required')
    compile(program, '<independent-saved-source-syntax-check>', 'exec')
    for row in rows:
        if (set(row) != {'value', 'seconds'} or type(row['seconds']) not in (int, float)
                or not math.isfinite(row['seconds']) or row['seconds'] < 0):
            raise ValueError('Actual complete returned values and finite timings required')
    wire = json.dumps(raw, allow_nan=False) + '\n'
    if sha(wire) != process['stdout_sha256']:
        raise ValueError('Saved returned values do not reconstruct the harness stdout digest')
    actual = [row['value'] for row in rows]
    if cases['mode'] == 'stdin':
        legacy = [isinstance(a, str) and a.strip() == b.strip() for a, b in zip(actual, cases['outputs'])]
    else:
        legacy = [a == b or (isinstance(b, list) and bool(b) and a == b[0]) for a, b in zip(actual, cases['outputs'])]
    expected_status = 'all_finite_published_cases_pass' if all(legacy) else 'completed_published_case_mismatch'
    if (observed['case_passes'] != legacy or observed['passed_cases'] != sum(legacy)
            or observed['all_finite_published_cases_pass'] is not all(legacy)
            or observed['status'] != expected_status):
        raise ValueError('Retained legacy-runtime flags differ from their raw actual values')
    return actual


def verify_score(cases, program, observed, saved, reference_program, comparison):
    actual = observations(cases, program, observed)
    if (saved['complete_case_identity_sha256'] != identity(cases)
            or saved['candidate_program_sha256'] != sha(program)
            or saved['reference_program_sha256'] != sha(reference_program)
            or saved['comparison'] != comparison or saved['total_cases'] != cases['total_cases']
            or saved['independent_semantic_correctness_certified'] is not False
            or saved['SFT_admitted'] is not False or saved['RL_reward_admitted'] is not False):
        raise ValueError('Exact source, reference-fixed convention and unadmitted scope required')
    if actual is None:
        if (saved['status'] != 'candidate_runtime_resource_or_incomplete_output_qualification'
                or saved['all_cases_observed'] is not False or saved['all_finite_cases_pass'] is not False):
            raise ValueError('An incomplete candidate cannot receive a finite-case positive')
        return dict(complete=False, positive=False, checked_return_values=0)
    flags = [equal(a, b, comparison) for a, b in zip(actual, cases['outputs'])]
    expected_status = 'all_complete_original_cases_pass' if all(flags) else 'completed_original_case_mismatch'
    if (saved['case_passes'] != flags or saved['passed_cases'] != sum(flags)
            or saved['all_cases_observed'] is not True or saved['all_finite_cases_pass'] is not all(flags)
            or saved['status'] != expected_status):
        raise ValueError('Candidate finite-case verdicts do not match the retained actual values')
    return dict(complete=True, positive=all(flags), checked_return_values=len(actual))


def verify(cases, reference_program, candidate_program, reference, candidate, *, stdin_comparison):
    actual = observations(cases, reference_program, reference['original_reference_observation'])
    fixed = reference['fixed_output_convention']
    qualified = False
    comparison = None
    flags = None
    if actual is not None:
        if cases['mode'] == 'stdin':
            if stdin_comparison not in ('stdin_exact_ordered_tokens', 'stdin_outside_whitespace_only'):
                raise ValueError('Original source-specific stdin comparison required')
            proposed = stdin_comparison
        elif all(a == b for a, b in zip(actual, cases['outputs'])):
            proposed = 'function_exact_JSON_value'
        elif all(isinstance(b, list) and len(b) == 1 for b in cases['outputs']):
            proposed = 'function_original_singleton_wrapper'
        else:
            proposed = None
        if proposed is not None:
            flags = [equal(a, b, proposed) for a, b in zip(actual, cases['outputs'])]
            qualified = all(flags)
            comparison = proposed if qualified else None
    if (fixed['complete_case_identity_sha256'] != identity(cases)
            or fixed['reference_program_sha256'] != sha(reference_program)
            or fixed['total_cases'] != cases['total_cases'] or fixed['reference_qualified'] is not qualified
            or fixed['comparison'] != comparison
            or fixed['output_convention_fixed_before_candidate_execution'] is not True
            or (flags is not None and fixed['reference_case_passes'] != flags)):
        raise ValueError('Original reference qualification or fixed output convention differs')
    usable = False
    checked = len(actual) if actual is not None else 0
    if qualified:
        noop = 'pass\n' if cases['mode'] == 'stdin' else 'def ' + cases['function'] + '(*args, **kwargs):\n    return None\n'
        no_op = verify_score(cases, noop, reference['no_op_observation'], reference['no_op_score'], reference_program, comparison)
        usable = no_op['complete'] and not no_op['positive']
        checked += no_op['checked_return_values']
    elif reference['no_op_observation'] is not None or reference['no_op_score'] is not None:
        raise ValueError('Unqualified source must not claim a no-op scoring contract')
    if candidate['data_admitted_to_training'] is not False:
        raise ValueError('Execution evidence alone cannot grant training admission')
    if candidate['candidate_execution_attempted'] is not usable:
        raise ValueError('Candidate execution needs a passing reference and a complete failing no-op control')
    if not usable:
        if candidate['status'] != 'original_reference_or_no_op_population_qualification':
            raise ValueError('Unusable original runtime must abstain, never label a candidate incorrect')
        return dict(finite_case_positive_verified=False, checked_return_values=checked,
            original_source_runtime_qualified=False, no_new_execution=True, training_admitted=False)
    scored = verify_score(cases, candidate_program, candidate['candidate_observation'], candidate['candidate_score'],
        reference_program, comparison)
    if candidate['status'] != candidate['candidate_score']['status']:
        raise ValueError('Top-level and retained candidate status differ')
    return dict(finite_case_positive_verified=scored['positive'], checked_return_values=checked+scored['checked_return_values'],
        original_source_runtime_qualified=True, no_new_execution=True, training_admitted=False)
