"""Whole public train programs with separately observed output counterexamples.

These are machine preferences, never new human judgments. Merely carrying a
publisher 'incorrect' label is insufficient: a completed wrong output must
recur in a second fresh interpreter. Timeout/version failures are qualifications.
This module produces candidates only; it grants no optimization admission.
"""
import base64
import hashlib
import math

from .code_contests_curriculum_candidates_v1 import native_sequence
from .code_contests_execution_readback_v1 import verify
from .code_contests_Python_cases_v1 import execute_reference

KINDS = ('public_tests', 'private_tests', 'generated_tests')


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def original_cases(problem):
    return [dict(kind=kind, position=i, input=case.input, expected=case.output)
            for kind in KINDS for i, case in enumerate(getattr(problem, kind))]


def select(row, problem, reference, tokenizer):
    """Choose the first original Python3 negative before observing its behavior."""
    checked = verify(row, problem, reference)
    if not checked['fully_qualified']:
        return None, 'original_reference_not_qualified'
    original = problem.solutions[row['original_solution_position']]
    if int(original.language) != 3 or original.solution != row['response']:
        raise ValueError('The positive must be the full original accepted Python3 source')
    position = next((i for i, item in enumerate(problem.incorrect_solutions)
                     if int(item.language) == 3), None)
    if position is None:
        return None, 'no_original_incorrect_Python3_source'
    rejected = problem.incorrect_solutions[position].solution
    if not rejected or '\0' in rejected or len(rejected.encode()) > 32768:
        return None, 'whole_first_negative_exceeds_runtime_program_contract'
    if rejected == row['response']:
        return None, 'first_negative_is_byte_identical_to_original_positive'
    chosen_native, why = native_sequence(tokenizer, row['prompt'], row['response'])
    if chosen_native is None:
        raise ValueError('Already screened whole positive no longer fits: ' + why)
    rejected_native, why = native_sequence(tokenizer, row['prompt'], rejected)
    if rejected_native is None:
        return None, 'whole_first_negative_native_qualification:' + why
    return dict(
        id=row['source_problem_id'] + ':execution-contrast:first-Python3-negative',
        source_problem_id=row['source_problem_id'], group_id=row['group_id'],
        split='train', publisher_split='train', language='Python3',
        prompt=row['prompt'], chosen=row['response'], rejected=rejected,
        chosen_ids=chosen_native[0], chosen_labels=chosen_native[1],
        rejected_ids=rejected_native[0], rejected_labels=rejected_native[1],
        original_positive_solution_position=row['original_solution_position'],
        original_negative_solution_position=position,
        original_positive_program_sha256=sha(row['response']),
        original_negative_program_sha256=sha(rejected),
        original_positive_pack_position=row['token_pack_index'],
        original_reference_all_finite_cases_passed=True,
        first_negative_selected_before_actual_output_observation=True,
        private_or_generated_cases_appended_to_model_text=False,
        human_preferences=False, genuine_human_RLHF=False,
        optimization_admitted=False,
    ), 'complete_first_original_negative_ready_for_execution'


def execute_negative(program, cases):
    """Observe the original case prefix, then repeat its first completed mismatch."""
    first = execute_reference(program, cases, seconds_per_case=3., seconds_total=120.)
    result = dict(first=first, repeat=None, repeated_original_case_ordinal=None)
    rows = first['original_case_verdicts']
    if (first['status'] == 'native_reference_runtime_or_output_qualification' and rows
            and rows[-1]['process_status'] == 'completed'
            and rows[-1]['actual_output_tokens_equal_expected'] is False):
        ordinal = rows[-1]['original_case_ordinal']
        source = cases[ordinal]
        # The repeated one-case population has local position0; its original
        # case kind/position/ordinal and hashes are separately retained below.
        case = dict(source, position=0)
        result['repeat'] = execute_reference(program, [case], seconds_per_case=3., seconds_total=120.)
        result['repeated_original_case_ordinal'] = ordinal
    return result


def _read_observation(program, cases, observation):
    """Recompute raw-byte verdicts without the executor's comparator/contract."""
    digest = hashlib.sha256()
    counts = {kind: 0 for kind in KINDS}
    total_bytes = 0
    for case in cases:
        kind = case['kind']
        if kind not in counts or case['position'] != counts[kind]:
            raise ValueError('Whole original case order and local positions required')
        counts[kind] += 1
        a, b = case['input'].encode(), case['expected'].encode()
        total_bytes += len(a) + len(b)
        if max(len(a), len(b)) > 4 * 1024**2:
            raise ValueError('An original case exceeds the unmodified protocol')
        for value in (kind.encode(), case['position'].to_bytes(8, 'little'), a, b):
            digest.update(len(value).to_bytes(8, 'little')); digest.update(value)
    if (not 1 <= len(cases) <= 1000 or total_bytes > 8 * 1024**2
            or observation['complete_original_tests_sha256'] != digest.hexdigest()
            or observation['counts_by_original_kind'] != counts
            or observation['total_cases'] != len(cases)
            or observation['complete_original_input_and_expected_UTF8_bytes'] != total_bytes
            or observation['informative_nonempty_expected_cases'] != sum(bool(c['expected'].encode().split()) for c in cases)
            or observation['original_program_sha256'] != sha(program)
            or observation['expected_outputs_never_written_to_candidate_readable_filesystem'] is not True):
        raise ValueError('Exact complete original cases, source, bounds and expected-value separation required')
    try:
        compile(program, '<independent-negative-readback>', 'exec'); syntax = True
    except (SyntaxError, ValueError, RecursionError):
        syntax = False
    rows = observation['original_case_verdicts']
    if (observation['syntax_valid'] is not syntax or len(rows) > len(cases)
            or observation['actual_observed_case_count'] != len(rows) or (not syntax and rows)):
        raise ValueError('Actual original syntax and retained case prefix differ')
    flags = []
    for ordinal, actual in enumerate(rows):
        case = cases[ordinal]
        stdout = base64.b64decode(actual['actual_stdout_b64'], validate=True)
        stderr = base64.b64decode(actual['actual_stderr_b64'], validate=True)
        equal = stdout.split() == case['expected'].encode().split()
        if (actual['original_case_ordinal'] != ordinal or actual['kind'] != case['kind']
                or actual['original_kind_position'] != case['position']
                or actual['original_input_sha256'] != sha(case['input'])
                or actual['original_expected_sha256'] != sha(case['expected'])
                or actual['input_bytes'] != len(case['input'].encode())
                or not 0 <= actual['input_bytes_sent'] <= actual['input_bytes']
                or actual['actual_output_tokens_equal_expected'] is not equal
                or not math.isfinite(actual['seconds']) or actual['seconds'] < 0
                or actual['process_status'] == 'completed' and actual['returncode'] != 0):
            raise ValueError('A retained actual result contradicts the original raw-byte case verdict')
        flags.append(actual['process_status'] == 'completed' and equal)
        if ordinal + 1 < len(rows) and not flags[-1]:
            raise ValueError('The original execution prefix must end at its first qualification')
    return rows, flags


def counterexample(program, cases, execution):
    rows, flags = _read_observation(program, cases, execution['first'])
    first = execution['first']
    if first['status'] != 'native_reference_runtime_or_output_qualification' or not rows:
        return None, 'no_completed_original_case_output_counterexample'
    last = rows[-1]
    if last['process_status'] != 'completed' or last['actual_output_tokens_equal_expected'] is not False:
        return None, 'runtime_or_output_budget_qualification_not_output_counterexample'
    ordinal = len(rows) - 1
    if (first['first_unqualified_original_case_ordinal'] != ordinal or flags[-1]
            or not all(flags[:-1]) or first['all_original_finite_cases_pass'] is not False
            or first['all_cases_observed'] is not (len(rows) == len(cases))
            or execution['repeated_original_case_ordinal'] != ordinal):
        raise ValueError('The actual first completed mismatch and original prefix must agree')
    if execution['repeat'] is None:
        raise ValueError('A completed mismatch must retain its actual second fresh observation')
    repeated, repeated_flags = _read_observation(program, [dict(cases[ordinal], position=0)], execution['repeat'])
    second = execution['repeat']
    if (second['status'] != 'native_reference_runtime_or_output_qualification' or len(repeated) != 1
            or repeated_flags != [False] or repeated[0]['process_status'] != 'completed'):
        return None, 'second_fresh_observation_does_not_complete_with_wrong_output'
    if (last['actual_stdout_b64'] != repeated[0]['actual_stdout_b64']
            or last['actual_stderr_b64'] != repeated[0]['actual_stderr_b64']):
        return None, 'fresh_completed_wrong_outputs_not_byte_identical'
    case = cases[ordinal]
    return dict(original_case_ordinal=ordinal, original_kind=case['kind'],
                original_kind_position=case['position'], original_input_sha256=sha(case['input']),
                original_expected_sha256=sha(case['expected']),
                actual_wrong_stdout_sha256=hashlib.sha256(base64.b64decode(last['actual_stdout_b64'], validate=True)).hexdigest(),
                actual_original_negative_prefix_cases=len(rows),
                two_fresh_completed_byte_identical_wrong_outputs=True,
                all_other_original_negative_cases_not_claimed_executed=True,
                publisher_judge_or_general_incorrectness_not_claimed=True), 'two_fresh_completed_output_counterexamples'
