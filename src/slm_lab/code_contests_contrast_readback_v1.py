"""Independent source/native/raw-output reconstruction of machine preferences.

Does not import the candidate selector, its native formatter, execution
wrapper, counterexample reader, or comparator. Actual interpreter observations
are retained evidence; this reader does not reexecute or recertify confinement.
"""
import base64
import hashlib
import math

from .code_contests_execution_readback_v1 import verify

KINDS = ('public_tests', 'private_tests', 'generated_tests')


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def reconstruct_pair(row, problem, reference, tokenizer):
    if not verify(row, problem, reference)['fully_qualified']:
        return None, 'original_reference_not_qualified'
    original = problem.solutions[row['original_solution_position']]
    if int(original.language) != 3 or original.solution != row['response']:
        raise ValueError('Every original positive program must match its selected source position')
    selected = [(i, source.solution) for i, source in enumerate(problem.incorrect_solutions)
                if int(source.language) == 3]
    if not selected:
        return None, 'no_original_incorrect_Python3_source'
    index, negative = selected[0]
    if not negative or '\0' in negative or len(negative.encode()) > 32768:
        return None, 'whole_first_negative_exceeds_runtime_program_contract'
    if negative == row['response']:
        return None, 'first_negative_is_byte_identical_to_original_positive'
    prefix = [1, *tokenizer.encode('User: ' + row['prompt'] + '\nAssistant:').ids]
    values = {}
    for arm, response in [('chosen', row['response']), ('rejected', negative)]:
        tokens = tokenizer.encode(response).ids
        reason = ('source_contains_reserved_native_token' if any(v < 4 for v in prefix[1:] + tokens)
                  else 'empty_original_program' if not tokens else
                  'complete_context_exceeds_4096' if len(prefix) + len(tokens) + 1 > 4096 else None)
        if reason:
            if arm == 'chosen':
                raise ValueError('An unchanged screened positive no longer fits')
            return None, 'whole_first_negative_native_qualification:' + reason
        values[arm + '_ids'] = prefix + tokens + [2]
        values[arm + '_labels'] = [-100] * len(prefix) + tokens + [2]
    return dict(
        id=row['source_problem_id'] + ':execution-contrast:first-Python3-negative',
        source_problem_id=row['source_problem_id'], group_id=row['group_id'],
        split='train', publisher_split='train', language='Python3', prompt=row['prompt'],
        chosen=row['response'], rejected=negative, **values,
        original_positive_solution_position=row['original_solution_position'],
        original_negative_solution_position=index, original_positive_program_sha256=sha(row['response']),
        original_negative_program_sha256=sha(negative), original_positive_pack_position=row['token_pack_index'],
        original_reference_all_finite_cases_passed=True,
        first_negative_selected_before_actual_output_observation=True,
        private_or_generated_cases_appended_to_model_text=False, human_preferences=False,
        genuine_human_RLHF=False, optimization_admitted=False,
    ), 'complete_first_original_negative_ready_for_execution'


def _raw(program, cases, result):
    digest = hashlib.sha256(); counts = {k: 0 for k in KINDS}; byte_count = 0
    for kind, position, source in cases:
        if position != counts[kind]:
            raise ValueError('Every original contiguous case position is required')
        counts[kind] += 1
        a, b = source.input.encode(), source.output.encode(); byte_count += len(a) + len(b)
        if max(len(a), len(b)) > 4 * 1024**2:
            raise ValueError('Complete case exceeds the original runtime bound')
        for value in [kind.encode(), position.to_bytes(8, 'little'), a, b]:
            digest.update(len(value).to_bytes(8, 'little')); digest.update(value)
    if (not 1 <= len(cases) <= 1000 or byte_count > 8 * 1024**2
            or result['complete_original_tests_sha256'] != digest.hexdigest()
            or result['original_program_sha256'] != sha(program)
            or result['total_cases'] != len(cases) or result['counts_by_original_kind'] != counts
            or result['complete_original_input_and_expected_UTF8_bytes'] != byte_count
            or result['informative_nonempty_expected_cases'] != sum(bool(c.output.encode().split()) for _, _, c in cases)
            or result['expected_outputs_never_written_to_candidate_readable_filesystem'] is not True):
        raise ValueError('Every original raw test and unchanged negative program must agree')
    try:
        compile(program, '<negative-source-independent-reader>', 'exec'); syntax = True
    except (SyntaxError, ValueError, RecursionError):
        syntax = False
    observations = result['original_case_verdicts']
    if (result['syntax_valid'] is not syntax or len(observations) > len(cases)
            or result['actual_observed_case_count'] != len(observations) or (observations and not syntax)):
        raise ValueError('Every retained observed case needs its original source and syntax')
    flags = []
    for ordinal, observation in enumerate(observations):
        kind, position, source = cases[ordinal]
        output = base64.b64decode(observation['actual_stdout_b64'], validate=True)
        base64.b64decode(observation['actual_stderr_b64'], validate=True)
        equal = output.split() == source.output.encode().split()
        if (observation['original_case_ordinal'] != ordinal or observation['kind'] != kind
                or observation['original_kind_position'] != position
                or observation['original_input_sha256'] != sha(source.input)
                or observation['original_expected_sha256'] != sha(source.output)
                or observation['actual_output_tokens_equal_expected'] is not equal
                or observation['input_bytes'] != len(source.input.encode())
                or not 0 <= observation['input_bytes_sent'] <= observation['input_bytes']
                or not math.isfinite(observation['seconds']) or observation['seconds'] < 0
                or observation['process_status'] == 'completed' and observation['returncode'] != 0):
            raise ValueError('An actual raw stdout result or complete original case identity differs')
        flags.append(observation['process_status'] == 'completed' and equal)
        if ordinal and not flags[ordinal - 1]:
            raise ValueError('Observed prefix continues after its first qualification')
    return observations, flags


def reconstruct_witness(program, problem, execution):
    cases = [(kind, i, item) for kind in KINDS for i, item in enumerate(getattr(problem, kind))]
    first = execution['first']; rows, flags = _raw(program, cases, first)
    if first['status'] != 'native_reference_runtime_or_output_qualification' or not rows:
        return None, 'no_completed_original_case_output_counterexample'
    last = rows[-1]
    if last['process_status'] != 'completed' or last['actual_output_tokens_equal_expected'] is not False:
        return None, 'runtime_or_output_budget_qualification_not_output_counterexample'
    ordinal = len(rows) - 1
    if (first['first_unqualified_original_case_ordinal'] != ordinal or flags[-1] or not all(flags[:-1])
            or first['all_original_finite_cases_pass'] is not False
            or first['all_cases_observed'] is not (len(rows) == len(cases))
            or execution['repeated_original_case_ordinal'] != ordinal or execution['repeat'] is None):
        raise ValueError('The first completed original mismatch must retain a second actual observation')
    kind, position, source = cases[ordinal]
    second = execution['repeat']; repeated, repeated_flags = _raw(program, [(kind, 0, source)], second)
    if (second['status'] != 'native_reference_runtime_or_output_qualification'
            or len(repeated) != 1 or repeated_flags != [False] or repeated[0]['process_status'] != 'completed'):
        return None, 'second_fresh_observation_does_not_complete_with_wrong_output'
    if any(last[field] != repeated[0][field] for field in ['actual_stdout_b64', 'actual_stderr_b64']):
        return None, 'fresh_completed_wrong_outputs_not_byte_identical'
    return dict(original_case_ordinal=ordinal, original_kind=kind, original_kind_position=position,
                original_input_sha256=sha(source.input), original_expected_sha256=sha(source.output),
                actual_wrong_stdout_sha256=hashlib.sha256(base64.b64decode(last['actual_stdout_b64'], validate=True)).hexdigest(),
                actual_original_negative_prefix_cases=len(rows), two_fresh_completed_byte_identical_wrong_outputs=True,
                all_other_original_negative_cases_not_claimed_executed=True,
                publisher_judge_or_general_incorrectness_not_claimed=True), 'two_fresh_completed_output_counterexamples'
