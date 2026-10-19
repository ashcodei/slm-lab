import base64
from copy import deepcopy
from types import SimpleNamespace as NS

import pytest

from slm_lab.code_contests_contrast_candidates_v1 import (
    counterexample, execute_negative, original_cases, select, sha,
)
from slm_lab.code_contests_Python_cases_v1 import execute_reference


class Alphabet:
    def encode(self, value):
        return NS(ids=[ord(c) + 4 for c in value])


@pytest.fixture(scope='module')
def observed():
    good = 'import sys\nn=int(sys.stdin.buffer.readline())\nprint(n*2)\n'
    bad = 'import sys\nn=int(sys.stdin.buffer.readline())\nprint(n+2)\n'
    problem = NS(
        public_tests=[NS(input='2\n', output='4\n')],
        private_tests=[NS(input='3\n', output='6\n')],
        generated_tests=[NS(input='987654321\n', output='1975308642\n')],
        solutions=[NS(language=3, solution=good)],
        incorrect_solutions=[NS(language=2, solution='not Python3'), NS(language=3, solution=bad)],
    )
    row = dict(id='publisher-source:accepted:0', source_problem_id='publisher-source',
               group_id='original-component', split='train', publisher_split='train', language='Python3',
               original_solution_position=0, original_program_sha256=sha(good), token_pack_index=5,
               prompt='Write a Python 3 program that doubles its integer input.', response=good)
    cases = original_cases(problem)
    positive = execute_reference(good, cases)
    reference = dict(original_problem_id=row['source_problem_id'], original_program_sha256=sha(good),
        original_solution_position=0, original_pack_position=5,
        complete_original_test_counts=positive['counts_by_original_kind'], complete_original_test_population=3,
        no_original_program_or_case_clipped=True, original_publisher_split='train', admitted_source_role='train',
        execution_reward_training_admitted=False, all_original_finite_cases_pass=True,
        status=positive['status'], actual_execution=positive)
    return row, problem, reference, cases, execute_negative(bad, cases)


def test_real_completed_counterexample_after_passing_public_case_and_full_native_EOS(observed):
    row, problem, reference, cases, actual = observed
    pair, why = select(row, problem, reference, Alphabet())
    assert why == 'complete_first_original_negative_ready_for_execution'
    assert pair['original_negative_solution_position'] == 1
    witness, why = counterexample(pair['rejected'], cases, actual)
    assert why == 'two_fresh_completed_output_counterexamples'
    assert witness['original_case_ordinal'] == 1
    assert witness['original_kind'] == 'private_tests'
    assert witness['two_fresh_completed_byte_identical_wrong_outputs']
    assert actual['first']['actual_observed_case_count'] == 2
    assert actual['repeat']['actual_observed_case_count'] == 1
    for arm in ['chosen', 'rejected']:
        ids, labels = pair[arm + '_ids'], pair[arm + '_labels']
        start = next(i for i, token in enumerate(labels) if token != -100)
        assert ids[0] == 1 and ids[-1] == 2 and labels == [-100] * start + ids[start:]
        assert len(ids) <= 4096
    text = pair['prompt'] + pair['chosen'] + pair['rejected']
    assert '987654321' not in text and '1975308642' not in text
    assert pair['optimization_admitted'] is False and pair['human_preferences'] is False


def test_reader_rejects_forged_raw_verdict_and_changed_original_case_and_qualifies_unstable_repeat(observed):
    row, problem, reference, cases, actual = observed
    bad = problem.incorrect_solutions[1].solution
    forged = deepcopy(actual)
    forged['first']['original_case_verdicts'][-1]['actual_output_tokens_equal_expected'] = True
    with pytest.raises(ValueError):
        counterexample(bad, cases, forged)
    changed = deepcopy(cases); changed[1]['expected'] = '5\n'
    with pytest.raises(ValueError):
        counterexample(bad, changed, actual)
    unstable = deepcopy(actual)
    unstable['repeat']['original_case_verdicts'][0]['actual_stdout_b64'] = base64.b64encode(b'999\n').decode()
    witness, why = counterexample(bad, cases, unstable)
    assert witness is None and why == 'fresh_completed_wrong_outputs_not_byte_identical'
    with pytest.raises(ValueError):
        select(dict(row, split='test'), problem, reference, Alphabet())
    changed_reference = deepcopy(reference)
    changed_reference['actual_execution']['actual_observed_case_count'] = 2
    with pytest.raises(ValueError):
        select(row, problem, changed_reference, Alphabet())


def test_no_completed_failure_no_label_and_first_native_overflow_never_falls_back(observed):
    row, problem, reference, cases, actual = observed
    no_failure = execute_negative(row['response'], cases)
    witness, why = counterexample(row['response'], cases, no_failure)
    assert witness is None and why == 'no_completed_original_case_output_counterexample'
    malformed = execute_negative('if :\n', cases)
    witness, why = counterexample('if :\n', cases, malformed)
    assert witness is None and why == 'no_completed_original_case_output_counterexample'
    long_first = deepcopy(problem)
    long_first.incorrect_solutions = [NS(language=3, solution='x=1\n' * 1500), problem.incorrect_solutions[1]]
    pair, why = select(row, long_first, reference, Alphabet())
    assert pair is None and why.startswith('whole_first_negative_native_qualification:')
    no_candidate = deepcopy(problem); no_candidate.incorrect_solutions = []
    assert select(row, no_candidate, reference, Alphabet())[1] == 'no_original_incorrect_Python3_source'
