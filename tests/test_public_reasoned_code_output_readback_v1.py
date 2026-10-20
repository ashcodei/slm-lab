from copy import deepcopy
import json

import pytest

from slm_lab.public_reasoned_code_cases_v1 import contract, execute, qualify_reference, score_candidate
from slm_lab.public_reasoned_code_output_readback_v1 import verify


@pytest.fixture(scope='module')
def actual_outputs():
    cases = contract(json.dumps(dict(inputs=[[1], [5]], outputs=[[2], [10]], fn_name='twice')).encode())
    source = 'def twice(x):\n    return x*2\n'
    noop = 'def twice(*args, **kwargs):\n    return None\n'
    candidate = 'class Solution:\n    def twice(self,x):\n        return x*2\n'
    original = execute(source, cases, seconds=5)
    fixed = qualify_reference(cases, original)
    empty = execute(noop, cases, seconds=5)
    reference = dict(original_reference_observation=original, fixed_output_convention=fixed,
        no_op_observation=empty, no_op_score=score_candidate(cases, empty, fixed))
    observed = execute(candidate, cases, seconds=5)
    record = dict(data_admitted_to_training=False, candidate_execution_attempted=True,
        candidate_observation=observed, candidate_score=score_candidate(cases, observed, fixed),
        status='all_complete_original_cases_pass')
    return cases, source, candidate, reference, record


def check(parts):
    return verify(*parts, stdin_comparison='stdin_outside_whitespace_only')


def test_actual_complete_outputs_and_noop_are_rebuilt(actual_outputs):
    result = check(actual_outputs)
    assert result['finite_case_positive_verified']
    assert result['checked_return_values'] == 6
    assert result['no_new_execution'] and not result['training_admitted']


def test_missing_outputs_forged_flags_changed_program_or_cases_reject(actual_outputs):
    for mutation in ('drop_value', 'forged_flag', 'changed_program', 'changed_case'):
        parts = deepcopy(actual_outputs)
        if mutation == 'drop_value':
            parts[4]['candidate_observation']['raw_results']['results'].pop()
        elif mutation == 'forged_flag':
            parts[4]['candidate_score']['case_passes'][0] = False
        elif mutation == 'changed_program':
            parts = (parts[0], parts[1], parts[2] + '#different bytes\n', parts[3], parts[4])
        else:
            parts[0]['outputs'][0] = [3]
        with pytest.raises(ValueError):
            check(parts)


def test_wrong_shape_and_fake_noop_or_admission_cannot_pass(actual_outputs):
    cases, source, _, reference, _ = deepcopy(actual_outputs)
    wrong = 'def twice(x):\n    return [x*2]\n'
    observed = execute(wrong, cases, seconds=5)
    score = score_candidate(cases, observed, reference['fixed_output_convention'])
    record = dict(data_admitted_to_training=False, candidate_execution_attempted=True,
        candidate_observation=observed, candidate_score=score, status=score['status'])
    result = verify(cases, source, wrong, reference, record, stdin_comparison='stdin_outside_whitespace_only')
    assert not result['finite_case_positive_verified']
    for mutation in ('no_op', 'admission'):
        parts = deepcopy(actual_outputs)
        if mutation == 'no_op':
            parts[3]['no_op_score']['all_finite_cases_pass'] = True
        else:
            parts[4]['data_admitted_to_training'] = True
        with pytest.raises(ValueError):
            check(parts)
