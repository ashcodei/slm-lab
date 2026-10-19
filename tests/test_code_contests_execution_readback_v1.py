import copy
import hashlib
from types import SimpleNamespace

import pytest

from slm_lab.code_contests_execution_readback_v1 import verify
from slm_lab.code_contests_Python_cases_v1 import execute_reference


def actual(program='import sys\nprint(2*int(sys.stdin.buffer.readline()))\n',*,mismatch=False):
    problem=SimpleNamespace(public_tests=[SimpleNamespace(input='3\n',output='6\n')],
        private_tests=[SimpleNamespace(input='7\n',output='14\n')],
        generated_tests=[SimpleNamespace(input='4\n',output='WRONG\n' if mismatch else '8\n')])
    row=dict(response=program,language='Python3',split='train',publisher_split='train',
        original_program_sha256=hashlib.sha256(program.encode()).hexdigest(),source_problem_id='problem:train:1',
        original_solution_position=8,token_pack_index=19)
    cases=[dict(kind=k,position=i,input=c.input,expected=c.output) for k in ('public_tests','private_tests','generated_tests')
        for i,c in enumerate(getattr(problem,k))]
    execution=execute_reference(program,cases)
    result=dict(original_problem_id=row['source_problem_id'],original_program_sha256=row['original_program_sha256'],
        original_solution_position=8,original_pack_position=19,complete_original_test_counts=execution['counts_by_original_kind'],
        complete_original_test_population=3,no_original_program_or_case_clipped=True,
        original_publisher_split='train',admitted_source_role='train',execution_reward_training_admitted=False,
        all_original_finite_cases_pass=execution['all_original_finite_cases_pass'],status=execution['status'],actual_execution=execution)
    return row,problem,result


def test_actual_all_three_kind_outputs_rebuilt_and_corrupted_output_or_source_rejected():
    row,problem,result=actual()
    checked=verify(row,problem,result)
    assert checked['fully_qualified'] and checked['actual_retained_case_outputs_checked']==3
    damaged=copy.deepcopy(result);damaged['actual_execution']['original_case_verdicts'][1]['actual_stdout_b64']='OTkK'
    with pytest.raises(ValueError):verify(row,problem,damaged)
    damaged=copy.deepcopy(result);damaged['original_program_sha256']='0'*64
    with pytest.raises(ValueError):verify(row,problem,damaged)


def test_actual_last_case_mismatch_cannot_become_full_success_or_be_silently_dropped():
    row,problem,result=actual(mismatch=True)
    checked=verify(row,problem,result)
    assert not checked['fully_qualified'] and checked['actual_retained_case_outputs_checked']==3
    damaged=copy.deepcopy(result);damaged['all_original_finite_cases_pass']=True
    with pytest.raises(ValueError):verify(row,problem,damaged)
    damaged=copy.deepcopy(result);damaged['actual_execution']['original_case_verdicts'].pop()
    with pytest.raises(ValueError):verify(row,problem,damaged)


def test_real_syntax_qualification_preserved_and_all_original_counts_rebuilt():
    row,problem,result=actual(program='def broken(:\n')
    checked=verify(row,problem,result)
    assert not checked['fully_qualified'] and checked['actual_retained_case_outputs_checked']==0
    damaged=copy.deepcopy(result);damaged['complete_original_test_counts']['private_tests']=0
    with pytest.raises(ValueError):verify(row,problem,damaged)
