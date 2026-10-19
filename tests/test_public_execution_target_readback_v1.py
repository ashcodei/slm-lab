import base64
import copy
import hashlib
from types import SimpleNamespace

import pytest

from slm_lab.public_execution_target_readback_v1 import rebuild


class BytesTokenizer:
    def encode(self,text):return SimpleNamespace(ids=[v+4 for v in text.encode()])


def fixture():
    sha=lambda text:hashlib.sha256(text.encode()).hexdigest()
    code='n=int(input())\nprint(2*n)\n'
    row=dict(split='train',publisher_split='train',language='Python3',source_problem_id='source:train',
        response=code,original_solution_position=3,original_program_sha256=sha(code))
    problem=SimpleNamespace(public_tests=[SimpleNamespace(input='7\n',output='14\n')],
        private_tests=[SimpleNamespace(input='PRIVATE_INPUT_POISON',output='PRIVATE_OUTPUT_POISON')],
        generated_tests=[SimpleNamespace(input='GENERATED_INPUT_POISON',output='GENERATED_OUTPUT_POISON')])
    actual=dict(kind='public_tests',original_kind_position=0,process_status='completed',
        original_input_sha256=sha('7\n'),original_expected_sha256=sha('14\n'),
        actual_stdout_b64=base64.b64encode(b'14  \n').decode())
    ref=dict(original_problem_id='source:train',original_program_sha256=sha(code),original_solution_position=3,
        all_original_finite_cases_pass=True,actual_execution=dict(counts_by_original_kind=dict(public_tests=1,private_tests=1,generated_tests=1),
            actual_observed_case_count=3,all_cases_observed=True,all_original_finite_cases_pass=True,
            original_case_verdicts=[actual,dict(kind='private_tests'),dict(kind='generated_tests')]))
    return row,problem,ref


def test_exact_actual_stdout_not_gold_preserved_and_private_values_never_formatted():
    row,problem,ref=fixture();got=rebuild(row,problem,ref,BytesTokenizer())
    assert got['response']=='14  \n' and got['response']!=problem.public_tests[0].output
    assert 'POISON' not in got['prompt']+got['response']
    assert row['response'] in got['prompt'] and problem.public_tests[0].input in got['prompt']
    start=next(i for i,v in enumerate(got['labels']) if v!=-100)
    assert got['labels'][:start]==[-100]*start and got['ids'][-1]==2
    assert bytes(v-4 for v in got['ids'][start:-1])==b'14  \n'


def test_no_clipping_and_first_whole_fitting_public_case_reconstructed():
    row,problem,ref=fixture()
    problem.public_tests[0].input='7\n'+' '*5000
    ref['actual_execution']['original_case_verdicts'][0]['original_input_sha256']=hashlib.sha256(problem.public_tests[0].input.encode()).hexdigest()
    assert rebuild(row,problem,ref,BytesTokenizer()) is None
    problem.public_tests.append(SimpleNamespace(input='7\n',output='14\n'))
    other=copy.deepcopy(ref['actual_execution']['original_case_verdicts'][0]);other['original_kind_position']=1
    other['original_input_sha256']=hashlib.sha256(b'7\n').hexdigest()
    ref['actual_execution']['original_case_verdicts'].insert(1,other)
    ref['actual_execution']['actual_observed_case_count']=4;ref['actual_execution']['counts_by_original_kind']['public_tests']=2
    assert rebuild(row,problem,ref,BytesTokenizer())['original_public_case_position']==1
    assert len(problem.public_tests[0].input)==5002


def test_wrong_source_role_output_or_partial_population_rejected():
    row,problem,ref=fixture()
    with pytest.raises(ValueError):rebuild(dict(row,split='validation'),problem,ref,BytesTokenizer())
    with pytest.raises(ValueError):rebuild(row,problem,dict(ref,original_problem_id='another'),BytesTokenizer())
    ref['actual_execution']['actual_observed_case_count']=1
    with pytest.raises(ValueError):rebuild(row,problem,ref,BytesTokenizer())
    row,problem,ref=fixture();ref['actual_execution']['original_case_verdicts'][0]['actual_stdout_b64']=base64.b64encode(b'999\n').decode()
    with pytest.raises(ValueError):rebuild(row,problem,ref,BytesTokenizer())
