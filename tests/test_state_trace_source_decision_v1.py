import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer

from slm_lab.code_contests_Python_cases_v1 import execute_reference
from slm_lab.public_python_state_trace_v1 import capture
from slm_lab.state_trace_source_decision_v1 import assess


def fixture(program='n=int(input())\nprint(2*n)\n'):
    values=[('public_tests','7\n','14\n'),('private_tests','831742\n','1663484\n'),('generated_tests','-3\n','-6\n')]
    cases=[dict(kind=k,position=0,input=i,expected=o) for k,i,o in values]
    problem=SimpleNamespace(**{k:[SimpleNamespace(input=i,output=o)] for k,i,o in values})
    execution=execute_reference(program,cases)
    assert execution['all_original_finite_cases_pass']
    sha=hashlib.sha256(program.encode()).hexdigest()
    row=dict(split='train',publisher_split='train',language='Python3',response=program,
        original_program_sha256=sha,source_problem_id='unit-only:source',original_solution_position=0)
    reference=dict(original_problem_id=row['source_problem_id'],original_program_sha256=sha,
        original_solution_position=0,all_original_finite_cases_pass=True,actual_execution=execution)
    a=capture(program,'7\n');b=capture(program,'7\n')
    disposition=dict(status='whole_trace_native_candidate',original_public_case_position=0,
        original_public_input_sha256=hashlib.sha256(b'7\n').hexdigest())
    root=Path(__file__).resolve().parents[1];tokenizer=Tokenizer.from_file(str(root/'data/tokenizer-corpus/tokenizer.json'))
    return row,problem,reference,disposition,a,b,tokenizer


def test_real_all_case_reference_two_state_runs_and_own_native_mask_rebuild_without_private_values():
    args=fixture();result=assess(*args);native=result['target']
    assert result['status']=='whole_trace_native_candidate' and native['actual_stdout']=='14\n'
    assert native['ids'][0]==1 and native['ids'][-1]==2 and len(native['ids'])<=4096
    start=next(i for i,t in enumerate(native['labels']) if t!=-100)
    assert native['labels']==[-100]*start+native['ids'][start:]
    assert '831742' not in native['prompt']+native['response'] and '1663484' not in native['prompt']+native['response']
    broken=deepcopy(args);broken[2]['actual_execution']['actual_observed_case_count']=2
    with pytest.raises(ValueError,match='Every original'):assess(*broken)
    broken=deepcopy(args);broken[0]['publisher_split']='test'
    with pytest.raises(ValueError,match='screened training'):assess(*broken)


def test_changed_actual_repeat_or_raw_state_cannot_become_a_supervised_target():
    args=list(fixture());bad=deepcopy(args[5]);raw=bad['actual_observations']
    event=next(e for e in raw['events'] if 'n' in e['changes']);event['changes']['n']=['int','8']
    bad['raw_observation_stdout_b64']=base64.b64encode(json.dumps(raw).encode()).decode();args[5]=bad
    with pytest.raises(ValueError,match='disposition differs'):assess(*args)
    args[3]['status']='whole_state_or_stdout_not_reproduced_in_second_fresh_interpreter'
    assert assess(*args)['target'] is None
    bad['raw_observation_stdout_b64']=base64.b64encode(b'{}').decode()
    with pytest.raises(ValueError,match='transport'):assess(*args)


def test_whole_context_qualification_and_reference_failure_never_become_targets():
    program='# '+ ' '.join('distinct_identifier_'+str(i) for i in range(1000))+'\nn=int(input())\nprint(2*n)\n'
    args=list(fixture(program));args[3]['status']='complete_context_exceeds_4096';args[5]=None
    assert assess(*args)['target'] is None
    args=list(fixture());args[2]['all_original_finite_cases_pass']=False
    args[3]['status']='original_reference_not_fully_qualified';args[4]=None;args[5]=None
    assert assess(*args)['target'] is None
