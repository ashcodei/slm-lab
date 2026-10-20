import pytest

from slm_lab.codeio_published_cases_v1 import contract,execute as fresh_execute
from slm_lab.codeio_whole_cases_v2 import execute


def test_whole_cached_cases_match_independent_fresh_interpreters_and_reset_module_state():
    cases=contract(dict(refcode='counter=0\ndef main_solution(*,x):\n    global counter\n    counter+=1\n    return {"result": [x+counter, True]}\n',
        funcname='main_solution',ios=[dict(input={'x':x},output={'result':[x+1,True]}) for x in [2,9,-1]]))
    for seed in [0,1]:
        fresh=fresh_execute(cases,seed=seed,seconds=10);whole=execute(cases,seed=seed,seconds=10)
        assert fresh['all_original_cached_values_match'] and whole['all_original_cached_values_match']
        assert [r['returned']['value'] for r in fresh['records']]==[r['value'] for r in whole['records']]
        assert whole['fresh_module_namespace_per_case'] and not whole['fresh_interpreter_per_case']
        assert whole['expected_outputs_not_sent_to_children'] and not whole['training_or_RL_admitted']
    wrong=execute(cases,seed=0,program='def main_solution(**kwargs):\n    return None\n')
    assert wrong['all_original_cases_observed'] and wrong['case_passes']==[False]*3


def test_incomplete_or_unsupported_programs_abstain_without_partial_pass_scores(tmp_path):
    forbidden=tmp_path/'secret.txt';forbidden.write_text('unreadable')
    cases=contract(dict(refcode='def main_solution(x):\n    return x\n',funcname='main_solution',
        ios=[dict(input={'x':1},output=1),dict(input={'x':2},output=2)]))
    for code in ['import numpy\ndef main_solution(x):\n    return x\n',
            'def main_solution(x):\n    return open('+repr(str(forbidden))+').read()\n',
            'def main_solution(x):\n    if x==2: raise ValueError("later case")\n    return x\n']:
        result=execute(cases,seed=0,seconds=10,program=code)
        assert not result['all_original_cases_observed'] and result['case_passes'] is None
        assert result['records']==[] and not result['all_original_cached_values_match']
    with pytest.raises(ValueError):execute(cases,seed=True)
