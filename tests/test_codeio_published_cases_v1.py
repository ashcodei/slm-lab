import json

import pytest

from slm_lab.codeio_published_cases_v1 import contract,equal_json,execute,strict_loads


def test_original_keyword_and_nested_dict_returns_are_checked_in_fresh_processes():
    row=dict(refcode='count=0\ndef main_solution(*,x):\n    global count\n    count+=1\n    return {"value":[x+count,True]}\n',
        funcname='main_solution',ios=[dict(input={'x':2},output={'value':[3,True]}),dict(input={'x':9},output={'value':[10,True]})])
    cases=contract(row)
    for seed in (0,1):
        result=execute(cases,seed=seed,seconds=10)
        assert result['all_original_cached_values_match'] and result['fresh_process_per_case']
        assert result['case_passes']==[True,True] and not result['training_or_RL_admitted']
    wrong=execute(cases,seed=0,seconds=10,program='def main_solution(**kwargs):\n    return None\n')
    assert wrong['all_original_cases_observed'] and not wrong['all_original_cached_values_match']


def test_JSON_types_and_original_nonfinite_or_duplicate_keys_are_not_reinterpreted():
    assert not equal_json(True,1) and not equal_json(False,0)
    assert not equal_json([1],1) and not equal_json({'a':1},{'a':'1'})
    assert equal_json({'a':[1.,2]}, {'a':[1,2]})
    assert equal_json(10**400,10**400) and not equal_json(10**400,10**400+1)
    for raw in ['{"a":1,"a":2}','{"a":NaN}','{"a":1e999}']:
        with pytest.raises(ValueError):strict_loads(raw)
    with pytest.raises(ValueError):contract(dict(refcode='pass',funcname='main_solution',ios=[]))


def test_unavailable_dependencies_and_forbidden_external_file_reads_abstain(tmp_path):
    secret=tmp_path/'unreadable-value.txt';secret.write_text('not granted to the confined child')
    cases=contract(dict(refcode='def main_solution():\n    return 3\n',funcname='main_solution',ios=[dict(input={},output=3)]))
    for code in ['import numpy\ndef main_solution():\n    return 3\n',
            'def main_solution():\n    return open('+repr(str(secret))+').read()\n']:
        result=execute(cases,seed=0,seconds=10,program=code)
        assert not result['all_original_cases_observed'] and not result['all_original_cached_values_match']
        assert result['case_passes'] is None and result['status']=='source_runtime_or_resource_qualification'
