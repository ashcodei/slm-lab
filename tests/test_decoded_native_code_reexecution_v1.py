from copy import deepcopy

import pytest

from slm_lab.code_tasks import TASKS,request
from slm_lab.code_quality import execute
from slm_lab.decoded_native_code_reexecution_v1 import reconstruct,replay


def case(code,language='Python'):
    t=TASKS[0]
    return dict(language=language,task=t[0],specification=t[2],raw_response=code,code=code,
        candidate=execute(request(t,language,code)),efficiency_stress_passed=False)


def test_exact_decoded_code_bytes_are_checked_before_any_reexecution():
    row=case(TASKS[0][4]);task,code=reconstruct(row,'instruction');assert code==TASKS[0][4]
    bad=deepcopy(row);bad['code']+='\n# repaired by auditor'
    with pytest.raises(ValueError,match='without repair'):reconstruct(bad,'instruction')
    bad=deepcopy(row);bad['specification']='different task'
    with pytest.raises(ValueError,match='specification'):reconstruct(bad,'instruction')


def test_actual_fresh_python_javascript_and_ordinary_cases_distinguish_correct_and_semantically_wrong_code():
    for language,code in [('Python',TASKS[0][4]),('JavaScript',TASKS[0][5])]:
        row=replay(case(code,language),'instruction')
        assert row['fresh_native_execution']['functional_passed']
        if language=='Python':assert row['fresh_ordinary_Python_execution']['functional_passed']
    row=replay(case('def sum_even(xs):\n    return len(xs)'),'instruction')
    assert row['fresh_native_execution']['syntax_valid'] and not row['fresh_native_execution']['functional_passed']
    assert not row['fresh_ordinary_Python_execution']['functional_passed']
