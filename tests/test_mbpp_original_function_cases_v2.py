import pytest

from slm_lab.finite_Python_values_v2 import decode,encode
from slm_lab.mbpp_original_function_cases_v2 import cases,execute


def test_real_globals_class_input_setup_stdlib_and_exact_container_complex_outputs():
    source=('import math\nOFFSET=3\nclass Node:\n    def __init__(self,value): self.value=value\n'
        'def answer(node,mode):\n    if mode==0: return (node.value+OFFSET,{1,2},{("a",1):2},1+2j)\n'
        '    return [math.isqrt(node.value),node.value]\n')
    row=dict(task_id=1,test_setup_code='root=Node(4)\n',test_list=[
        'assert answer(root,0) == (7, {1,2}, {("a",1):2}, 1+2j)',
        'assert answer(root,1) == [2,4]'],challenge_test_list=['assert answer(root,2) == [2,4]'])
    result=execute(source,row)
    assert result['functional_passed'] and result['actual_observed_cases']==3 and result['original_challenge_cases']==1
    value=(b'a',{('key',2):[1.25,True,None]},frozenset({2,3}),2+3j)
    assert decode(encode(value))==value
    with pytest.raises(ValueError):decode(['module','os.system'])


def test_actual_worker_settings_contain_only_inputs_and_tuple_list_comparison_is_not_coerced():
    source=('import json\ndef settings_keys():\n    return sorted(json.load(open("settings.json")))\n'
        'def pair(): return (1,2)\n')
    row=dict(task_id=2,test_setup_code='',test_list=[
        'assert settings_keys() == ["expressions","program","setup"]'],
        challenge_test_list=['assert pair() == [1,2]'])
    result=execute(source,row)
    assert result['all_cases_observed'] and result['passed_cases']==1 and not result['functional_passed']
    assert result['expected_RHS_not_sent_to_confined_worker']
    assert [r['passed'] for r in result['original_case_verdicts']]==[True,False]


def test_infinite_program_retains_every_original_case_and_nonliteral_parent_code_is_rejected():
    row=dict(task_id=3,test_setup_code='',test_list=['assert answer(1) == 2','assert answer(2) == 3'],challenge_test_list=[])
    result=execute('def answer(n):\n    while True: pass\n',row,seconds=.2)
    assert result['original_cases']==2 and not result['functional_passed'] and not result['all_cases_observed']
    assert result['execution']['status']=='wall_deadline_exceeded'
    with pytest.raises(ValueError):cases(dict(row,test_list=['assert answer(1) == __import__("os").getcwd()']))


def test_actual_Counter_return_preserves_original_mapping_equality_and_nested_Counter_values():
    from collections import Counter
    row=dict(task_id=40,test_setup_code='',test_list=['assert freq([1,2,1]) == {1:2,2:1}'],challenge_test_list=[])
    result=execute('from collections import Counter\ndef freq(xs): return Counter(xs)\n',row)
    assert result['functional_passed'] and result['actual_observed_cases']==1
    value=[Counter({1:2,2:1}),{3:Counter({'a':2})}]
    restored=decode(encode(value))
    assert type(restored[0]) is Counter and restored==value
