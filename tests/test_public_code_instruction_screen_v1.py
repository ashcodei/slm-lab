import json
import pytest

from slm_lab.public_code_instruction_screen_v1 import UnsupportedExample,parse_whole,observe


def source(code, tests):
    return dict(id='fixture',input='A fixture question',output='```python\n'+code+'\n```',unit_tests=json.dumps(tests))


def test_literal_types_kwargs_and_every_case_survive():
    row=source('def f(x, *, n=1):\n    return (x, n)',[
        'assert f({1,2}, n=3) == ({1,2},3)',
        "assert f({(1,2):'v'}) == ({(1,2):'v'},1)"])
    parsed=parse_whole(row);actual=observe(parsed)
    assert parsed['program']==row['output'][10:-4]
    assert actual['all_cases_observed'] and actual['all_published_tests_passed']
    assert actual['whole_original_tests']==2 and not actual['expected_outputs_sent_to_worker']


def test_complete_test_set_is_never_filtered():
    parsed=parse_whole(source('def f(x): return x',['assert f(1)==1','assert f(2)==999']))
    result=observe(parsed)
    assert result['all_cases_observed'] and result['test_passes']==[True,False]
    assert not result['all_published_tests_passed']
    with pytest.raises(UnsupportedExample):
        parse_whole(source('def f(x): return x',['assert f(1)==1',"assert f(open('/etc/passwd'))==1"]))


def test_unsupported_assertions_abstain_whole_example():
    for test in ['assert f(1)', 'assert f(1)==2==2', 'x=1\nassert f(x)==1', 'assert f(*[1])==1',
                 'assert Solution().f(1)==1', 'assert f(1)==float("nan")']:
        with pytest.raises(UnsupportedExample):parse_whole(source('def f(x): return x',[test]))


def test_actual_os_confinement_blocks_host_file_and_network():
    code="""def f(x):
    if x == 0:
        return open('/etc/passwd').read()
    import socket
    return socket.socket()
"""
    for x in [0,1]:
        result=observe(parse_whole(source(code,[f'assert f({x})==0'])))
        assert result['status']=='runtime_resource_or_confinement_abstention'
        assert not result['all_published_tests_passed']


def test_infinite_program_abstains_without_false_incorrect_label():
    result=observe(parse_whole(source('def f(x):\n    while True: pass',['assert f(1)==1'])),seconds=.25)
    assert result['status']=='runtime_resource_or_confinement_abstention'
    assert not result['all_cases_observed']
