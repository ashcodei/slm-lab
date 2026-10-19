import json
from pathlib import Path

import pytest

from slm_lab.apps_published_cases_v2 import contract, execute


def cases(inputs, outputs, fn=None):
    return contract(json.dumps(dict(inputs=inputs, outputs=outputs, fn_name=fn)).encode())


def test_actual_complete_stdin_class_and_recursive_source():
    source = 'import sys\na=list(map(int, sys.stdin.buffer.read().split()))\nprint(sum(a))\n'
    result = execute(source, cases(['1 2 3\n', '4 -9\n'], ['6\n', '-5\n']))
    assert result['all_finite_published_cases_pass'] and result['all_cases_observed']
    assert result['passed_cases'] == result['total_cases'] == 2
    result = execute('class Solution:\n def twice(self, xs: List[int]):\n  return [x*2 for x in xs]\n',
        cases([[[1, 2]], [[-2]]], [[[2, 4]], [[-4]]], 'twice'))
    assert result['all_finite_published_cases_pass'] and result['passed_cases'] == 2
    result = execute('def fact(n):\n return 1 if n==0 else n*fact(n-1)\n',
        cases([[0], [5]], [1, 120], 'fact'))
    assert result['all_finite_published_cases_pass']


def test_wrong_answers_forged_pass_missing_cases_and_exit_do_not_pass():
    c = cases(['1\n', '2\n'], ['2\n', '3\n'])
    wrong = execute('print(2)\n', c)
    assert wrong['all_cases_observed'] and wrong['case_passes'] == [True, False]
    assert not wrong['all_finite_published_cases_pass']
    forged = execute('print(\'{"all_finite_published_cases_pass": true}\')\n', c)
    assert not forged['all_finite_published_cases_pass']
    stopped = execute('import os\nos._exit(0)\n', c)
    assert not stopped['all_cases_observed'] and not stopped['all_finite_published_cases_pass']
    scope = execute('def f():\n nonlocal impossible\n', cases([[]], [0], 'f'))
    assert not scope['syntax_valid']


def test_confinement_timeout_and_complete_schema(tmp_path):
    secret = tmp_path/'not-granted.txt'
    secret.write_text('sensitive-test-value')
    for body in [f'print(open({str(secret)!r}).read())\n',
            'import socket\nsocket.socket()\nprint("escaped")\n',
            'import subprocess\nsubprocess.run(["/bin/true"])\nprint("escaped")\n']:
        outcome = execute(body, cases([''], ['escaped']))
        assert outcome['status'] == 'runtime_or_resource_qualification'
        assert not outcome['all_finite_published_cases_pass']
    timeout = execute('while True: pass\n', cases([''], ['']), seconds=.4)
    assert timeout['execution']['status'] == 'wall_deadline_exceeded'
    assert not timeout['all_finite_published_cases_pass']
    for raw in [b'{"inputs":[[]],"outputs":[0,1]}', b'{"inputs":[[{"1":2}]],"outputs":[0],"fn_name":"f"}',
            b'{"inputs":[[]],"outputs":[NaN],"fn_name":"f"}',
            b'{"inputs":[""],"outputs":[""],"inputs":["changed"]}']:
        with pytest.raises(ValueError):
            contract(raw)
