import copy
import hashlib
import json

import pytest

from slm_lab.apps_published_cases_v2 import contract, execute
from slm_lab.apps_execution_reward_v1 import control, flags, noop, reward


def fixture(mode='function'):
    original = ('def f(x):\n    return None if x == 0 else 2*x\n' if mode == 'function'
                else 'print(int(input()) * 2)\n')
    cases = contract(json.dumps(dict(inputs=[[0], [2], [3]] if mode == 'function' else ['2\n', '3\n'],
        outputs=[None, 4, 6] if mode == 'function' else ['4\n', '6\n'],
        fn_name='f' if mode == 'function' else None)).encode())
    source = dict(id='train:fixture', split='train', publisher_split='train',
        original_source_body_sha256=hashlib.sha256(original.encode()).hexdigest(),
        original_tests_sha256=cases['original_tests_sha256'])
    ref = execute(original, cases); empty = execute(noop(cases), cases)
    return source, cases, original, ref, empty


def test_actual_noop_discrimination_partial_ceiling_and_real_delivery_EOS():
    source, cases, original, ref, empty = fixture()
    admitted = control(source, cases, original, ref, empty)
    assert admitted['informative_case_indices'] == [1, 2]
    assert reward(source, cases, admitted, noop(cases), stopped_on_eos=True)['reward'] == 0
    assert reward(source, cases, admitted, original, stopped_on_eos=True)['reward'] == 1
    assert reward(source, cases, admitted, original, stopped_on_eos=False)['reward'] == .9
    wrong = 'def f(x):\n    return 0 if x == 0 else 2*x\n'
    outcome = reward(source, cases, admitted, wrong, stopped_on_eos=True)
    assert outcome['reward'] == .9 and not outcome['all_finite_original_cases_passed']
    partial = 'def f(x):\n    return 4\n'
    assert reward(source, cases, admitted, partial, stopped_on_eos=True)['reward'] == .45


def test_real_stdin_faults_forged_values_and_held_source_reject():
    source, cases, original, ref, empty = fixture('stdin')
    admitted = control(source, cases, original, ref, empty)
    assert reward(source, cases, admitted, original, stopped_on_eos=True)['reward'] == 1
    for code in ['print(4)', 'raise RuntimeError("failed")', 'def broken(:',
                 'print(\'{"reward": 1}\')', '']:
        assert reward(source, cases, admitted, code, stopped_on_eos=True)['reward'] < 1
    changed = copy.deepcopy(ref); changed['raw_results']['results'][0]['value'] = 'wrong'
    with pytest.raises(ValueError, match='raw values'):
        flags(changed, cases)
    changed = dict(source, split='test')
    with pytest.raises(ValueError, match='publisher-train'):
        reward(changed, None, None, None, stopped_on_eos=True)
    with pytest.raises(ValueError, match='publisher-train'):
        control(changed, None, None, None, None)
