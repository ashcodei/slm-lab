import json

import pytest

from slm_lab.public_reasoned_code_cases_v1 import contract, execute, qualify_reference, score_candidate


def test_reference_fixes_singleton_convention_before_candidate():
    cases = contract(json.dumps(dict(inputs=[[1], [8]], outputs=[[2], [9]], fn_name='inc')).encode())
    reference = execute('def inc(x):\n    return x+1\n', cases, seconds=5)
    qualified = qualify_reference(cases, reference)
    assert qualified['reference_qualified']
    assert qualified['comparison'] == 'function_original_singleton_wrapper'
    # The legacy runtime accepts both shapes; the new parent comparator must
    # reject the wrong shape under the convention fixed by the reference.
    wrong_shape = execute('def inc(x):\n    return [x+1]\n', cases, seconds=5)
    assert wrong_shape['all_finite_published_cases_pass']
    scored = score_candidate(cases, wrong_shape, qualified)
    assert scored['all_cases_observed'] and not scored['all_finite_cases_pass']
    assert scored['passed_cases'] == 0
    correct = execute('class Solution:\n    def inc(self,x):\n        return x+1\n', cases, seconds=5)
    assert score_candidate(cases, correct, qualified)['all_finite_cases_pass']
    assert not scored['SFT_admitted'] and not scored['RL_reward_admitted']


def test_complete_stdin_uses_declared_ordered_tokens_and_rejects_wrong_order():
    cases = contract(json.dumps(dict(inputs=['1\n', '9\n'], outputs=['1  2\n', '9  10\n'])).encode())
    reference = execute('n=int(input())\nprint(n,n+1)\n', cases, seconds=5)
    assert not qualify_reference(cases, reference)['reference_qualified']
    qualified = qualify_reference(cases, reference, stdin_comparison='stdin_exact_ordered_tokens')
    assert qualified['reference_qualified']
    good = score_candidate(cases, reference, qualified)
    assert good['all_finite_cases_pass'] and good['passed_cases'] == 2
    wrong = execute('n=int(input())\nprint(n+1,n)\n', cases, seconds=5)
    scored = score_candidate(cases, wrong, qualified)
    assert scored['all_cases_observed'] and not scored['all_finite_cases_pass']
    changed = contract(json.dumps(dict(inputs=['1\n'], outputs=['1 2\n'])).encode())
    with pytest.raises(ValueError, match='Same complete original cases'):
        score_candidate(changed, wrong, qualified)


def test_runtime_failure_never_becomes_complete_reference_or_positive():
    cases = contract(json.dumps(dict(inputs=[[1], [2]], outputs=[2, 3], fn_name='inc')).encode())
    complete = execute('def inc(x):\n    return x+1\n', cases, seconds=5)
    reference = qualify_reference(cases, complete)
    assert reference['reference_qualified']
    failed = execute('def inc(x):\n    raise RuntimeError("no value")\n', cases, seconds=5)
    assert not qualify_reference(cases, failed)['reference_qualified']
    result = score_candidate(cases, failed, reference)
    assert not result['all_cases_observed'] and not result['all_finite_cases_pass']
