import doctest
import pytest

from slm_lab.code_prompt_controls_v2 import strip_doctests, without_doctests, new_function_without_doctests
from slm_lab.code_prompt_controls_v1 import MARKER, ENDING
from slm_lab.parameter_binding_curriculum_v1 import check


def row():
    return {'id': 'source', 'function': 'factors',
        'prompt': 'Implement this Python function:\n\ndef factors(n):\n\nReturn the positive proper factors of n.\n\ndef factors(n) -> list[int]:\n\n# >>> factors(6)\n# [1, 2, 3]\n\n' + MARKER + 'factors(6,) -> [1, 2, 3]\n' + ENDING,
        'response': '```python\ndef factors(n):\n    return [i for i in range(1, n) if n % i == 0]\n```',
        'verification_cases': [{'args_literal': '(6,)', 'expected_literal': '[1, 2, 3]'}]}


def test_ordinary_and_commented_examples_are_removed_without_losing_prose():
    text = 'Description before.\n\n>>> sum([1, 2])\n3\n\n# More original prose.\n# >>> sum([2, 3])\n# 5\n\nDescription after.\n'
    new = strip_doctests(text)
    assert new == 'Description before.\n\n\n# More original prose.\n\nDescription after.\n'
    assert not doctest.DocTestParser().get_examples(new) and '>>>' not in new


def test_continuation_and_multiline_output_are_removed_together():
    text = 'Before.\n\n>>> [x\n...  for x in range(2)]\n[0,\n 1]\n\nAfter.\n'
    assert strip_doctests(text) == 'Before.\n\n\nAfter.\n'


def test_duplicate_declarations_are_renamed_consistently_and_gold_still_passes():
    original = row(); stripped = without_doctests(original); changed = new_function_without_doctests(original)
    assert 'Return the positive proper factors of n.' in changed['prompt']
    assert '>>>' not in changed['prompt'] and MARKER not in changed['prompt']
    assert changed['prompt'].count('def ' + changed['function'] + '(') == 2
    assert 'def factors(' not in changed['prompt']
    assert stripped['response'] == original['response']
    assert check(changed)['functional_passed']


def test_unparseable_python_example_is_not_silently_accepted():
    with pytest.raises(ValueError): strip_doctests('>>>x\n3\n')
