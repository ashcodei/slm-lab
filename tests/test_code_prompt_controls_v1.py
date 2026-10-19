from copy import deepcopy
import pytest

from slm_lab.code_prompt_controls_v1 import MARKER, ENDING, without_examples, new_function_name
from slm_lab.parameter_binding_curriculum_v1 import check


def source():
    return {'id': 'known-public-program', 'function': 'fact',
        'prompt': 'Implement this Python function:\n\ndef fact(n):\n\nReturn factorial of n.' + MARKER + 'fact(6,) -> 720\n' + ENDING,
        'response': '```python\ndef fact(n):\n    if n == 0:\n        return 1\n    return n * fact(n - 1)\n```',
        'verification_cases': [{'args_literal': '(6,)', 'expected_literal': '720'}]}


def test_example_control_keeps_description_signature_code_and_checks_exact():
    row = source(); before = deepcopy(row); changed = without_examples(row)
    assert row == before
    assert 'Return factorial of n.' in changed['prompt'] and 'def fact(n):' in changed['prompt']
    assert '720' not in changed['prompt'] and MARKER not in changed['prompt']
    assert changed['response'] == row['response'] and changed['verification_cases'] == row['verification_cases']


def test_new_identifier_changes_recursion_and_preserves_published_behavior():
    row = new_function_name(source())
    assert row['function'] != 'fact' and 'def ' + row['function'] + '(n):' in row['prompt']
    assert 'return n * ' + row['function'] + '(n - 1)' in row['response']
    assert check(row)['functional_passed']


@pytest.mark.parametrize('change', ['missing_block', 'duplicate_block', 'missing_tail'])
def test_ambiguous_or_nonstandard_prompt_is_not_silently_rewritten(change):
    row = source()
    if change == 'missing_block': row['prompt'] = row['prompt'].replace(MARKER, '\n\n')
    elif change == 'duplicate_block': row['prompt'] += MARKER + ENDING
    else: row['prompt'] = row['prompt'].replace(ENDING, '')
    with pytest.raises(ValueError): without_examples(row)
