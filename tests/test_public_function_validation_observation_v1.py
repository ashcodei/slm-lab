from copy import deepcopy
import pytest

from slm_lab.public_function_validation_observation_v1 import observe


def source():
    return dict(id='unit-validation',split='validation',source_candidate=dict(id='unit-validation',role='valid',function='increment',
        original_literal_docstring='Increment x.\n>>> increment(0)\n1\n>>> increment(-2)\n-1\n',whole_original_example_count=2))


def test_actual_validation_public_cases_execute_without_changing_the_sealed_role():
    row=source();result=observe(row,'def increment(x):\n    return x+1')
    assert result['source_role']=='validation' and result['all_original_examples_passed']
    assert result['source_pack_validation_not_certified_all_history_independent']
    assert not result['execution_reward_training_admitted'] and not result['held_benchmark']
    wrong=observe(row,'def increment(x):\n    return x-1')
    assert wrong['syntax_valid'] and not wrong['all_original_examples_passed'] and len(wrong['raw_results'])==2


def test_train_role_aliases_and_missing_public_cases_are_rejected():
    for modify in [lambda r:r.update(split='train'),lambda r:r['source_candidate'].update(role='train'),
                   lambda r:r['source_candidate'].update(whole_original_example_count=1)]:
        row=source();modify(row)
        with pytest.raises(ValueError):observe(row,'def increment(x):\n    return x+1')
