import copy
import pytest

from slm_lab.documented_doctest_v1 import execute
from slm_lab.public_example_code_reward_v1 import evaluate_candidate
from test_documented_doctest_v1 import candidate


def source():
    text=">>> items = [1, 2]\n>>> transform(items)\n[2, 4]\n>>> items.append(3)\n>>> transform(items)\n[2, 4, 6]\n"
    c=candidate('def transform(xs):\n    return [2*x for x in xs]',text)
    return c,execute(c)


def test_setup_success_earns_zero_and_actual_call_examples_control_partial_and_complete_rewards():
    c,reference=source();before=copy.deepcopy(c)
    failed=evaluate_candidate(c,reference,'def transform(xs):\n    return []')
    partial=evaluate_candidate(c,reference,'def transform(xs):\n    return [2, 4]')
    good=evaluate_candidate(c,reference,'def transform(xs):\n    return [x*2 for x in xs]')
    assert failed['reward']==0 and partial['reward']==.5 and good['reward']==1
    assert good['scored_original_example_indices']==[1,3] and good['all_complete_original_examples_passed']
    assert good['original_reference_implementation_not_sent_to_worker'] and good['expected_outputs_stay_outside_worker']
    assert c==before and not good['human_preference_label']


def test_wrong_entry_point_missing_context_and_held_roles_cannot_supply_positive_execution_rewards():
    c,reference=source()
    assert evaluate_candidate(c,reference,'def wrong(xs):\n    return [x*2 for x in xs]')['reward']==0
    assert evaluate_candidate(c,reference,'def transform(xs):\n    return missing(xs)')['reward']==0
    c['source']['original_partition']='test'
    with pytest.raises(ValueError,match='Held'):evaluate_candidate(c,reference,'def transform(xs):\n    return xs')
