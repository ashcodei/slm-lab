import copy
import pytest

from slm_lab.code_training_views_v1 import sha
from slm_lab.documented_doctest_v1 import execute
from slm_lab.public_example_code_reward_v1 import evaluate_candidate as legacy_reward
from slm_lab.causal_code_reward_v2 import ablate,evaluate


def candidate(text,program='def add_one(x):\n    return x + 1'):
    formatted='```python\n'+program+'\n```'
    return dict(source=dict(id='fixture',original_partition='train',language='Python',prompt=text,response=formatted),
        program=program,program_sha256=sha(program),doctest_text=text,doctest_text_sha256=sha(text),function=program.split('def ')[1].split('(')[0],
        existing_formatted_source_target_sha256=sha(formatted),whole_original_example_count=text.count('>>>'))


def test_assignment_only_legacy_loophole_is_reproduced_and_post_call_checks_now_score():
    c=candidate('>>> result = add_one(2)\n>>> result\n3\n>>> result = add_one(7)\n>>> result\n8\n')
    reference=execute(c);before=copy.deepcopy(c);noop='def add_one(x):\n    pass'
    assert legacy_reward(c,reference,noop)['reward']==1. # The previously missed post-call inspection.
    control=ablate(c,reference)
    assert control['dependent_example_indices']==[1,3] and control['post_call_dependent_check_indices']==[1,3]
    assert evaluate(c,reference,control,noop,origin='test_fixture')['reward']==0.
    assert evaluate(c,reference,control,c['program'],origin='test_fixture')['reward']==1.
    partial=evaluate(c,reference,control,'def add_one(x):\n    return 3',origin='test_fixture')
    assert partial['reward']==pytest.approx(.45) and not partial['full_original_stream_passed']
    assert c==before


def test_side_effects_are_checked_without_crediting_unrelated_setup():
    c=candidate('>>> values = []\n>>> append_one(values)\n>>> values\n[1]\n>>> 2 + 2\n4\n',
        'def append_one(values):\n    values.append(1)')
    reference=execute(c);control=ablate(c,reference)
    assert control['dependent_example_indices']==[2]
    assert evaluate(c,reference,control,'def append_one(values):\n    pass',origin='test_fixture')['reward']==0
    assert evaluate(c,reference,control,c['program'],origin='test_fixture')['reward']==1


def test_uninformative_sources_are_not_called_correct_and_held_sources_are_rejected():
    c=candidate('>>> result = add_one(2)\n>>> 3 + 3\n6\n');reference=execute(c);control=ablate(c,reference)
    assert not control['supported']
    result=evaluate(c,reference,control,c['program'],origin='test_fixture')
    assert result['reward']==0 and result['not_a_candidate_correctness_verdict']
    c['source']['original_partition']='test'
    with pytest.raises(ValueError,match='train source'):evaluate(c,reference,control,c['program'],origin='test_fixture')
