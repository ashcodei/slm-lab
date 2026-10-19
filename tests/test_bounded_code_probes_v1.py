import copy

import pytest

from slm_lab.bounded_code_probes_v1 import proposals,qualify,evaluate,execute_suite
from slm_lab.causal_code_reward_v2 import ablate,evaluate as old_reward
from slm_lab.documented_doctest_v1 import execute,sha


def candidate(program,text,name):
    formatted='```python\n'+program+'\n```'
    return dict(source=dict(id='explicit-test-fixture',original_partition='train',language='Python',prompt=text,response=formatted),
        program=program,program_sha256=sha(program),doctest_text=text,doctest_text_sha256=sha(text),function=name,
        existing_formatted_source_target_sha256=sha(formatted),whole_original_example_count=text.count('>>>'))


def test_original_example_pass_can_fail_new_closed_string_and_mutable_type_probes():
    c=candidate('def is_seq(obj):\n    try:\n        iter(obj)\n        len(obj)\n        return not isinstance(obj, (str, bytes))\n    except (TypeError, AttributeError):\n        return False',
        '>>> values = [1, 2, 3]\n>>> is_seq(values)\nTrue\n>>> is_seq(iter(values))\nFalse\n','is_seq')
    before=copy.deepcopy(c);reference=execute(c);causal=ablate(c,reference)
    bad="def is_seq(obj):\n    return hasattr(obj, '__iter__') and hasattr(obj, '__getitem__')"
    assert old_reward(c,reference,causal,bad,origin='test_fixture')['reward']==1
    q=qualify(c);assert q['available'] and any("'abc'" in p['call_expression'] for p in q['accepted'])
    assert all(p['call_expression']!='is_seq(values)' for p in q['accepted'])
    actual=evaluate(c,reference,causal,q,bad,origin='test_fixture')
    assert actual['all_original_published_examples_passed'] and not actual['all_augmented_reference_observations_passed'] and actual['reward']<1
    assert evaluate(c,reference,causal,q,c['program'],origin='test_fixture')['reward']==1
    assert evaluate(c,reference,causal,q,'def is_seq(obj):\n    pass',origin='test_fixture')['reward']==0
    changed=copy.deepcopy(q);changed['raw_reference_discovery']['actual_program_sha256']='0'*64
    with pytest.raises(ValueError,match='provenance'):evaluate(c,reference,causal,changed,bad,origin='test_fixture')
    assert c==before


def test_post_call_argument_state_observes_mutations_even_when_return_is_none():
    c=candidate('def append_one(xs):\n    xs.append(1)','>>> xs = []\n>>> append_one(xs)\n>>> xs\n[1]\n','append_one')
    reference=execute(c);control=ablate(c,reference);q=qualify(c)
    assert q['available'] and all('None' in p['expected_observation_stdout'] for p in q['accepted'])
    assert evaluate(c,reference,control,q,c['program'],origin='test_fixture')['reward']==1
    assert evaluate(c,reference,control,q,'def append_one(xs):\n    pass',origin='test_fixture')['reward']==0
    assert any(d['status']=='reference_did_not_accept_this_input' for d in q['dispositions'])


def test_held_roles_and_changed_observation_commands_are_rejected():
    c=candidate('def plus_one(x):\n    return x + 1','>>> plus_one(1)\n2\n>>> plus_one(2)\n3\n','plus_one')
    p=proposals(c)[0];p['observation_expression']='True\n'
    with pytest.raises(ValueError,match='reconstruct'):execute_suite(c,[p])
    c['source']['original_partition']='test'
    with pytest.raises(ValueError,match='publisher-train'):proposals(c)


def test_no_argument_sites_stays_explicitly_unavailable():
    c=candidate('def value():\n    return 1','>>> value()\n1\n>>> value()\n1\n','value')
    q=qualify(c);assert not q['available'] and q['accepted']==[]


def test_docstring_only_source_keeps_explicit_ablation_qualification():
    c=candidate('def noop(x):\n    """No behavior."""','>>> noop(1)\n>>> noop(2)\n','noop')
    q=qualify(c);assert not q['available'] and q['status']=='source_body_ablation_unavailable'
