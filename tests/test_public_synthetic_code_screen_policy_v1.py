import json

import pytest

from slm_lab.bounded_disk_reserve_wait_v1 import wait_for_reserve
from slm_lab.public_synthetic_code_screen_policy_v1 import published_all_pass,body_fingerprint,supervision_scope


def row():
    return dict(unit_tests=json.dumps(['assert f(1)==1','assert f(2)==2']),
        tests_execution_status=json.dumps(['pass','pass']),average_test_score=1.)


def test_whole_provider_status_is_only_a_precondition_not_correctness_or_admission():
    assert published_all_pass(row())
    for change in [dict(tests_execution_status='["pass"]'),dict(tests_execution_status='["pass","fail"]'),
            dict(average_test_score=.9),dict(average_test_score=True),dict(average_test_score=float('nan')),
            dict(unit_tests='[]'),dict(unit_tests='null'),dict(tests_execution_status='oops')]:
        assert not published_all_pass(dict(row(),**change))
    scope=supervision_scope()
    assert scope['use_as_RL_reward_oracle_forbidden'] and scope['weak_supervised_learning_candidate_only']
    assert not scope['human_preferences'] and not scope['training_admitted'] and not scope['independent_semantic_correctness_certified']


def test_known_bad_AST_veto_survives_whitespace_comments_and_docstrings_only():
    source='def f(x):\n    """A docstring."""\n    return x + 1\n'
    equivalent='# a comment\ndef f(x):\n  "Other docs"\n  return x+1 # end\n'
    assert body_fingerprint(source)==body_fingerprint(equivalent)
    assert body_fingerprint(source)!=body_fingerprint('def f(x): return x - 1')
    assert source.endswith('    return x + 1\n')


def test_disk_wait_recovers_after_transient_atomic_writer_and_rechecks_sources_each_time():
    clock=[0.];values=iter([90,80,120]);checks=[];sleeps=[]
    def pause(seconds):sleeps.append(seconds);clock[0]+=seconds
    result=wait_for_reserve('.',minimum_bytes=100,maximum_seconds=30,interval_seconds=5,
        guard=lambda:checks.append(clock[0]),free_bytes=lambda:next(values),now=lambda:clock[0],pause=pause)
    assert result==dict(waited_seconds=10.,failed_space_observations=2,final_free_bytes=120)
    assert sleeps==[5,5] and checks==[0.,5.,10.]


def test_disk_wait_is_bounded_and_cannot_ignore_changed_sources():
    clock=[0.]
    with pytest.raises(ValueError,match='bounded wait'):
        wait_for_reserve('.',minimum_bytes=100,maximum_seconds=7,interval_seconds=5,
            free_bytes=lambda:99,now=lambda:clock[0],pause=lambda s:clock.__setitem__(0,clock[0]+s))
    assert clock[0]==7
    def changed():raise ValueError('pinned source changed')
    with pytest.raises(ValueError,match='pinned source'):
        wait_for_reserve('.',minimum_bytes=100,guard=changed,free_bytes=lambda:1000)


def test_invalid_wait_controls_reject_without_sleeping_or_reading_files():
    for options in [dict(minimum_bytes=0),dict(maximum_seconds=float('nan')),dict(maximum_seconds=1801),dict(interval_seconds=31)]:
        with pytest.raises(ValueError):wait_for_reserve('.',**dict(dict(minimum_bytes=100),**options))
