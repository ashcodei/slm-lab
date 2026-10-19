import copy
import importlib.util
import json
import sys
from pathlib import Path
import pytest


def driver():
    p=Path(__file__).resolve().parents[1]/'scripts/train_dense_PPO_pilot_v1.py';sys.path.insert(0,str(p.parent))
    spec=importlib.util.spec_from_file_location('dense_PPO_driver_test',p);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_exact_preference_owner_or_registered_worker_blocks_stale_completion_and_failed_predecessor_rejects(monkeypatch,tmp_path):
    d=driver();monkeypatch.setattr(d,'PREVIOUS',tmp_path);owner=dict(pid=1,ticks='10');child=dict(pid=2,ticks='20')
    (tmp_path/'children.json').write_text(json.dumps([child]))
    for name in ('complete.json','restored.json','independent-comparison.json'):(tmp_path/name).write_text('{}')
    for live in (owner,child):
        monkeypatch.setattr(d.owner.resources,'same',lambda a:a if a==live else None);assert not d.previous_ended(owner)
    monkeypatch.setattr(d.owner.resources,'same',lambda a:None);assert d.previous_ended(owner)
    (tmp_path/'failure.json').write_text('{}')
    with pytest.raises(ValueError,match='restoration'):d.previous_ended(owner)


def test_active_original_work_rejects_before_gpu_query_or_any_signal(monkeypatch):
    d=driver()
    def active():raise ValueError('Original active work')
    monkeypatch.setattr(d.owner,'idle_original',active)
    monkeypatch.setattr(d.subprocess,'check_output',lambda *a,**k:pytest.fail('GPU query before original-work admission'))
    monkeypatch.setattr(d.os,'kill',lambda *a,**k:pytest.fail('Signal to original active work'))
    with pytest.raises(ValueError,match='active'):d.require_idle_devices()


def test_all_four_actual_profile_counts_full_horizon_and_maximum_source_required():
    d=driver();plan=dict(arms={str(i):{} for i in range(4)},actual_counts={'actual_unique':999706442},reward_counts={'actual_training_only_total':706210944},capacity_maximum_prompt_position=7)
    proofs=[dict(arm=str(i),physical_GPU=i,actual_counts=plan['actual_counts'],separate_training_only_reward=plan['reward_counts'],
        actual_actor_critic_Adam_states=562,actual_actor_critic_values=15306752,actual_updates=2,exact_next_update_and_private_CUDA_RNG_rejection=True,
        full_maximum_prompt_and_fixed_response_stress=dict(original_source=dict(original_train_position=7),full_fixed_action_tokens=128,
            actual_prefills=1,actual_decodes=127,actor_critic_finite_gradients=True,no_optimizer_step=True)) for i in range(4)]
    d.validate_capacity(proofs,plan)
    for mutate in [lambda p:p.pop(),lambda p:p[2].update(physical_GPU=0),lambda p:p[0].update(actual_actor_critic_Adam_states=561),
        lambda p:p[1]['full_maximum_prompt_and_fixed_response_stress'].update(full_fixed_action_tokens=75),
        lambda p:p[3]['full_maximum_prompt_and_fixed_response_stress']['original_source'].update(original_train_position=8)]:
        bad=copy.deepcopy(proofs);mutate(bad)
        with pytest.raises(ValueError):d.validate_capacity(bad,plan)
