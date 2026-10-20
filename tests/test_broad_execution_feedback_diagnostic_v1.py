import copy
import importlib.util
import math
from pathlib import Path

import pytest

PATH=Path(__file__).resolve().parents[1]/'scripts/analyze_broad_execution_feedback_CPU_v1.py'
spec=importlib.util.spec_from_file_location('feedback_diagnostic',PATH);reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)


def row(step,scores):
    mean=sum(scores)/2;std=math.sqrt(sum((v-mean)**2 for v in scores)/2)
    return dict(step=step,method='anchor_CE',world=2,genuine_human_RLHF=False,position=step-1,source_id='apps:fixture:'+str(step),
        global_rewards=scores,global_advantages=[(v-mean)/(std+1e-6) for v in scores],global_sampled_action_tokens=4,action_horizon=8,
        trajectories=[dict(response_ids=[4,2],actual_cached_behavior_logps=[-.5,-.2],actual_execution_reward=dict(
            reward=score,complete_delivery_stopped_on_EOS=True,status='controlled_recorded_feedback_only')) for score in scores])


def test_zero_and_varying_reward_groups_reconstruct_exact_stops_counts_and_advantages():
    result=reader.summarize([row(1,[0.,0.]),row(2,[0.,.5])],expected_groups=2,expected_world=2)
    total=result['totals']
    assert total['groups']==2 and total['all_zero_reward_groups']==1 and total['groups_with_reward_variation']==1
    assert total['trajectories']==4 and total['positive_reward_trajectories']==1 and total['full_reward_trajectories']==0
    assert total['EOS_stops']==4 and result['sampled_EOS_fraction']==1 and result['reward_varying_group_fraction']==.5
    assert result['CE_control_rewards_observed_not_execution_loss_optimized'] and result['recorded_runtime_qualifications_not_certified_algorithmic_failures']


def test_changed_reward_advantage_stop_horizon_count_and_source_order_reject():
    original=[row(1,[0.,0.]),row(2,[0.,.5])]
    for kind in ['reward','advantage','EOS','horizon','step','actions','method']:
        rows=copy.deepcopy(original)
        if kind=='reward':rows[1]['trajectories'][1]['actual_execution_reward']['reward']=.25
        if kind=='advantage':rows[1]['global_advantages'][1]+=.1
        if kind=='EOS':rows[0]['trajectories'][0]['actual_execution_reward']['complete_delivery_stopped_on_EOS']=False
        if kind=='horizon':rows[0]['action_horizon']=1
        if kind=='step':rows.reverse()
        if kind=='actions':rows[0]['global_sampled_action_tokens']=3
        if kind=='method':rows[0]['method']='anchor_execution'
        with pytest.raises(ValueError):reader.summarize(rows,expected_groups=2,expected_world=2)
    with pytest.raises(ValueError):reader.summarize(original[:1],expected_groups=2,expected_world=2)
