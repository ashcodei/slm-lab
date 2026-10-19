import copy
import math

import pytest
import torch
from torch.nn import functional as F

from slm_lab.frozen_own_actor_reference_v1 import FrozenOwnActorReference
from slm_lab.own_function_trajectory_preferences_v1 import OwnFunctionTrajectoryPreferences,native_pair,control_group
from slm_lab.own_function_trajectory_preference_stage_v1 import gradients
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
from test_own_dense_actor_v1 import actor


def fixture():
    torch.set_num_threads(1);model=actor();data=OwnFunctionTrajectoryPreferences.__new__(OwnFunctionTrajectoryPreferences)
    data.role='train';data.purpose='optimization';data.rows=[]
    # Synthetic tiny neural/mask fixture, not actual14-source data admission.
    for i in range(3):
        group=next('unit-group-'+str(k) for k in range(i*100,i*100+100) if control_group('unit-group-'+str(k))==(i==2))
        row=dict(id='unit-'+str(i),split='train',group_id=group,source_candidate=dict(id='unit-'+str(i),role='train'))
        data.rows.append(native_pair(row,[1,5,6],[7+i,8,2],[9+i,10,11] if i==0 else [9+i,2],stopped=i!=0,budget=3))
    data.optimization_positions=[0,1];data.unoptimized_control_positions=[2]
    ref=FrozenOwnActorReference(model,sources=dict(unit_own_start_sha256='a'*64))
    return model,data,ref


def direct_logp(model,batch):
    logits=model(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'])['logits']
    labels=batch['labels'][:,1:];mask=labels!=-100
    return -F.cross_entropy(logits[:,:-1].float()[mask],labels[mask],reduction='sum')


def test_actual_gradients_match_direct_complete_positive_and_censored_negative_log_ratio_objective():
    model,data,ref=fixture();other=copy.deepcopy(model);other_ref=FrozenOwnActorReference(other,sources=dict(unit_own_start_sha256='a'*64))
    fixed=tensor_digest(model.base.named_parameters());critic=model.parts['critic'].weight.detach().clone()
    result=gradients(model,data,[0,1],ref,method='trajectory_DPO',beta=.2,chosen_CE_weight=1.)
    target_count=6;objective=0
    for p in [0,1]:
        cb=data.batch(p,'chosen');rb=data.batch(p,'rejected')
        chosen,rejected=direct_logp(other,cb),direct_logp(other,rb)
        with torch.no_grad():qc,qr=direct_logp(other_ref,cb),direct_logp(other_ref,rb)
        objective=objective-chosen/target_count+F.softplus(-.2*((chosen-rejected)-(qc-qr)))/2
    parameters=[p for _,p in other.named_selected(('actor',))];objective.backward(inputs=parameters)
    assert result['chosen_response_EOS_targets']==6 and result['rejected_actual_action_targets']==5
    assert math.isclose(result['summed_objective'],float(objective.detach()),abs_tol=2e-6)
    for (_,a),(_,b) in zip(model.named_selected(('actor',)),other.named_selected(('actor',))):
        assert a.grad is not None and torch.allclose(a.grad,b.grad,rtol=2e-5,atol=2e-6)
    assert tensor_digest(model.base.named_parameters())==fixed and torch.equal(model.parts['critic'].weight,critic)
    assert all(p.grad is None for p in model.base.parameters())


def test_matched_CE_control_never_queries_negative_or_reference_and_controls_cannot_supply_gradients(monkeypatch):
    model,data,ref=fixture()
    def forbidden(*args,**kwargs):raise AssertionError('Reference forward must not be queried by chosen CE control')
    monkeypatch.setattr(ref,'forward',forbidden)
    original=data.batch
    def batch(position,field,*args,**kwargs):
        assert field=='chosen';return original(position,field,*args,**kwargs)
    monkeypatch.setattr(data,'batch',batch)
    result=gradients(model,data,[0,1],ref,method='chosen_CE')
    assert result['pair_mean_preference']==0 and not result['own_starting_actor_reference_queried']
    assert all(r['policy_rejected_logp'] is None for r in result['original_trajectory_pairs'])
    with pytest.raises(ValueError,match='optimization-source-train'):gradients(model,data,[2],ref,method='chosen_CE')
    data.purpose='audit'
    with pytest.raises(ValueError):gradients(model,data,[0],ref,method='chosen_CE')
