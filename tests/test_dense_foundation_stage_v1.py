import copy
import math

import pytest
import torch
from torch.nn import functional as F

from slm_lab.dense_foundation_data_v1 import DenseFoundationSources,NAMES,REFERENCE_ROUTES
from slm_lab.dense_foundation_stage_v1 import DenseFoundationStage,accumulate_gradients,signature
from test_own_dense_growth_v1 import model as make_model
from test_own_aligned_assistant_v1 import fixture
from test_own_alignment_checkpoint_v1 import assert_nested_equal

SOURCES=dict(synthetic_test_sources='d'*64)


def data():
    # Graph/order fixture only. The real source constructor and its admission
    # guards are exercised separately on all complete publisher populations.
    d=object.__new__(DenseFoundationSources);d.total=7;d.ends=[2,4,6,7]
    d.populations=dict(education=2,variety=2,human=2,program=1)
    values=[([1,4,5,6,2],1),([1,6,7,2],1),([1,7,5,4,6,2],4),([1,6,5,7,2],2),
            ([1,8,9,7,5,6,2],3),([1,9,8,2],2),([1,5,6,7,8,9,10,2],5)]
    def information(p):
        name,local=d.locate(p);seq,start=values[p]
        return dict(global_position=p,family=name,original_train_position=local,reference_route=REFERENCE_ROUTES[name],
                    complete_context_tokens=len(seq),response_EOS_targets=len(seq)-start)
    def batch(p,device='cpu'):
        seq,start=values[p]
        return dict(input_ids=torch.tensor([seq],device=device),labels=torch.tensor([[-100]*start+seq[start:]],device=device),
                    attention_mask=torch.ones(1,len(seq),dtype=torch.bool,device=device))
    d.information=information;d.batch=batch;return d


def options(d,alpha=.1):
    return dict(seed=51313,epochs=2,steps=2*math.ceil(len(d)/3),microbatch=1,accumulation=3,learning_rate=.001,
        betas=[.9,.95],weight_decay=.01,gradient_clip=1.,reference_KL_alpha=alpha,
        coefficients=dict(education=1.,variety=1.,human=1.,program=4.))


def stage(m,r,d,setting=None):
    setting=setting or options(d)
    opt=torch.optim.AdamW([p for _,p in m.language_named_parameters()],lr=setting['learning_rate'],
        betas=tuple(setting['betas']),weight_decay=setting['weight_decay'],foreach=False)
    return DenseFoundationStage(m,opt,r,d,setting,SOURCES)


def test_complete_source_weighted_CE_and_full_vocabulary_KL_gradients_match_independent_joint_oracle():
    m,_=make_model();oracle=copy.deepcopy(m);r=fixture();d=data();setting=options(d);positions=[0,2,4,6]
    m.train();oracle.train();m.network.gradient_checkpointing=oracle.network.gradient_checkpointing=True;r.eval()
    record=accumulate_gradients(m,r,d,positions,setting)
    normalizer=sum(setting['coefficients'][d.information(p)['family']]*d.information(p)['response_EOS_targets'] for p in positions)
    joint=0.
    for p in positions:
        b=d.batch(p);labels=b['labels'][:,1:];mask=labels!=-100;name=d.information(p)['family']
        logits=oracle(b['input_ids'],attention_mask=b['attention_mask'])['logits'][:,:-1].float()[mask]
        with torch.no_grad():ref=r(b['input_ids'],attention_mask=b['attention_mask'],route=REFERENCE_ROUTES[name])['logits'][:,:-1].float()[mask]
        ce=F.cross_entropy(logits,labels[mask],reduction='sum')
        kl=F.kl_div(logits.log_softmax(-1),ref.log_softmax(-1),log_target=True,reduction='sum')
        joint=joint+setting['coefficients'][name]*(ce+.1*kl)/normalizer
    joint.backward(inputs=[p for _,p in oracle.language_named_parameters()])
    assert record['weighted_target_normalizer']==22 and record['summed_objective']==pytest.approx(float(joint.detach()),rel=2e-6)
    for (_,a),(_,b) in zip(m.language_named_parameters(),oracle.language_named_parameters()):
        assert a.grad is not None and bool(torch.isfinite(a.grad).all())
        assert torch.allclose(a.grad,b.grad,atol=3e-7,rtol=3e-5)
    assert all(p.grad is None for p in r.parameters())


def test_actual_dense_mixed_update_Adam_RNG_recovery_preserves_input_payload_and_all_unrepeated_epochs():
    m,_=make_model();r=fixture();d=data();learner=stage(m,r,d);reference=signature(r)
    learner.update();saved=learner.state();pristine=copy.deepcopy(saved)
    expected=learner.update();weights=m.export(SOURCES);adam=copy.deepcopy(learner.optimizer.state_dict());rng=torch.get_rng_state().clone()
    learner.resume(saved);assert learner.update()==expected
    assert_nested_equal(weights,m.export(SOURCES));assert_nested_equal(adam,learner.optimizer.state_dict());assert_nested_equal(pristine,saved)
    assert torch.equal(rng,torch.get_rng_state()) and signature(r)==reference
    for epoch in range(2):
        draws=sum([learner.positions(step) for step in range(epoch*3,(epoch+1)*3)],[])
        assert sorted(draws)==list(range(len(d))) and len(set(draws))==len(d)
    assert len(learner.positions(2))==len(learner.positions(5))==1
    assert learner.cursor(6)['actual_draws']==14 and learner.cursor(6)['repeated_rounding_sources']==0


def test_bad_source_mask_objective_reference_Adam_and_held_roles_reject_before_mutation_and_control_has_no_teacher():
    m,_=make_model();r=fixture();d=data();learner=stage(m,r,d);learner.update();saved=learner.state()
    weights=m.export(SOURCES);adam=copy.deepcopy(learner.optimizer.state_dict());rng=torch.get_rng_state().clone()
    for mutate in [lambda s:s['core']['metrics'][0]['original_source_records'][0].update(response_EOS_targets=999),
                   lambda s:s['core']['metrics'][0].update(weighted_target_normalizer=999.),
                   lambda s:s.update(reference_digest='a'*64),
                   lambda s:s['core']['optimizer']['state'][0]['exp_avg'].fill_(float('nan'))]:
        bad=copy.deepcopy(saved);mutate(bad);before=signature(m)
        with pytest.raises(ValueError):learner.resume(bad)
        assert before==signature(m);assert_nested_equal(weights,m.export(SOURCES));assert_nested_equal(adam,learner.optimizer.state_dict())
        assert torch.equal(rng,torch.get_rng_state())
    control=stage(m,None,d,options(d,0.));assert control.update()['reference_queried'] is False
    d.role='valid'
    with pytest.raises(ValueError,match='train'):stage(m,r,d)
