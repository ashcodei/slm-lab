import copy
import hashlib
import math

import pytest
import torch

from slm_lab.dense_human_preference_stage_v1 import DenseHumanPreferenceStage,validate_metric,chosen_gradients
from slm_lab.whole_human_data_v2 import WholeHumanData
from test_own_dense_actor_v1 import actor,SOURCES
from test_own_alignment_checkpoint_v1 import assert_nested_equal
from test_own_aligned_assistant_v1 import original_signature


def pairs():
    d=object.__new__(WholeHumanData);d.family='preferences';d.role='train';d.purpose='optimization';d.rows=[]
    values=[]
    for p in range(5):
        chosen=[1,4+p,8,9,2];rejected=[1,4+p,10,11,12,2]
        values.append(dict(chosen=chosen,rejected=rejected))
        d.rows.append(dict(id=str(p),source_record_sha256=hashlib.sha256(str(p).encode()).hexdigest(),
            split='train',original_split='train',feedback_source='human',response_origin='Synthetic graph fixture only',
            whole_original_response_whitespace_preserved=True,
            native={f:dict(length=len(seq),target_start=2) for f,seq in values[-1].items()}))
    def sequence(p,f):return list(values[p][f]),[-100,-100]+values[p][f][2:]
    def batch(positions,f,device):
        assert len(positions)==1;seq,labels=sequence(positions[0],f)
        return dict(input_ids=torch.tensor([seq],device=device),labels=torch.tensor([labels],device=device),
            attention_mask=torch.ones(1,len(seq),device=device,dtype=torch.bool))
    d.sequence=sequence;d.batch=batch;return d


def options(data):return dict(seed=950131,epochs=2,steps=2*math.ceil(len(data)/2),microbatch=1,accumulation=2,
    learning_rate=.001,betas=[.9,.95],weight_decay=.01,gradient_clip=1.,DPO_beta=.1,DPO_weight=1,chosen_CE_weight=.1)


def stage():
    model=actor();data=pairs();setting=options(data)
    optimizer=torch.optim.AdamW([p for _,p in model.named_selected(('actor',))],lr=setting['learning_rate'],
        betas=tuple(setting['betas']),weight_decay=setting['weight_decay'],foreach=False)
    return DenseHumanPreferenceStage(model,optimizer,data,setting,SOURCES)


def test_full_unrepeated_epochs_tail_and_actual_Adam_recovery_with_all_frozen_metadata_and_pristine_payload():
    learner=stage();fixed=original_signature(learner.model.base)
    learner.update();saved=learner.state();pristine=copy.deepcopy(saved)
    expected=learner.update();weights=learner.model.export(SOURCES);adam=copy.deepcopy(learner.optimizer.state_dict());rng=torch.get_rng_state().clone()
    learner.resume(saved);assert learner.update()==expected
    assert_nested_equal(weights,learner.model.export(SOURCES));assert_nested_equal(adam,learner.optimizer.state_dict());assert_nested_equal(pristine,saved)
    assert torch.equal(rng,torch.get_rng_state()) and fixed==original_signature(learner.model.base)
    for epoch in range(2):
        draws=sum([learner.positions(s) for s in range(epoch*3,(epoch+1)*3)],[])
        assert sorted(draws)==list(range(5)) and len(draws)==5
    assert len(learner.positions(2))==len(learner.positions(5))==1
    assert learner.cursor(6)['actual_pair_draws']==10 and learner.cursor(6)['repeated_rounding_pairs']==0


def test_corrupt_source_masks_DPO_objective_frozen_base_Adam_and_held_roles_reject_before_mutation():
    learner=stage();learner.update();saved=learner.state()
    weights=learner.model.export(SOURCES);adam=copy.deepcopy(learner.optimizer.state_dict());rng=torch.get_rng_state().clone()
    for mutate in [lambda s:s['core']['metrics'][0]['original_human_pairs'][0].update(chosen_response_EOS_targets=999),
                   lambda s:s['core']['metrics'][0].update(summed_objective=999.),
                   lambda s:s.update(frozen_dense_sha256='0'*64),
                   lambda s:s['core']['optimizer']['state'][0]['exp_avg'].fill_(float('nan'))]:
        bad=copy.deepcopy(saved);mutate(bad);before=original_signature(learner.model)
        with pytest.raises(ValueError):learner.resume(bad)
        assert before==original_signature(learner.model);assert_nested_equal(weights,learner.model.export(SOURCES))
        assert_nested_equal(adam,learner.optimizer.state_dict());assert torch.equal(rng,torch.get_rng_state())
    learner.data.role='valid'
    with pytest.raises(ValueError,match='train-only'):DenseHumanPreferenceStage(learner.model,learner.optimizer,learner.data,learner.options,SOURCES)


def test_whole_long_response_margin_reconstructs_FP32_scalar_subtraction_without_double_cancellation_error():
    learner=stage();record=learner.update();row=record['original_human_pairs'][0]
    values=[-32768.00390625,-.001953125,-32768.,-.0009765625]
    for name,value in zip(['policy_chosen_logp','policy_rejected_logp','reference_chosen_logp','reference_rejected_logp'],values):row[name]=value
    pc,pr,qc,qr=[torch.tensor(v,dtype=torch.float32) for v in values]
    assert float((pc-pr)-(qc-qr))!=(values[0]-values[1])-(values[2]-values[3])
    row['implicit_preference_margin']=float((pc-pr)-(qc-qr));row['pair_DPO']=float(torch.nn.functional.softplus(-.1*((pc-pr)-(qc-qr))))
    record['pair_mean_DPO']=sum(r['pair_DPO'] for r in record['original_human_pairs'])/2
    total=sum(r['policy_chosen_logp'] for r in record['original_human_pairs'])
    record['chosen_token_mean_NLL']=-total/record['chosen_response_EOS_targets']
    record['summed_objective']=record['pair_mean_DPO']-.1*total/record['chosen_response_EOS_targets']
    validate_metric(learner.data,record['original_source_positions'],learner.options,record,0)


def test_chosen_CE_control_matches_joint_oracle_without_rejected_or_reference_neural_queries(monkeypatch):
    learner=stage();learner.options['DPO_weight']=0;oracle=copy.deepcopy(learner.model)
    positions=learner.positions(0);old_batch=learner.data.batch
    def batch(ps,field,device):
        assert field=='chosen','CE control must not encode rejected answers'
        return old_batch(ps,field,device)
    monkeypatch.setattr(learner.data,'batch',batch)
    monkeypatch.setattr(learner.model.base,'forward',lambda *a,**k:pytest.fail('CE control must not query the reference'))
    record=chosen_gradients(learner.model,learner.data,positions,learner.options)
    targets=sum(learner.data.rows[p]['native']['chosen']['length']-2 for p in positions);joint=0.
    for p in positions:
        b=old_batch([p],'chosen','cpu');labels=b['labels'][:,1:];mask=labels!=-100
        logits=oracle(b['input_ids'])['logits'][:,:-1].float()[mask]
        joint=joint+.1*torch.nn.functional.cross_entropy(logits,labels[mask],reduction='sum')/targets
    joint.backward()
    assert record['summed_objective']==pytest.approx(float(joint.detach()),rel=2e-6)
    for (_,a),(_,b) in zip(learner.model.named_selected(('actor',)),oracle.named_selected(('actor',))):
        assert torch.allclose(a.grad,b.grad,atol=3e-7,rtol=3e-5)
    learner.optimizer.zero_grad(set_to_none=True);actual=learner.update()
    assert actual['actual_policy_context_tokens']==10 and actual['complete_context_tokens']==22
    assert actual['reference_queried'] is False and actual['actual_policy_rejected_targets']==0
    validate_metric(learner.data,positions,learner.options,actual,0)
