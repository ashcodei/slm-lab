from copy import deepcopy
import hashlib
import json
import math
import random

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from slm_lab.corpus import file_hash
from slm_lab.execution_contrast_data_v1 import ExecutionContrastData
from slm_lab.frozen_own_actor_reference_v1 import FrozenOwnActorReference, signature
from slm_lab.machine_code_preference_stage_v1 import gradients, MachineCodePreferenceStage
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
from test_own_dense_actor_v1 import actor
from test_own_alignment_checkpoint_v1 import assert_nested_equal

SOURCES = dict(own_policy='a'*64, admitted_machine_pairs='b'*64)


def data(count=5):
    # Synthetic graph fixture only; actual-source admission is tested by the
    # full1620pair reader and the separate full-size CPU experiment.
    d=object.__new__(ExecutionContrastData);d.purpose='optimization';d.rows=[]
    for p in range(count):
        chosen=[1,4+p,8,9,2];rejected=[1,4+p,10,11,12,2]
        d.rows.append(dict(id=str(p),source_problem_id='fixture:'+str(p),original_record_sha256=hashlib.sha256(str(p).encode()).hexdigest(),
            split='train',publisher_split='train',human_preferences=False,optimization_admitted=True,
            chosen_ids=chosen,chosen_labels=[-100,-100]+chosen[2:],
            rejected_ids=rejected,rejected_labels=[-100,-100]+rejected[2:],
            counterexample=dict(two_fresh_completed_byte_identical_wrong_outputs=True)))
    return d


def trained():
    m=actor();ids=torch.tensor([[1,5,7,2]]);params=[p for _,p in m.named_selected(('actor',))]
    opt=torch.optim.AdamW(params,lr=.002,foreach=False)
    m(ids,labels=ids)['loss'].backward(inputs=params);opt.step();opt.zero_grad(set_to_none=True)
    return m


def settings(method='machine_DPO'):
    return dict(method=method,seed=991417,global_batch_size=2,learning_rate=.001,
                betas=[.9,.95],weight_decay=.01,gradient_clip=1.,beta=.1,chosen_CE_weight=.2)


def make_stage(method='machine_DPO'):
    m=trained();d=data();o=settings(method);ref=FrozenOwnActorReference(m,sources=dict(own_start='a'*64))
    opt=torch.optim.AdamW([p for _,p in m.named_selected(('actor',))],lr=o['learning_rate'],betas=tuple(o['betas']),
                          weight_decay=o['weight_decay'],foreach=False)
    return MachineCodePreferenceStage(m,opt,d,ref,o,SOURCES)


def logp(model,batch):
    out=model(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'])['logits'][:,:-1].float().log_softmax(-1)
    labels=batch['labels'][:,1:];mask=labels!=-100
    return out[mask].gather(1,labels[mask,None]).sum()


def test_actual_nonzero_own_reference_machine_DPO_matches_independent_joint_objective():
    m=trained();oracle=deepcopy(m);d=data(2)
    ref=FrozenOwnActorReference(m,sources=dict(own_start='a'*64));q=FrozenOwnActorReference(oracle,sources=dict(own_start='a'*64))
    probe=torch.tensor([[1,5,7,2]])
    with torch.no_grad():
        assert torch.equal(ref(probe)['logits'],m(probe)['logits'])
        assert not torch.equal(ref(probe)['logits'],m.base(probe)['logits'])
    frozen=tensor_digest(m.base.named_parameters());private=tensor_digest(ref.parts.named_parameters())
    row=gradients(m,d,[0,1],ref,method='machine_DPO',beta=.1,chosen_CE_weight=.2)
    objective=0.;targets=6
    for p in range(2):
        a,b=[d.batch([p],f,'cpu') for f in ['chosen','rejected']]
        pc,pr=logp(oracle,a),logp(oracle,b)
        with torch.no_grad():qc,qr=logp(q,a),logp(q,b)
        objective=objective-F.logsigmoid(.1*((pc-pr)-(qc-qr)))/2-.2*pc/targets
    objective.backward()
    assert row['human_preferences'] is False and row['own_starting_actor_reference_queried']
    assert row['pair_mean_DPO']==pytest.approx(math.log(2),rel=1e-6)
    assert row['summed_objective']==pytest.approx(float(objective.detach()),rel=2e-6)
    for (_,a),(_,b) in zip(m.named_selected(('actor',)),oracle.named_selected(('actor',))):
        assert a.grad is not None and b.grad is not None and torch.allclose(a.grad,b.grad,atol=3e-7,rtol=3e-5)
    assert all(p.grad is None for p in m.base.parameters()) and m.parts['critic'].weight.grad is None
    assert tensor_digest(m.base.named_parameters())==frozen and tensor_digest(ref.parts.named_parameters())==private


def test_chosen_control_queries_no_rejected_or_reference_and_global_SUM_normalization_matches(monkeypatch):
    m=trained();d=data(2);ref=FrozenOwnActorReference(m,sources=dict(own_start='a'*64));original=d.batch;fields=[]
    def batch(positions,field,device):fields.append(field);return original(positions,field,device)
    d.batch=batch
    def forbidden(*args,**kwargs):raise AssertionError('Chosen CE must not query the reference')
    monkeypatch.setattr(ref,'forward',forbidden)
    row=gradients(m,d,[0,1],ref,method='chosen_CE',chosen_CE_weight=.2)
    assert fields==['chosen','chosen'] and row['own_starting_actor_reference_queried'] is False
    assert all(r['policy_rejected_logp'] is None and r['reference_chosen_logp'] is None for r in row['original_machine_pairs'])
    a=trained();b=deepcopy(a);qa=FrozenOwnActorReference(a,sources=dict(start='a'*64));qb=FrozenOwnActorReference(b,sources=dict(start='a'*64));d=data(2)
    gradients(a,d,[0,1],qa,method='machine_DPO')
    for p in range(2):gradients(b,d,[p],qb,method='machine_DPO',normalization_pairs=2,normalization_chosen_targets=6)
    for (_,x),(_,y) in zip(a.named_selected(('actor',)),b.named_selected(('actor',))):
        assert torch.allclose(x.grad,y.grad,atol=3e-7,rtol=3e-5)


def test_whole_unrepeated_epoch_tail_and_actual_durable_next_update_Adam_and_private_RNG_replay(tmp_path):
    learner=make_stage();learner.update();saved=learner.state();path=tmp_path/'private.pt';torch.save(saved,path);digest=file_hash(path)
    def draws():return [random.random(),float(np.random.random()),float(torch.rand(()))]
    expected_draws=draws();expected=learner.update();weights=learner.model.export(SOURCES);adam=deepcopy(learner.optimizer.state_dict());rng=torch.get_rng_state().clone()
    learner.resume(torch.load(path,weights_only=True,mmap=True));assert draws()==expected_draws
    assert learner.update()==expected;assert_nested_equal(weights,learner.model.export(SOURCES));assert_nested_equal(adam,learner.optimizer.state_dict())
    assert torch.equal(rng,torch.get_rng_state()) and file_hash(path)==digest
    assert sorted(sum([learner.positions(i) for i in range(learner.planned)],[]))==list(range(5))
    assert len(learner.positions(2))==1 and learner.cursor(3)['actual_pair_draws']==5
    assert learner.cursor(3)['repeated_rounding_pairs']==0 and learner.cursor(3)['full_source_epoch']


def test_corrupt_machine_source_reference_Adam_or_private_RNG_rejects_before_mutation():
    learner=make_stage();learner.update();saved=learner.state();weights=learner.model.export(SOURCES)
    adam=deepcopy(learner.optimizer.state_dict());rng=torch.get_rng_state().clone()
    for change in [lambda s:s['core']['metrics'][0]['original_machine_pairs'][0].update(chosen_response_EOS_targets=999),
                   lambda s:s['own_reference_binding'].update(frozen_actor_parts_sha256='0'*64),
                   lambda s:s['core']['optimizer']['state'][0]['exp_avg'].fill_(float('nan')),
                   lambda s:s['core']['rng'].update(torch=torch.zeros(4,dtype=torch.uint8))]:
        bad=deepcopy(saved);change(bad);before=signature(learner.model.parameters())
        with pytest.raises(ValueError):learner.resume(bad)
        assert signature(learner.model.parameters())==before and torch.equal(rng,torch.get_rng_state())
        assert_nested_equal(weights,learner.model.export(SOURCES));assert_nested_equal(adam,learner.optimizer.state_dict())
    m=trained();d=data();ref=FrozenOwnActorReference(m,sources=dict(start='a'*64));d.role='valid'
    with pytest.raises(ValueError,match='train-only'):gradients(m,d,[0],ref,method='machine_DPO')
    d.role='train';d.rows[0]['human_preferences']=True
    with pytest.raises(ValueError,match='machine pair'):gradients(m,d,[0],ref,method='machine_DPO')
    assert all(p.grad is None for p in m.parameters())
