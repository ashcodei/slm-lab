import copy
import hashlib
import math

import pytest
import torch
from torch.nn import functional as F

from slm_lab.dense_human_preference_v1 import preference_gradients,sequence_logp
from slm_lab.whole_human_data_v2 import WholeHumanData
from test_own_dense_actor_v1 import actor


def pairs():
    data=object.__new__(WholeHumanData);data.family='preferences';data.role='train';data.purpose='optimization'
    sequences=[dict(chosen=([1,4,5,7,2],2),rejected=([1,4,8,2],2)),
               dict(chosen=([1,6,8,2],2),rejected=([1,6,9,5,7,2],2))]
    data.rows=[]
    for i,pair in enumerate(sequences):
        data.rows.append(dict(id=str(i),source_record_sha256=hashlib.sha256(str(pair).encode()).hexdigest(),
            split='train',original_split='train',feedback_source='human',whole_original_response_whitespace_preserved=True,
            response_origin='Other-model responses human-ranked: synthetic test fixture only',
            native={f:dict(length=len(ids),target_start=start) for f,(ids,start) in pair.items()}))
    def sequence(position,field):
        ids,start=sequences[position][field];return ids,[-100]*start+ids[start:]
    def batch(positions,field,device):
        assert len(positions)==1;ids,labels=sequence(positions[0],field)
        return dict(input_ids=torch.tensor([ids],device=device),labels=torch.tensor([labels],device=device),
            attention_mask=torch.ones(1,len(ids),device=device,dtype=torch.bool))
    data.sequence=sequence;data.batch=batch;return data


def test_complete_human_DPO_plus_chosen_CE_gradients_match_independent_joint_objective():
    model=actor();oracle=copy.deepcopy(model);data=pairs()
    record=preference_gradients(model,data,[0,1],beta=.1,chosen_CE_weight=.2)
    total=0.;chosen_targets=5
    for p in range(2):
        values={};reference={}
        for field in ('chosen','rejected'):
            b=data.batch([p],field,'cpu');labels=b['labels'][:,1:];mask=labels!=-100
            out=oracle(b['input_ids'])['logits'][:,:-1].float().log_softmax(-1)
            values[field]=out[mask].gather(1,labels[mask,None]).sum()
            with torch.no_grad():ref=oracle.base(b['input_ids'])['logits'][:,:-1].float().log_softmax(-1)
            reference[field]=ref[mask].gather(1,labels[mask,None]).sum()
        total=total-F.logsigmoid(.1*((values['chosen']-values['rejected'])-(reference['chosen']-reference['rejected'])))/2-.2*values['chosen']/chosen_targets
    total.backward()
    assert record['pair_mean_DPO']==pytest.approx(math.log(2),rel=1e-6)
    assert record['chosen_response_EOS_targets']==5 and record['rejected_response_EOS_targets']==6
    assert record['summed_objective']==pytest.approx(float(total.detach()),rel=2e-6)
    for (_,a),(_,b) in zip(model.named_selected(('actor',)),oracle.named_selected(('actor',))):
        assert a.grad is not None and b.grad is not None and torch.allclose(a.grad,b.grad,atol=3e-7,rtol=3e-5)
    assert all(p.grad is None for p in model.base.parameters()) and model.parts['critic'].weight.grad is None


def test_held_machine_labels_different_prompts_or_missing_EOS_reject_before_any_gradient():
    model=actor();data=pairs();data.role='valid'
    with pytest.raises(ValueError,match='train-only'):preference_gradients(model,data,[0])
    data.role='train';data.rows[0]['feedback_source']='machine'
    with pytest.raises(ValueError,match='human-ranked'):preference_gradients(model,data,[0])
    data.rows[0]['feedback_source']='human';original=data.sequence
    def wrong(p,f):
        ids,labels=original(p,f);ids=list(ids)
        if f=='rejected':ids[1]=12
        return ids,labels
    data.sequence=wrong
    with pytest.raises(ValueError,match='identical whole prompt'):preference_gradients(model,data,[0])
    assert all(p.grad is None for p in model.parameters())
