"""Compare the streamed full-code gradient with an independently built objective."""
from contextlib import nullcontext
from copy import deepcopy

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from slm_lab.code_prompt_matching_v1 import (pair_map,validate_pair_map,mismatch_sequence,
    matching_gradients)


class Causal(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding=torch.nn.Embedding(32,4,dtype=torch.float64)
        self.head=torch.nn.Linear(4,32,bias=False,dtype=torch.float64)

    def forward(self,input_ids,labels):
        hidden=self.embedding(input_ids).cumsum(1)/torch.arange(1,input_ids.shape[1]+1,dtype=torch.float64)[None,:,None]
        logits=self.head(hidden)
        return {'loss':F.cross_entropy(logits[:,:-1].reshape(-1,32),labels[:,1:].reshape(-1),ignore_index=-100)}


def examples():
    return [dict(role='train',positive=([1,4,10,11,2],[-100,-100,10,11,2]),
        negative=([1,5,6,7,10,11,2],[-100]*4+[10,11,2])),
        dict(role='train',positive=([1,8,12,13,14,15,2],[-100,-100,12,13,14,15,2]),
            negative=([1,12,13,14,15,2],[-100,12,13,14,15,2])),
        dict(role='train',positive=([1,9,16,2],[-100,-100,16,2]),negative=None)]


@pytest.mark.parametrize('alpha',[0.,.1])
def test_full_unequal_target_weighted_gradient_matches_direct_autograd_without_retained_multi_graphs(alpha):
    torch.manual_seed(32)
    model=Causal()
    reference=deepcopy(model)
    rows=examples();counts=[sum(y!=-100 for y in e['positive'][1][1:]) for e in rows]
    total=sum(counts);objective=0.
    for row,n in zip(rows,counts):
        ids,labels=row['positive']
        correct=reference(torch.tensor([ids]),torch.tensor([labels]))['loss']
        objective=objective+(n/total)*correct
        if row['negative'] is not None:
            other,other_labels=row['negative']
            wrong=reference(torch.tensor([other]),torch.tensor([other_labels]))['loss']
            objective=objective+alpha*(n/total)*F.softplus(10*(correct-wrong)+.1)
    objective.backward()
    result=matching_gradients(model,rows,alpha=alpha,beta=10.,margin=.1,device='cpu',amp=nullcontext)
    assert result['positive_source_target_tokens']==10
    assert result['total_objective']==pytest.approx(float(objective.detach()),abs=1e-12)
    for actual,expected in zip(model.parameters(),reference.parameters()):
        assert torch.allclose(actual.grad,expected.grad,atol=1e-12,rtol=1e-12)
    assert result['paired_examples']==2 and not result['correctness_or_human_preference_labels']


def records():
    values=[('a','g1','A',2,4094),('b','g2','A',3,3),('c','g3','B',2,4),('d','g4','B',4,5)]
    return [dict(id=name,group_id=group,language=language,prefix_tokens=p,target_tokens=t,
        role='train',body_sha256='body-'+name,prompt_sha256='prompt-'+name) for name,group,language,p,t in values]


def test_all_sources_remain_present_when_a_complete_long_program_has_no_fitting_other_question():
    rows=records()
    pairs=pair_map(rows,51)
    assert np.array_equal(pairs,pair_map(rows,51))
    validate_pair_map(rows,pairs)
    assert len(pairs)==4 and pairs[0]==-1 and pairs[1]==0
    assert pairs[2]==3 and pairs[3]==2
    bad=pairs.copy();bad[0]=1
    with pytest.raises(ValueError,match='complete original prompt pairing'):
        validate_pair_map(rows,bad)


def test_original_full_response_and_eos_are_preserved_with_the_other_complete_question():
    a=([1,4,5,10,11,2],[-100]*3+[10,11,2])
    b=([1,8,12,2],[-100]*2+[12,2])
    assert mismatch_sequence(a,b)==([1,8,10,11,2],[-100,-100,10,11,2])


def test_held_rows_and_changed_targets_reject_before_any_parameter_gradient():
    model=Causal();rows=examples();rows[0]['role']='validation'
    with pytest.raises(ValueError,match='Held roles'):
        matching_gradients(model,rows,alpha=.1,beta=10,margin=.1,device='cpu',amp=nullcontext)
    assert all(p.grad is None for p in model.parameters())
    rows=examples();rows[0]['negative'][1][-2]=20
    with pytest.raises(ValueError,match='every original code target'):
        matching_gradients(model,rows,alpha=.1,beta=10,margin=.1,device='cpu',amp=nullcontext)
    assert all(p.grad is None for p in model.parameters())
