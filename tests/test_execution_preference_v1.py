"""Compare complete variable-length streamed DPO against direct autograd."""
from contextlib import nullcontext
from copy import deepcopy

import pytest
import torch
from torch.nn import functional as F

from slm_lab.execution_preference_v1 import reference_scores,preference_gradients


class OwnCausalFixture(torch.nn.Module):
    def __init__(self):
        super().__init__();self.embedding=torch.nn.Embedding(32,4,dtype=torch.float64)
        self.head=torch.nn.Linear(4,32,bias=False,dtype=torch.float64)
    def forward(self,input_ids,labels):
        hidden=self.embedding(input_ids).cumsum(1)
        logits=self.head(hidden)
        return {'loss':F.cross_entropy(logits[:,:-1].reshape(-1,32),labels[:,1:].reshape(-1),ignore_index=-100)}


def examples():
    return [dict(id='a',split='train',feedback_source='machine_execution_of_original_publisher_tests',
        human_preferences=False,finite_execution_verified=True,
        chosen=([1,4,5,7,8,2],[-100]*3+[7,8,2]),rejected=([1,4,5,9,10,11,12,2],[-100]*3+[9,10,11,12,2])),
        dict(id='b',split='train',feedback_source='machine_execution_of_original_publisher_tests',
            human_preferences=False,finite_execution_verified=True,
            chosen=([1,3,12,13,14,2],[-100]*2+[12,13,14,2]),rejected=([1,3,15,2],[-100]*2+[15,2]))]


@pytest.mark.parametrize('alpha',[0.,.1])
def test_exact_complete_response_sum_DPO_gradients_and_values_match_direct_autograd(alpha):
    torch.manual_seed(112);model=OwnCausalFixture();reference_model=deepcopy(model)
    with torch.no_grad():reference_model.head.weight.add_(torch.linspace(-.01,.01,128).reshape(32,4))
    rows=examples();cached=reference_scores(reference_model,rows,policy_sha256='a'*64,device='cpu',amp=nullcontext)
    direct=deepcopy(model);objective=0.
    for row in rows:
        def lp(which):
            ids,labels=row[which];n=sum(v!=-100 for v in labels[1:])
            return -n*direct(torch.tensor([ids]),torch.tensor([labels]))['loss']
        ref=cached['records'][row['id']]
        margin=(lp('chosen')-lp('rejected'))-(ref['chosen_logp']-ref['rejected_logp'])
        objective=objective+alpha*(-F.logsigmoid(.1*margin))/len(rows)
    objective.backward()
    result=preference_gradients(model,rows,cached,expected_policy_sha256='a'*64,
        alpha=alpha,beta=.1,device='cpu',amp=nullcontext)
    assert result['weighted_DPO']==pytest.approx(float(objective.detach()),abs=1e-12)
    for actual,expected in zip(model.parameters(),direct.parameters()):
        value=torch.zeros_like(actual) if actual.grad is None else actual.grad
        assert torch.allclose(value,expected.grad,atol=1e-12,rtol=1e-12)


def test_changed_prompt_or_reference_policy_rejects_before_gradients():
    model=OwnCausalFixture();rows=examples();cache=reference_scores(model,rows,policy_sha256='b'*64,device='cpu',amp=nullcontext)
    rows[0]['chosen'][0][1]=6;rows[0]['rejected'][0][1]=6
    with pytest.raises(ValueError,match='whole question/program/mask'):
        preference_gradients(model,rows,cache,expected_policy_sha256='b'*64,alpha=.1,beta=.1,device='cpu',amp=nullcontext)
    assert all(p.grad is None for p in model.parameters())
    with pytest.raises(ValueError,match='exact frozen own reference'):
        preference_gradients(model,examples(),cache,expected_policy_sha256='c'*64,alpha=.1,beta=.1,device='cpu',amp=nullcontext)


def test_held_roles_and_changed_question_between_pair_members_reject_before_scores():
    model=OwnCausalFixture();rows=examples();rows[0]['split']='validation'
    with pytest.raises(ValueError,match='Held source roles'):
        preference_gradients(model,rows,{},expected_policy_sha256='a'*64,alpha=.1,beta=.1,device='cpu',amp=nullcontext)
    rows=examples();rows[0]['rejected'][0][1]=9
    with pytest.raises(ValueError,match='same complete question'):
        reference_scores(model,rows,policy_sha256='a'*64,device='cpu',amp=nullcontext)
    assert all(p.grad is None for p in model.parameters())
