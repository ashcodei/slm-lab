import pytest
import torch

from slm_lab.execution_group_RL_v1 import group_objective


def test_group_execution_objective_and_full_reference_KL_match_independent_token_oracle():
    torch.manual_seed(51881);new=[torch.randn(n,7,requires_grad=True) for n in [2,3,1]]
    references=[torch.randn_like(x,requires_grad=True) for x in new];actions=[[1,4],[2,5,1],[3]]
    old=[x.detach().log_softmax(-1).gather(1,torch.tensor(a)[:,None]).squeeze(1).tolist() for x,a in zip(new,actions)]
    scores=[0.,.5,1.];loss,metric=group_objective(new,references,actions,old,scores,KL_weight=.03)
    rewards=torch.tensor(scores,dtype=torch.float64);adv=(rewards-rewards.mean())/(rewards.std(unbiased=False)+1e-6);oracle=0.
    for x,q,a,p,advantage in zip(new,references,actions,old,adv):
        lp=x.log_softmax(-1);lq=q.detach().log_softmax(-1);chosen=lp[torch.arange(len(a)),a]
        ratio=(chosen-torch.tensor(p)).exp();surrogate=torch.minimum(ratio*advantage.float(),ratio.clamp(.8,1.2)*advantage.float())
        oracle=oracle+(-surrogate.sum()+.03*(lq.exp()*(lq-lp)).sum())/6
    assert float(loss.detach())==pytest.approx(float(oracle.detach()),abs=2e-7)
    actual=torch.autograd.grad(loss,new,retain_graph=True);expected=torch.autograd.grad(oracle,new)
    assert all(torch.allclose(a,b,atol=1e-7,rtol=1e-6) for a,b in zip(actual,expected))
    assert not metric['verified_public_target_fallback_required']
    assert all(q.grad is None for q in references)


def test_all_failed_group_exposes_flat_RL_signal_and_requires_public_target_fallback_without_invented_reward():
    logits=[torch.randn(2,5,requires_grad=True) for _ in range(4)];refs=[x.detach().clone() for x in logits];actions=[[1,2]]*4
    old=[x.detach().log_softmax(-1)[:,1].tolist() for x in logits]
    loss,metric=group_objective(logits,refs,actions,old,[0.]*4,KL_weight=0.)
    assert metric['group_advantages']==[0.]*4 and metric['verified_public_target_fallback_required']
    assert float(loss.detach())==0 and all(g.count_nonzero()==0 for g in torch.autograd.grad(loss,logits))
    with pytest.raises(ValueError):group_objective(logits,refs,actions,old,[0.,0.,0.,float('nan')])
