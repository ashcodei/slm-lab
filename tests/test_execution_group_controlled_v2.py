import pytest
import torch

from slm_lab.execution_group_controlled_v2 import advantages,trajectory_objective


def test_global_action_normalization_matches_analytic_gradient_for_unequal_replica_lengths():
    torch.manual_seed(957118)
    rewards=[0.,.25,.5,1.];relative=advantages(rewards);lengths=[1,2,4,3];total=sum(lengths)
    for n,adv in zip(lengths,relative):
        x=torch.randn(n,7,requires_grad=True);q=torch.randn(n,7);actions=list(range(n))
        p=x.detach().softmax(-1);chosen=x.detach().log_softmax(-1)[torch.arange(n),actions].tolist()
        loss,_=trajectory_objective(x,q,actions,chosen,adv,total,policy_scale=1,KL_weight=.03)
        actual=torch.autograd.grad(loss,x)[0]
        onehot=torch.nn.functional.one_hot(torch.tensor(actions),7)
        expected=(-adv*(onehot-p)+.03*(p-q.softmax(-1)))/total
        assert torch.allclose(actual,expected,atol=4e-8,rtol=2e-6)


def test_control_removes_reward_policy_gradient_but_keeps_exact_reference_KL():
    torch.manual_seed(957119);x=torch.randn(3,9,requires_grad=True);q=torch.randn(3,9);actions=[2,1,7]
    old=x.detach().log_softmax(-1)[torch.arange(3),actions].tolist()
    loss,_=trajectory_objective(x,q,actions,old,-1.2,11,policy_scale=0,KL_weight=.02)
    assert torch.allclose(torch.autograd.grad(loss,x)[0],.02*(x.detach().softmax(-1)-q.softmax(-1))/11,atol=2e-8)
    assert advantages([0.]*8)==[0.]*8
    with pytest.raises(ValueError):advantages([0.,float('nan')])
    with pytest.raises(ValueError):trajectory_objective(x,q,actions,old,0.,2,policy_scale=1,KL_weight=.02)
