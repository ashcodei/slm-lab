import pytest
import torch

from slm_lab.execution_group_controlled_v2 import trajectory_objective
from slm_lab.single_update_code_policy_v8 import objective


def test_fresh_group_gradient_equals_independent_score_function_and_original_ratio1_gradient():
    torch.manual_seed(959219)
    values = torch.randn(5,11)
    reference = torch.randn_like(values)
    actions = [1,4,3,9,2]
    logits = values.clone().requires_grad_()
    old_logits = values.clone().requires_grad_()
    behavior = values.log_softmax(-1)[torch.arange(5),actions].tolist()
    loss,_ = objective(logits,reference,actions,behavior,.7,13,policy_scale=1,KL_weight=.01,
                     freshly_sampled_single_update=True)
    old,_ = trajectory_objective(old_logits,reference,actions,behavior,.7,13,policy_scale=1,KL_weight=.01)
    loss.backward()
    old.backward()
    probability = values.softmax(-1)
    indicator = torch.nn.functional.one_hot(torch.tensor(actions),11)
    expected = (.7*(probability-indicator)+.01*(probability-reference.softmax(-1)))/13
    assert torch.allclose(logits.grad,expected,atol=3e-8,rtol=3e-6)
    assert torch.allclose(logits.grad,old_logits.grad,atol=3e-8,rtol=3e-6)
    perturbed = values.clone().requires_grad_()
    changed_behavior = [p-.0001 for p in behavior]
    other,_ = objective(perturbed,reference,actions,changed_behavior,.7,13,policy_scale=1,KL_weight=.01,
                      freshly_sampled_single_update=True)
    other.backward()
    assert torch.equal(logits.grad,perturbed.grad)


def test_off_policy_or_incomplete_behavior_rejects_before_gradient():
    logits = torch.zeros(2,4,requires_grad=True)
    for behavior,fresh in [([-1.,-1.],False),([-1.],True),([.1,-1.],True)]:
        with pytest.raises(ValueError,match='fresh'):
            objective(logits,logits.detach(),[1,2],behavior,1.,2,policy_scale=1,KL_weight=.01,
                      freshly_sampled_single_update=fresh)
    assert logits.grad is None
