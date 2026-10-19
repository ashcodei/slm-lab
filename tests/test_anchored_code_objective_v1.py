import pytest
import torch
from torch.nn import functional as F

from slm_lab.anchored_code_objective_v1 import reference_anchor
from slm_lab.single_update_code_policy_v8 import objective
from slm_lab.execution_group_controlled_v2 import advantages


def test_unequal_mask_replica_gradients_equal_direct_global_CE_without_second_average():
    torch.manual_seed(900)
    whole = torch.randn(3, 6, 7, requires_grad=True)
    labels = torch.tensor([[-100, -100, 3, 4, 1, 2], [-100, 1, 2, -100, -100, -100], [-100, -100, 5, 3, 2, -100]])
    global_targets = int((labels[:, 1:] != -100).sum())
    losses = [reference_anchor(whole[i:i+1], labels[i:i+1], global_targets, weight=.7)[0] for i in range(3)]
    grad = torch.autograd.grad(sum(losses), whole, retain_graph=True)[0]
    direct = .7 * F.cross_entropy(whole[:, :-1].reshape(-1, 7), labels[:, 1:].reshape(-1), ignore_index=-100)
    expected = torch.autograd.grad(direct, whole)[0]
    torch.testing.assert_close(grad, expected, atol=2e-8, rtol=2e-6)
    assert grad[:, -1].count_nonzero() == 0


def test_positive_partial_and_flat_group_rewards_always_retain_exact_anchor_gradient():
    torch.manual_seed(901)
    for rewards in [[0., 0.], [.5, .5], [.2, .9], [1., 1.]]:
        anchor_logits = torch.randn(1, 5, 6, requires_grad=True)
        labels = torch.tensor([[-100, -100, 3, 4, 2]])
        own = [torch.randn(2, 6, requires_grad=True) for _ in rewards]
        actions = [[1, 2], [3, 2]]; relative = advantages(rewards)
        loss, metrics = reference_anchor(anchor_logits, labels, 3)
        for i in range(2):
            reference = own[i].detach().clone()
            old = own[i].detach().log_softmax(-1).gather(1, torch.tensor(actions[i])[:, None]).flatten().tolist()
            term, _ = objective(own[i], reference, actions[i], old, relative[i], 4,
                policy_scale=1, KL_weight=.01, freshly_sampled_single_update=True)
            loss = loss + term
        actual = torch.autograd.grad(loss, [anchor_logits, *own], retain_graph=True)
        direct = F.cross_entropy(anchor_logits[:, :-1].reshape(-1, 6), labels[:, 1:].reshape(-1), ignore_index=-100)
        expected = torch.autograd.grad(direct, anchor_logits)[0]
        torch.testing.assert_close(actual[0], expected)
        assert actual[0].abs().sum() > 0 and metrics['original_reference_supervision_is_unconditional_on_execution_rewards']
        if len(set(rewards)) > 1:
            assert all(g.abs().sum() > 0 for g in actual[1:])
    with pytest.raises(ValueError):
        reference_anchor(torch.zeros(1, 4, 6), torch.full((1, 4), -100), 1)
