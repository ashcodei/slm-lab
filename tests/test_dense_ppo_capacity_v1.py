import copy
import pytest
import torch

from slm_lab.dense_ppo_capacity_v1 import stress
from test_dense_ppo_control_v1 import stage
from test_own_alignment_checkpoint_v1 import assert_nested_equal
from test_own_aligned_assistant_v1 import original_signature


def test_full_declared_horizon_allocates_real_cached_reward_and_both_gradients_without_optimizer_or_source_updates():
    learner=stage(1);learner.options['max_new_tokens']=8;learner.update()
    before=original_signature(learner.model);weights=learner.model.export(learner.sources);adam=copy.deepcopy(learner.optimizer.state_dict())
    rng=torch.get_rng_state().clone();completed=learner.completed
    row=stress(learner,0)
    assert row['full_fixed_action_tokens']==8 and row['actual_prefills']==1 and row['actual_decodes']==7
    assert row['actual_policy_context_tokens']==10 and row['actual_reward_context_tokens']==11
    assert row['actor_critic_finite_gradients'] and row['no_optimizer_step'] and row['fixed_actions_not_training_rollouts_or_human_judgments']
    assert before==original_signature(learner.model) and completed==learner.completed and torch.equal(rng,torch.get_rng_state())
    assert_nested_equal(weights,learner.model.export(learner.sources));assert_nested_equal(adam,learner.optimizer.state_dict())
    assert all(p.grad is None for p in learner.parameters) and all(p.grad is None for p in learner.fixed)


def test_complete_prompt_plus_full_horizon_rejects_overflow_instead_of_truncating():
    learner=stage(0);learner.options['max_new_tokens']=learner.model.config.max_text_length
    rng=torch.get_rng_state().clone()
    with pytest.raises(ValueError,match='must fit'):stress(learner,0)
    assert not learner.optimizer.state and learner.completed==0 and torch.equal(rng,torch.get_rng_state())
