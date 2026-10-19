import copy

import pytest
import torch

from slm_lab.dense_ppo_control_v1 import DenseControlledPPOStage,validate_controlled
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
from test_dense_ppo_v1 import stage as original_stage
from test_own_alignment_checkpoint_v1 import assert_nested_equal
from test_own_aligned_assistant_v1 import original_signature


def stage(scale):
    torch.manual_seed(952531);old=original_stage()
    return DenseControlledPPOStage(old.model,old.optimizer,old.data,old.reward,old.options,old.sources,human_reward_scale=scale)


def test_matched_profiles_sample_identical_initial_actions_and_preserve_raw_human_scores_but_apply_only_declared_reward():
    control,treated=stage(0),stage(1)
    assert tensor_digest(control.model.named_parameters())==tensor_digest(treated.model.named_parameters())
    assert tensor_digest(control.reward.named_parameters())==tensor_digest(treated.reward.named_parameters())
    torch.manual_seed(952631);rng=torch.get_rng_state().clone();a=control.collect(control.positions(0))
    torch.set_rng_state(rng);b=treated.collect(treated.positions(0))
    for x,y in zip(a,b):
        assert x['response_ids']==y['response_ids'] and x['actual_sampling_logps']==y['actual_sampling_logps']
        assert x['raw_human_reward_score']==y['raw_human_reward_score'] and x['human_reward_score']==0.
        assert y['human_reward_score']==y['raw_human_reward_score']
        assert torch.equal(torch.tensor(x['token_rewards'][:-1]),torch.tensor(y['token_rewards'][:-1]))
        assert torch.isclose(torch.tensor(y['token_rewards'][-1]-x['token_rewards'][-1]),torch.tensor(y['raw_human_reward_score']))
        assert x['reference_logps']==y['reference_logps'] and x['scoring_ids']==y['scoring_ids']


def test_both_profiles_actual_actor_critic_update_recovery_Adam_RNG_and_pristine_checkpoint():
    for scale in (0,1):
        learner=stage(scale);fixed=original_signature(learner.model.base);scorer=original_signature(learner.reward)
        learner.update();saved=learner.state();pristine=copy.deepcopy(saved);expected=learner.update()
        weights=learner.model.export(learner.sources);adam=copy.deepcopy(learner.optimizer.state_dict());rng=torch.get_rng_state().clone()
        learner.resume(saved);assert learner.update()==expected
        assert_nested_equal(weights,learner.model.export(learner.sources));assert_nested_equal(adam,learner.optimizer.state_dict());assert_nested_equal(saved,pristine)
        assert torch.equal(rng,torch.get_rng_state()) and fixed==original_signature(learner.model.base) and scorer==original_signature(learner.reward)


def test_changed_raw_or_applied_reward_and_control_profile_reject_before_any_model_Adam_or_RNG_mutation():
    learner=stage(1);learner.update();saved=learner.state();weights=learner.model.export(learner.sources)
    adam=copy.deepcopy(learner.optimizer.state_dict());rng=torch.get_rng_state().clone()
    for mutate in [lambda s:s.update(terminal_human_reward_scale=0),
        lambda s:s['core']['metrics'][0]['rollouts'][0].update(terminal_human_reward_scale=0),
        lambda s:s['core']['metrics'][0]['rollouts'][0].update(raw_human_reward_score=999.),
        lambda s:s['core']['metrics'][0]['rollouts'][0].update(human_reward_score=999.)]:
        bad=copy.deepcopy(saved);mutate(bad);before=original_signature(learner.model)
        with pytest.raises(ValueError):learner.resume(bad)
        assert before==original_signature(learner.model);assert_nested_equal(weights,learner.model.export(learner.sources))
        assert_nested_equal(adam,learner.optimizer.state_dict());assert torch.equal(rng,torch.get_rng_state())
