import copy
import pytest

from slm_lab.dense_ppo_evidence_v1 import budget,training_readback
from slm_lab.dense_ppo_inference_v1 import sources,export
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
from test_dense_ppo_control_v1 import stage


def sample():
    learner=stage(1);learner.options['steps']=2;setting=dict(learner.options);rate=setting.pop('learning_rate');sha='d'*64
    plan=dict(training_options=setting,arms=dict(human=dict(learning_rate=rate,human_reward_scale=1)),initial_policy=dict(checkpoint_sha256='a'*64),
        reward_inputs=dict(checkpoint_sha256='b'*64),human_admission_sha256='c'*64,actual_counts=learner.model.counts(),
        initial_parts_sha256=tensor_digest(learner.model.parts.named_parameters()),frozen_dense_sha256=tensor_digest(learner.model.base.named_parameters()),
        reward_provenance=learner.reward.provenance,reward_counts=learner.reward.counts(),vocab_size=learner.model.config.vocab_size,
        max_text_length=learner.model.config.max_text_length,training_device='cpu_fixture',source_exposures=dict(human=budget(learner.data,learner.options)))
    learner.sources=sources(plan,sha);initial=export(learner.model,plan,sha,'initial',0)
    for _ in range(2):learner.update()
    return learner,plan,sha,initial,export(learner.model,plan,sha,'human',2),learner.state(),dict(actual_trained_policy_text_image_critic_cold_reload_exact=True)


def test_real_bounded_controlled_updates_all_actor_critic_states_source_draws_rewards_and_cursors_read_back():
    learner,plan,sha,initial,trained,state,receipt=sample()
    result=training_readback(plan,sha,'human',initial,trained,state,receipt,copy.deepcopy(learner.metrics),learner.data)
    assert result['actual_updates']==2 and result['actor_critic_Adam_states']==len(learner.optimizer.state)
    assert result['declared_source_exposures']['original_source_rows_drawn']==4 and not result['declared_source_exposures']['complete_epoch']
    assert result['actual_own_action_tokens']>=4 and result['raw_reward_mean_is_not_human_judgment_or_policy_quality']


def test_corrupt_original_actions_source_counts_control_reward_Adam_and_GPU_rng_requirement_reject():
    learner,plan,sha,initial,trained,state,receipt=sample()
    for mutate in [lambda s:s['core']['metrics'][0]['rollouts'][0].update(raw_human_reward_score=999.),
        lambda s:s['core']['metrics'][0].update(actual_action_tokens=999),lambda s:s['core']['cursor'].update(actual_prompt_draws=99),
        lambda s:s['sampler'].update(precision='bf16'),
        lambda s:s['core']['optimizer']['state'][0]['exp_avg'].fill_(float('nan'))]:
        bad=copy.deepcopy(state);mutate(bad)
        with pytest.raises(ValueError):training_readback(plan,sha,'human',initial,trained,bad,receipt,bad['core']['metrics'],learner.data)
    with pytest.raises(ValueError):training_readback({**plan,'training_device':'cuda'},sha,'human',initial,trained,state,receipt,learner.metrics,learner.data)


def test_bounded_prefix_budget_rejects_rounding_repeats_or_changed_original_prompt_masks():
    learner=stage(0);setting={**learner.options,'steps':2};value=budget(learner.data,setting)
    assert value['unique_original_source_positions']==4 and value['source_rounding_repeats']==0
    with pytest.raises(ValueError,match='bounded'):budget(learner.data,{**setting,'steps':3})
    learner.data.rows[0]['native']['chosen']['target_start']=3
    with pytest.raises(ValueError):budget(learner.data,setting)
