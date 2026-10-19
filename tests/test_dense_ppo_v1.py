import copy
import math

import pytest
import torch

from slm_lab.dense_ppo_reward_v1 import FrozenGeneralReward
from slm_lab.dense_ppo_v1 import DensePPOStage,action_outputs,validate_rollouts
from slm_lab.point_grounding_evaluation_v1 import state_digest
from slm_lab.whole_human_reward_inference_v1 import frozen_general_reward
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
from test_dense_human_preference_stage_v1 import pairs
from test_own_dense_actor_v1 import actor,SOURCES
from test_own_alignment_checkpoint_v1 import assert_nested_equal
from test_own_aligned_assistant_v1 import original_signature
from test_whole_human_reward_v1 import setup


def reward():
    original,*_=setup();encoder=original.original.language.base;head=original.parts['reward']
    return FrozenGeneralReward(encoder,head,dict(encoder_sha256=state_digest(encoder),
        head_sha256=tensor_digest(head.named_parameters()),fixture_only=True))


def stage():
    model=actor();data=pairs();scorer=reward()
    setting=dict(seed=951031,steps=3,prompts_per_update=2,max_new_tokens=3,learning_rate=.001,
        betas=[.9,.95],weight_decay=.01,gradient_clip=1.,KL_coefficient=.05,gamma=1.,GAE_lambda=.95,clip=.2,value_coefficient=.5)
    opt=torch.optim.AdamW([p for _,p in model.named_selected(('actor','critic'))],lr=setting['learning_rate'],
        betas=tuple(setting['betas']),weight_decay=setting['weight_decay'],foreach=False)
    return DensePPOStage(model,opt,data,scorer,setting,SOURCES)


def test_original_encoder_reward_matches_existing_feature_entrypoint_and_rejects_changed_basis_or_training():
    original,*_=setup();encoder=original.original.language.base;head=original.parts['reward'];ids=torch.tensor([[1,7,9,2]])
    before=original_signature(original)
    scorer=FrozenGeneralReward(encoder,head,dict(encoder_sha256=state_digest(encoder),head_sha256=tensor_digest(head.named_parameters())))
    assert torch.equal(scorer.score(ids),frozen_general_reward(original,ids).squeeze())
    assert not scorer.score(ids).requires_grad and before==original_signature(original)
    assert scorer.counts()['actual_training_only_total']==sum(p.numel() for p in encoder.parameters())+sum(p.numel() for p in head.parameters())
    with pytest.raises(ValueError,match='evaluation'):scorer.train()
    with pytest.raises(ValueError,match='EOS'):scorer.score(torch.tensor([[1,7,9]]))
    with torch.no_grad():encoder.norm.weight.add_(.01)
    with pytest.raises(ValueError,match='fixed'):scorer.score(ids)
    with pytest.raises(ValueError,match='feature basis'):FrozenGeneralReward(encoder,head,scorer.provenance)


def test_actual_own_sampling_actor_critic_update_private_Adam_RNG_recovery_and_all_frozen_values():
    learner=stage();base=original_signature(learner.model.base);frozen=tensor_digest(learner.model.base.named_parameters())
    scorer=original_signature(learner.reward);score_digest=tensor_digest(learner.reward.named_parameters())
    before=tensor_digest(learner.model.parts.named_parameters());torch.manual_seed(951131)
    first=learner.update();saved=learner.state();pristine=copy.deepcopy(saved)
    assert first['actual_action_tokens']>=2 and all(r['temperature']==1 for r in first['rollouts'])
    assert len(learner.optimizer.state)==len(learner.model.named_selected(('actor','critic')))
    assert learner.model.parts['critic'].weight.count_nonzero()>0 and before!=tensor_digest(learner.model.parts.named_parameters())
    expected=learner.update();weights=learner.model.export(SOURCES);adam=copy.deepcopy(learner.optimizer.state_dict());rng=torch.get_rng_state().clone()
    learner.resume(saved);assert learner.update()==expected
    assert_nested_equal(weights,learner.model.export(SOURCES));assert_nested_equal(adam,learner.optimizer.state_dict());assert_nested_equal(saved,pristine)
    assert torch.equal(rng,torch.get_rng_state()) and base==original_signature(learner.model.base) and scorer==original_signature(learner.reward)
    assert frozen==tensor_digest(learner.model.base.named_parameters()) and score_digest==tensor_digest(learner.reward.named_parameters())


def test_token_weighted_actor_critic_gradients_match_independent_joint_objective():
    learner=stage();oracle=copy.deepcopy(learner.model)
    opt=torch.optim.AdamW([p for _,p in oracle.named_selected(('actor','critic'))],lr=learner.options['learning_rate'],
        betas=tuple(learner.options['betas']),weight_decay=learner.options['weight_decay'],foreach=False)
    torch.manual_seed(951231);rng=torch.get_rng_state().clone();rows=learner.collect(learner.positions(0))
    logps=[];values=[]
    for row in rows:
        ids=torch.tensor([row['prompt_ids']+row['response_ids']]);p,v=action_outputs(oracle,ids,len(row['prompt_ids']),learner.amp)
        logps.append(p);values.append(v)
    logps,values=torch.cat(logps),torch.cat(values)
    old,adv,old_v,returns=[torch.cat([torch.tensor(r[k]) for r in rows]) for k in ('old_logps','advantages','old_values','returns')]
    ratio=(logps-old).exp();s1=ratio*adv;s2=ratio.clamp(.8,1.2)*adv
    clipped=old_v+(values-old_v).clamp(-.2,.2)
    loss=-torch.minimum(s1,s2).mean()+.5*.5*torch.maximum((values-returns).square(),(clipped-returns).square()).mean()
    loss.backward(inputs=[p for _,p in oracle.named_selected(('actor','critic'))])
    torch.nn.utils.clip_grad_norm_([p for _,p in oracle.named_selected(('actor','critic'))],1.);opt.step()
    torch.set_rng_state(rng);record=learner.update()
    assert math.isclose(record['token_weighted_PPO_loss'],float(loss.detach()),abs_tol=1e-7,rel_tol=1e-6)
    assert all(torch.allclose(p,oracle.parts.get_parameter(n),atol=3e-7,rtol=2e-5) for n,p in learner.model.parts.named_parameters())
    # GAE is also checked without calling the implementation primitive.
    for row in rows:
        gae=0.;expected=[];v=row['old_values']+[0.]
        for i in reversed(range(len(row['response_ids']))):
            gae=row['token_rewards'][i]+v[i+1]-v[i]+.95*gae;expected.insert(0,gae)
        assert torch.allclose(torch.tensor(expected),torch.tensor(row['raw_advantages']),atol=1e-6,rtol=1e-6)


def test_corrupt_own_actions_rewards_advantages_source_basis_and_Adam_reject_before_mutation():
    learner=stage();learner.update();saved=learner.state();weights=learner.model.export(SOURCES)
    adam=copy.deepcopy(learner.optimizer.state_dict());rng=torch.get_rng_state().clone()
    mutations=[lambda s:s['core']['metrics'][0]['rollouts'][0]['prompt_ids'].__setitem__(0,7),
        lambda s:s['core']['metrics'][0]['rollouts'][0]['scoring_ids'].append(3),
        lambda s:s['core']['metrics'][0]['rollouts'][0]['token_rewards'].__setitem__(0,999.),
        lambda s:s['core']['metrics'][0]['rollouts'][0]['advantages'].__setitem__(0,999.),
        lambda s:s['core']['metrics'][0]['rollouts'][0]['actual_sampling_logps'].__setitem__(0,-999.),
        lambda s:s['reward_provenance'].update(encoder_sha256='0'*64),
        lambda s:s['core']['optimizer']['state'][0]['exp_avg'].fill_(float('nan'))]
    for mutate in mutations:
        bad=copy.deepcopy(saved);mutate(bad);before=original_signature(learner.model)
        with pytest.raises(ValueError):learner.resume(bad)
        assert before==original_signature(learner.model);assert_nested_equal(weights,learner.model.export(SOURCES))
        assert_nested_equal(adam,learner.optimizer.state_dict());assert torch.equal(rng,torch.get_rng_state())


def test_whole_prompt_fit_rejects_before_sampling_RNG_and_held_prompts_are_never_optimized():
    learner=stage();learner.options['max_new_tokens']=learner.model.config.max_text_length
    rng=torch.get_rng_state().clone()
    with pytest.raises(ValueError,match='without truncation'):learner.update()
    assert torch.equal(rng,torch.get_rng_state()) and not learner.optimizer.state
    learner.data.role='valid'
    with pytest.raises(ValueError,match='train-only'):DensePPOStage(learner.model,learner.optimizer,learner.data,learner.reward,learner.options,SOURCES)


def test_EOS_action_and_budget_cutoff_keep_scoring_only_EOS_out_of_policy_actions(monkeypatch):
    learner=stage()
    for token in (2,7):
        monkeypatch.setattr(torch,'multinomial',lambda probs,num:torch.tensor([token],device=probs.device))
        rows=learner.collect(learner.positions(0))
        for row in rows:
            if token==2:
                assert row['response_ids']==[2] and row['termination']=='EOS' and not row['scoring_only_EOS_added']
            else:
                assert row['response_ids']==[7]*3 and row['termination']=='response_budget' and row['scoring_only_EOS_added']
                assert row['scoring_ids']==row['prompt_ids']+[7]*3+[2]
            assert row['finite_horizon_terminal'] and len(row['old_logps'])==len(row['response_ids'])
