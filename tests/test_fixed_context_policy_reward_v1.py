import copy
from dataclasses import replace

import pytest
import torch

from slm_lab.corpus import file_hash
from slm_lab.fixed_context_policy_reward_v1 import FixedContextPolicyReward, export, load
from slm_lab.fixed_text_context_v1 import Profile
from slm_lab.own_policy_reward_standalone_v1 import load as load_original
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
from test_own_policy_reward_standalone_v1 import package


def context_model(root, mode):
    _,sources,path=package(root)
    bundle,_=load_original(path,expected_checkpoint_sha256=file_hash(path),expected_sources=sources,allow_mechanics=True)
    ids=torch.tensor([[1,4,7,2]])
    score=bundle.preference_score(ids).clone();base=tensor_digest(bundle.base.named_parameters())
    before_count=sum(p.numel() for p in bundle.parameters())
    old_cache=bundle.policy.prefill(ids)['cache'];original=bundle.config.max_text_length
    profile=Profile(mode,original,original*4,bundle.config.rope_theta)
    model=FixedContextPolicyReward(bundle.actor,bundle.reward,profile,allow_mechanics=True)
    assert model.config.max_text_length==original*4 and model.base.config.max_text_length==original
    assert before_count==sum(p.numel() for p in model.parameters())
    assert model.actor.parts['critic'].weight.count_nonzero()>0
    return model,sources,path,ids,score,base,old_cache


@pytest.mark.parametrize('mode',['extrapolation','linear_PI','YaRN'])
def test_all_profiles_extend_policy_without_changing_original_trained_reward_features_or_critic(tmp_path,mode):
    model,sources,path,ids,score,base,old_cache=context_model(tmp_path,mode)
    assert torch.equal(model.preference_score(ids),score)
    assert tensor_digest(model.base.named_parameters())==base
    assert model.base is model.reward.base and model.counts()['additional_reward_encoder_values']==0
    with pytest.raises(ValueError):model.policy.decode(torch.tensor([[4]]),old_cache)
    with pytest.raises(ValueError):model.actor.export(sources)
    with pytest.raises(ValueError):model.preference_score(torch.ones((1,model.profile.original_text_length+1),dtype=torch.long))
    critic=model.actor.parts['critic'].weight.detach().clone()
    actor_before=tensor_digest(model.actor.parts.named_parameters())
    length=model.profile.original_text_length+8
    long_ids=(torch.arange(length)%max(1,model.config.vocab_size-4)+4)[None];long_ids[0,-1]=2
    labels=long_ids.clone();labels[:,:4]=-100
    optimizer=torch.optim.AdamW([p for _,p in model.actor.named_selected(('actor',))],lr=.001,weight_decay=0.)
    model.train();model(long_ids,labels=labels)['loss'].backward()
    assert all(p.grad is None for p in model.base.parameters())
    assert model.actor.parts['critic'].weight.grad is None
    optimizer.step();optimizer.zero_grad(set_to_none=True);model.eval()
    assert tensor_digest(model.actor.parts.named_parameters())!=actor_before
    assert torch.equal(model.actor.parts['critic'].weight,critic)
    assert tensor_digest(model.base.named_parameters())==base and torch.equal(model.preference_score(ids),score)
    with torch.no_grad():
        full=model(long_ids)['logits'];state=model.policy.prefill(long_ids[:,:model.profile.original_text_length-1])
        suffix=model.policy.decode(long_ids[:,model.profile.original_text_length-1:],state['cache'])
    assert torch.allclose(suffix['logits'],full[:,model.profile.original_text_length-1:],atol=3e-6,rtol=3e-5)


@pytest.mark.parametrize('mode',['extrapolation','linear_PI','YaRN'])
def test_unfused_profile_package_cold_preserves_one_backbone_all_modalities_nonzero_PPO_critic_and_reward(tmp_path,mode):
    model,sources,path,ids,score,base,_=context_model(tmp_path,mode)
    rng=torch.get_rng_state().clone();weights=tensor_digest(model.actor.named_parameters())
    payload=export(model,sources=sources,provenance=dict(actual_small_PPO_source=True),completed_updates=0,
        tokenizer_sha256=sources['tokenizer_sha256'])
    target=path.parent/'fixed-context.pt';torch.save(payload,target)
    cold,metadata=load(target,expected_checkpoint_sha256=file_hash(target),expected_sources=sources,allow_mechanics=True)
    assert torch.equal(rng,torch.get_rng_state()) and tensor_digest(cold.actor.named_parameters())==weights
    assert not ({p.data_ptr() for p in model.parameters()}&{p.data_ptr() for p in cold.parameters()})
    assert cold.base is cold.reward.base and cold.counts()==model.counts()
    assert metadata['original_reward_feature_basis_preserved']
    assert cold.policy.config.max_text_length==model.profile.maximum_text_length
    assert cold.base.config.max_text_length==model.profile.original_text_length
    pixels=torch.linspace(-1,1,3*model.config.max_image_size**2).reshape(1,3,model.config.max_image_size,model.config.max_image_size)
    words=torch.linspace(-1,1,3*8*32).reshape(1,3,8,32)
    with torch.no_grad():
        for image in [None,pixels]:
            a,b=model(ids,image),cold(ids,image)
            assert torch.equal(a['logits'],b['logits']) and torch.equal(a['values'],b['values'])
        assert torch.equal(model.word_logits(words),cold.word_logits(words))
        assert torch.equal(score,cold.preference_score(ids))
        assert torch.equal(model.generate(ids,max_new_tokens=5,use_cache=True),cold.generate(ids,max_new_tokens=5,use_cache=True))
    assert cold.actor.parts['critic'].weight.count_nonzero()>0 and not torch.cuda.is_initialized()
    for fault in ['profile','critic','reward','sources','reward_geometry']:
        corrupt=copy.deepcopy(payload)
        if fault=='profile':corrupt['fixed_policy_profile']['maximum_text_length']*=2
        if fault=='critic':corrupt['original_policy_adapter']['parts']['critic.weight'].fill_(float('nan'))
        if fault=='reward':corrupt['original_reward_provenance']['trained_head_sha256']='0'*64
        if fault=='sources':corrupt['sources']['tokenizer_sha256']='0'*64
        if fault=='reward_geometry':corrupt['complete_base']['config']['max_text_length']*=2
        torch.save(corrupt,target)
        with pytest.raises(ValueError):load(target,expected_checkpoint_sha256=file_hash(target),expected_sources=sources,allow_mechanics=True)
        assert torch.equal(rng,torch.get_rng_state()) and tensor_digest(model.actor.named_parameters())==weights
