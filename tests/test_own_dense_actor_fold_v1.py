"""Actual small own networks: mechanics only, never production quality."""
import copy

import pytest
import torch

from slm_lab.model import ModelConfig
from slm_lab.own_dense_actor_v1 import OwnDenseActor
from slm_lab.own_dense_actor_fold_v1 import fold_actor,recycle_actor,signature
from slm_lab.own_dense_growth_v1 import OwnDenseAssistant,new_backbone

SOURCES=dict(explicit_small_own_fixture='a'*64)
BUDGET=256*1024**2


def fixture():
    torch.set_num_threads(1)
    config=ModelConfig(vocab_size=128,hidden_size=16,intermediate_size=32,num_layers=2,num_heads=4,num_kv_heads=2,
        max_text_length=32,vision_hidden_size=16,vision_intermediate_size=32,vision_layers=2,vision_heads=4,
        image_size=16,patch_size=4,pool_size=2)
    base=OwnDenseAssistant(new_backbone(config,985117),pointing_rank=4,reward_hidden=4,word_temporal=4,
        router_hidden=4,detector_hidden=4,word_expansion=1)
    actor=OwnDenseActor(base,rank=2)
    ids=torch.tensor([[1,4,5,6,7,8],[1,8,7,6,5,4]])
    optimizer=torch.optim.AdamW([p for _,p in actor.named_selected(('actor','critic'))],lr=.004,foreach=False)
    for _ in range(3):
        actor.train();optimizer.zero_grad(set_to_none=True)
        out=actor(ids,labels=ids);loss=out['loss']+(out['values']-.5).square().mean();loss.backward();optimizer.step()
    optimizer.zero_grad(set_to_none=True);actor.eval()
    assert all(parts.up.count_nonzero()>0 for parts in actor.parts['linears'])
    return actor,ids


@pytest.mark.parametrize('side',[None,16,32])
def test_learned_FP32_text_images_values_masks_and_cached_decode_match_without_source_mutation(side):
    actor,ids=fixture();sig=signature(actor);state={n:p.detach().clone() for n,p in actor.named_parameters()};rng=torch.get_rng_state().clone()
    pixels=None if side is None else torch.linspace(-1,1,2*3*side*side).reshape(2,3,side,side)
    mask=torch.ones_like(ids,dtype=torch.bool);mask[0,4:]=False
    with torch.no_grad():
        source=actor(ids,pixel_values=pixels,attention_mask=mask)
        cache=actor.policy.prefill(ids[:1],pixel_values=None if pixels is None else pixels[:1])['cache']
        cached=actor.policy.decode(torch.tensor([[9]]),cache)
    folded,report=fold_actor(actor,sources=SOURCES,working_budget_bytes=BUDGET)
    with torch.no_grad():
        actual=folded(ids,pixel_values=pixels,attention_mask=mask)
        other=folded.network.prefill(ids[:1],pixel_values=None if pixels is None else pixels[:1])['cache']
        decoded=folded.network.decode(torch.tensor([[9]]),other)
    for key in ['logits','values']:
        torch.testing.assert_close(actual[key],source[key],atol=5e-7,rtol=2e-5)
        torch.testing.assert_close(decoded[key],cached[key],atol=5e-7,rtol=2e-5)
    assert sig==signature(actor) and torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(p,state[n]) and p.grad is None for n,p in actor.named_parameters())
    assert report['folded_counts']['total_unique']==actor.base.counts()['total_unique']
    assert report['removed_adapter_and_separate_critic_values']==sum(p.numel() for p in actor.parts.parameters())
    with torch.no_grad():assert torch.equal(actor.policy.decode(torch.tensor([[9]]),cache)['logits'],cached['logits'])
    with pytest.raises(ValueError):folded.network.decode(torch.tensor([[9]]),cache)


def test_every_vision_word_reward_pointing_router_detector_value_and_tied_table_retained_with_cold_reload():
    actor,ids=fixture();folded,report=fold_actor(actor,sources=SOURCES,working_budget_bytes=BUDGET)
    for first,second in [(actor.base.network.vision,folded.network.vision),(actor.base.heads,folded.heads)]:
        assert first.state_dict().keys()==second.state_dict().keys()
        assert all(torch.equal(v,second.state_dict()[n]) for n,v in first.state_dict().items())
    assert folded.network.embedding.weight is folded.network.lm_head.weight
    pixels=torch.linspace(-1,1,3*16*16).reshape(1,3,16,16)
    with torch.no_grad():
        assert torch.equal(actor.base.word_logits(pixels),folded.word_logits(pixels))
        assert torch.equal(actor.base.detection_logits(pixels),folded.detection_logits(pixels))
    payload=folded.export(SOURCES);cold=copy.deepcopy(actor.base)
    cold.import_export(payload,expected_sources=SOURCES);cold.eval()
    with torch.no_grad():
        assert torch.equal(cold(ids)['logits'],folded(ids)['logits'])
        assert torch.equal(cold.word_logits(pixels),folded.word_logits(pixels))
    assert report['policy_critic_folded_into_existing_dense_value_head']


def test_ambient_BF16_autocast_cannot_round_the_merge_but_BF16_inference_drift_is_measured():
    actor,ids=fixture()
    exact,report=fold_actor(actor,sources=SOURCES,working_budget_bytes=BUDGET)
    with torch.autocast('cpu',dtype=torch.bfloat16):
        ambient,_=fold_actor(actor,sources=SOURCES,working_budget_bytes=BUDGET)
        with torch.no_grad():a=actor(ids)['logits'];b=exact(ids)['logits']
    assert all(torch.equal(p,dict(ambient.named_parameters())[n]) for n,p in exact.named_parameters())
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    assert float((a.float()-b.float()).abs().max())<.01
    assert report['BF16_and_generation_equivalence_not_certified']


def test_recycling_preserves_learned_policy_critic_and_new_Adam_updates_only_fresh_parts():
    actor,ids=fixture();sig=signature(actor);rng=torch.get_rng_state().clone()
    new,report=recycle_actor(actor,sources=SOURCES,working_budget_bytes=BUDGET,seed=985219)
    with torch.no_grad():
        for key in ['logits','values']:torch.testing.assert_close(new(ids)[key],actor(ids)[key],atol=5e-7,rtol=2e-5)
    assert all(not parts.up.any() for parts in new.parts['linears'])
    assert all(torch.equal(a.weight,b.weight) for a,b in zip(actor.parts['norms'],new.parts['norms']))
    assert torch.equal(new.parts['critic'].weight,actor.parts['critic'].weight)
    assert sig==signature(actor) and torch.equal(rng,torch.get_rng_state())
    base_sig=signature(new.base);base_values={n:p.clone() for n,p in new.base.named_parameters()}
    opt=torch.optim.AdamW([p for _,p in new.named_selected(('actor','critic'))],lr=.004,foreach=False)
    assert not opt.state
    new.train();out=new(ids,labels=ids);(out['loss']+(out['values']-1).square().mean()).backward();opt.step();opt.zero_grad(set_to_none=True)
    assert signature(new.base)==base_sig and all(torch.equal(p,base_values[n]) and p.grad is None for n,p in new.base.named_parameters())
    assert any(parts.up.any() for parts in new.parts['linears'])
    assert report['new_optimizer_and_new_reference_bindings_required'] and report['production_cycle_controller_not_installed']
    with pytest.raises(ValueError,match='working-memory budget'):
        recycle_actor(actor,sources=SOURCES,working_budget_bytes=report['required_additional_working_bytes']-1,seed=985221)


def test_invalid_or_active_source_and_insufficient_budget_reject_before_mutation():
    actor,ids=fixture();sig=signature(actor)
    for options in [dict(sources={}),dict(working_budget_bytes=1),dict(sources={'bad':'x'*64})]:
        kwargs=dict(sources=SOURCES,working_budget_bytes=BUDGET);kwargs.update(options)
        with pytest.raises(ValueError):fold_actor(actor,**kwargs)
        assert signature(actor)==sig
    handle=actor.policy.embedding.register_forward_hook(lambda *_:None)
    with pytest.raises(ValueError,match='active hooks'):fold_actor(actor,sources=SOURCES,working_budget_bytes=BUDGET)
    handle.remove();actor.train()
    with pytest.raises(ValueError):fold_actor(actor,sources=SOURCES,working_budget_bytes=BUDGET)
    actor.eval();actor.parts['linears'][0].up.grad=torch.ones_like(actor.parts['linears'][0].up)
    with pytest.raises(ValueError):fold_actor(actor,sources=SOURCES,working_budget_bytes=BUDGET)
    actor.parts['linears'][0].up.grad=None
    with torch.no_grad():actor.parts['linears'][0].up[0,0]=float('nan')
    with pytest.raises(ValueError,match='finite'):fold_actor(actor,sources=SOURCES,working_budget_bytes=BUDGET)
