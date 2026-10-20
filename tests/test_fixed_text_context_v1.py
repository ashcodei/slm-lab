import copy
from dataclasses import replace
import math

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from slm_lab.corpus import file_hash
from slm_lab.deterministic_positions_v1 import install as install_vision
from slm_lab.fixed_context_native_checkpoint_v1 import export, load
from slm_lab.fixed_text_context_v1 import ContextCache, Profile, install, rotary
from slm_lab.model import ModelConfig, rotary as original_rotary
from slm_lab.own_dense_growth_v1 import OwnDenseAssistant, new_backbone


def native(length=32, hidden=16, layers=2):
    torch.set_num_threads(1)
    c=ModelConfig(vocab_size=32,hidden_size=hidden,intermediate_size=hidden*2,num_layers=layers,
        num_heads=1,num_kv_heads=1,max_text_length=length,vision_hidden_size=8,
        vision_intermediate_size=16,vision_layers=1,vision_heads=1,image_size=16,patch_size=4,pool_size=2)
    model=install_vision(OwnDenseAssistant(new_backbone(c,1001211),reward_hidden=8,word_temporal=4,
        pointing_rank=8,router_hidden=8,detector_hidden=4)).eval()
    with torch.no_grad():model.network.value_head.weight.fill_(.03)
    return model


@pytest.mark.parametrize('mode',['extrapolation','linear_PI','YaRN'])
def test_operator_values_and_input_gradients_match_separate_complex_rotation_oracle(mode):
    profile=Profile(mode,4096,16384)
    x=torch.linspace(-1,1,2*5*8).reshape(1,2,5,8).requires_grad_()
    out=rotary(x,profile,4094)
    expected=torch.empty_like(x);upstream=torch.linspace(-.4,.7,x.numel()).reshape_as(x);gradient=torch.empty_like(x)
    for position in range(5):
        for pair in range(4):
            frequency=10000.**(-2*pair/8)
            if mode=='linear_PI':frequency/=4
            if mode=='YaRN':
                rotations=4096*frequency/(2*math.pi);ramp=min(1.,max(0.,(rotations-1)/31))
                frequency*=ramp+(1-ramp)/4
            angle=(4094+position)*frequency;amplitude=1+.1*math.log(4) if mode=='YaRN' else 1.
            c,s=amplitude*math.cos(angle),amplitude*math.sin(angle)
            a,b=x.detach()[...,position,2*pair],x.detach()[...,position,2*pair+1]
            expected[...,position,2*pair]=a*c-b*s;expected[...,position,2*pair+1]=a*s+b*c
            u,v=upstream[...,position,2*pair],upstream[...,position,2*pair+1]
            gradient[...,position,2*pair]=u*c+v*s;gradient[...,position,2*pair+1]=-u*s+v*c
    (out*upstream).sum().backward()
    assert torch.allclose(out,expected,atol=5e-4,rtol=5e-4)
    assert torch.allclose(x.grad,gradient,atol=5e-4,rtol=5e-4)
    if mode=='extrapolation':assert torch.equal(out,original_rotary(x.detach(),10000.,4094))


def test_extrapolation_preserves_existing_text_photo_word_critic_reward_and_parameter_metadata():
    model=native();ids=torch.tensor([[1,4,7,2]])
    pixels=torch.linspace(-1,1,3*32*32).reshape(1,3,32,32);words=torch.linspace(-1,1,3*8*32).reshape(1,3,8,32)
    signature=[(id(p),p.data_ptr(),p._version,p.requires_grad) for p in model.parameters()]
    weights=[p.detach().clone() for p in model.parameters()];counts=model.counts();rng=torch.get_rng_state().clone()
    with torch.no_grad():
        before=[model(ids),model(ids,pixels),model.word_logits(words),model.preference_score(ids)]
        old_cache=model.network.prefill(ids)['cache']
    install(model,Profile('extrapolation',32,128))
    with torch.no_grad():after=[model(ids),model(ids,pixels),model.word_logits(words),model.preference_score(ids)]
    for left,right in zip(before[:2],after[:2]):
        assert torch.equal(left['logits'],right['logits']) and torch.equal(left['values'],right['values'])
    assert all(torch.equal(a,b) for a,b in zip(before[2:],after[2:]))
    assert signature==[(id(p),p.data_ptr(),p._version,p.requires_grad) for p in model.parameters()]
    assert all(torch.equal(p,q) for p,q in zip(model.parameters(),weights)) and model.counts()==counts
    assert torch.equal(rng,torch.get_rng_state())
    with pytest.raises(ValueError):model.network.decode(torch.tensor([[4]]),old_cache)
    with pytest.raises(ValueError):model.export({'own':'a'*64})
    with pytest.raises(ValueError):model.preference_score(torch.ones((1,33),dtype=torch.long))


@pytest.mark.parametrize('mode',['extrapolation','linear_PI','YaRN'])
def test_cached_single_and_multi_token_decoding_crosses_original_4096_boundary_with_one_fixed_profile(mode):
    model=native(4096,8,1);profile=Profile(mode,4096,16384);install(model,profile)
    ids=(torch.arange(4100)%28+4)[None]
    with torch.no_grad():
        full=model(ids)
        state=model.network.prefill(ids[:,:4095]);assert type(state['cache']) is ContextCache
        a=model.network.decode(ids[:,4095:4096],state['cache'])
        b=model.network.decode(ids[:,4096:],a['cache'])
    assert torch.allclose(a['logits'],full['logits'][:,4095:4096],atol=2e-6,rtol=2e-5)
    assert torch.allclose(b['logits'],full['logits'][:,4096:],atol=2e-6,rtol=2e-5)
    assert torch.allclose(b['values'],full['values'][:,4096:],atol=2e-6,rtol=2e-5)
    assert b['cache'].text_length==4100 and b['cache'].profile_fingerprint==profile.fingerprint
    other=native(4096,8,1);install(other,profile)
    with pytest.raises(ValueError):other.network.decode(torch.tensor([[4]]),b['cache'])
    with torch.no_grad():model.network.norm.weight.add_(.01)
    with pytest.raises(ValueError):model.network.decode(torch.tensor([[4]]),b['cache'])


def test_actual_small_16384_token_training_backward_has_finite_nonzero_gradients():
    model=native(4096,8,1);install(model,Profile('linear_PI',4096,16384));model.train()
    model.network.gradient_checkpointing=True
    ids=(torch.arange(16384)%28+4)[None];loss=model(ids,labels=ids)['loss'];loss.backward()
    assert torch.isfinite(loss) and any(p.grad is not None and bool((p.grad!=0).any()) for p in model.parameters())
    assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
    assert model.config.max_text_length==16384 and sum(p.numel() for p in model.parameters())<100000


@pytest.mark.parametrize('mode',['extrapolation','linear_PI','YaRN'])
def test_trained_native_profile_checkpoint_cold_loads_all_values_modalities_cached_actions_and_caller_RNG(tmp_path,mode):
    model=native();ids=torch.tensor([[1,4,7,2]])
    optimizer=torch.optim.AdamW([p for _,p in model.language_named_parameters()],lr=.001)
    model.train();model(ids,labels=ids)['loss'].backward();optimizer.step();optimizer.zero_grad(set_to_none=True);model.eval()
    install(model,Profile(mode,32,128));sources={'own_native_sha256':'a'*64}
    token=Tokenizer(WordLevel(vocab={'<pad>':0,'<bos>':1,'<eos>':2,**{f't{i}':i for i in range(3,32)}},unk_token='<pad>'))
    token.save(str(tmp_path/'tokenizer.json'));rng=torch.get_rng_state().clone()
    payload=export(model,sources=sources,tokenizer_sha256=file_hash(tmp_path/'tokenizer.json'),completed_updates=1)
    path=tmp_path/'model.pt';torch.save(payload,path)
    cold,meta=load(path,expected_checkpoint_sha256=file_hash(path),expected_sources=sources)
    assert all(torch.equal(a,b) for a,b in zip(model.parameters(),cold.parameters()))
    assert not ({p.data_ptr() for p in model.parameters()}&{p.data_ptr() for p in cold.parameters()})
    assert cold.network.embedding.weight is cold.network.lm_head.weight
    assert torch.equal(rng,torch.get_rng_state()) and meta['profile_fingerprint']==model._text_context_profile.fingerprint
    pixels=torch.linspace(-1,1,3*32*32).reshape(1,3,32,32);words=torch.linspace(-1,1,3*8*32).reshape(1,3,8,32)
    with torch.no_grad():
        for image in [None,pixels]:
            a,b=model(ids,image),cold(ids,image)
            assert torch.equal(a['logits'],b['logits']) and torch.equal(a['values'],b['values'])
        assert torch.equal(model.word_logits(words),cold.word_logits(words))
        assert torch.equal(model.generate(ids,max_new_tokens=5,use_cache=True),cold.generate(ids,max_new_tokens=5,use_cache=True))
    if mode!='extrapolation':
        with pytest.raises(ValueError):cold.preference_score(ids)
    changed=copy.deepcopy(payload);changed['fixed_text_context_profile']['maximum_text_length']=256;torch.save(changed,path)
    with pytest.raises(ValueError):load(path,expected_checkpoint_sha256=file_hash(path),expected_sources=sources)
    torch.save(payload,path)
    with pytest.raises(ValueError):load(path,expected_checkpoint_sha256=file_hash(path),expected_sources={'own_native_sha256':'b'*64})
