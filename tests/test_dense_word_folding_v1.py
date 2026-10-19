import copy

import pytest
import torch

from slm_lab.dense_word_folding_v1 import fold_words
from slm_lab.own_aligned_assistant_v3 import OwnAlignedAssistantV3
from slm_lab.own_dense_actor_v1 import OwnDenseActor
from test_own_aligned_assistant_v1 import fixture,original_signature
from test_own_dense_growth_v1 import model as make_dense

SOURCES=dict(own_word_source='a'*64)


def source():
    model=OwnAlignedAssistantV3(fixture(),rank=2,reward_hidden=8,word_temporal=4).eval()
    ids=torch.tensor([[1,4,7,8,2]]);pixels=torch.linspace(-1,1,3*16*16).reshape(1,3,16,16)
    words=torch.linspace(-1,1,3*8*32).reshape(1,3,8,32)
    params=model.extension_parameters('vision','word');opt=torch.optim.AdamW(params,lr=.001,foreach=False)
    model.train();model.original.eval()
    loss=model(ids,pixels,labels=ids)['loss']+model.word_logits(words).square().mean()
    loss.backward(inputs=params);opt.step();opt.zero_grad(set_to_none=True);model.eval()
    return model,ids,pixels,words


def test_actual_word_and_native_image_computation_merges_without_extra_parameters_or_language_changes():
    src,ids,pixels,words=source();target,_=make_dense();old=original_signature(src)
    fixed={n:(p.clone(),p._version,p.data_ptr(),p.requires_grad) for n,p in target.named_parameters()
           if not n.startswith(('network.vision.','heads.word.'))}
    counts=target.counts();rng=torch.get_rng_state().clone()
    with torch.no_grad():text=target(ids)['logits'].clone();old_cache=target.network.prefill(ids)['cache']
    report=fold_words(target,src,src.export(SOURCES),expected_sources=SOURCES)
    assert report['retained_extra_vision_encoders_or_adapter_factors']==0 and target.counts()==counts
    assert old==original_signature(src) and torch.equal(rng,torch.get_rng_state())
    for n,(value,version,pointer,flag) in fixed.items():
        p=target.get_parameter(n);assert torch.equal(p,value) and (p._version,p.data_ptr(),p.requires_grad)==(version,pointer,flag)
    with torch.no_grad():
        assert torch.equal(text,target(ids)['logits'])
        assert torch.allclose(src.word_logits(words),target.word_logits(words),atol=2e-6,rtol=2e-5)
        assert torch.allclose(src.aligned.vision(pixels),target.network.vision(pixels),atol=2e-6,rtol=2e-5)
        assert torch.allclose(src(ids,pixels)['logits'],target(ids,pixels)['logits'],atol=2e-6,rtol=2e-5)
    with pytest.raises(ValueError,match='unchanged'):target.network.decode(torch.tensor([[5]]),old_cache)
    actor=OwnDenseActor(target,rank=2).eval()
    with torch.no_grad():assert torch.equal(target(ids,pixels)['logits'],actor(ids,pixels)['logits'])
    assert actor.counts()['actual_unique']<10**9
    unmerged=OwnDenseActor(make_dense()[0],rank=2).eval().export(SOURCES)
    signature=original_signature(actor)
    with pytest.raises(ValueError,match='frozen dense features'):actor.import_export(unmerged,expected_sources=SOURCES)
    assert signature==original_signature(actor)


def test_every_matrix_norm_projection_and_word_value_comes_from_the_original_bound_source():
    src,_,_,_=source();target,_=make_dense();base={n:t.clone() for n,t in src.original.language.base.vision.state_dict().items()}
    fold_words(target,src,src.export(SOURCES),expected_sources=SOURCES)
    from slm_lab.own_code_delta_v1 import matrix_names,norm_names
    for i,n in enumerate(matrix_names(target.config.vision_layers)):
        f=src.parts['vision_linears'][i];assert torch.equal(target.network.vision.get_parameter(n),base[n]+f.up@f.down)
    for i,n in enumerate(norm_names(target.config.vision_layers)):
        assert torch.equal(target.network.vision.get_parameter(n),src.parts['vision_norms'][i].weight)
    f=src.parts['vision_projection'];assert torch.equal(target.network.vision.project.weight,base['project.weight']+f.up@f.down)
    assert all(torch.equal(t,target.heads['word'].state_dict()[n]) for n,t in src.parts['word'].state_dict().items())
    with pytest.raises(ValueError,match='double folding'):fold_words(target,src,src.export(SOURCES),expected_sources=SOURCES)


def test_corrupt_sources_geometry_nonmatching_base_and_fusion_overflow_reject_before_any_target_mutation():
    src,_,_,_=source()
    for kind in ('source','geometry','base','overflow'):
        target,_=make_dense();bad=copy.deepcopy(src);payload=bad.export(SOURCES)
        if kind=='source':payload['sources']['own_word_source']='b'*64
        if kind=='geometry':target.heads['word'].expansion=4
        if kind=='base':
            with torch.no_grad():target.network.vision.patch.weight.add_(1.)
        if kind=='overflow':
            with torch.no_grad():bad.parts['vision_linears'][0].up.fill_(1e30);bad.parts['vision_linears'][0].down.fill_(1e30)
            payload=bad.export(SOURCES)
        before=original_signature(target);weights={n:t.clone() for n,t in target.state_dict().items()}
        with pytest.raises(ValueError):fold_words(target,bad,payload,expected_sources=SOURCES)
        assert before==original_signature(target)
        assert all(torch.equal(t,target.state_dict()[n]) for n,t in weights.items())
