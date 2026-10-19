"""Structural CPU causality checks for the actual own dense/actor code paths.

Small explicit networks exercise the same frozen runtime functions. These
tests do not load current production weights or establish CUDA/full-size
numerical parity, throughput, learning or model-quality improvement.
"""
from contextlib import nullcontext
import json
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer

from slm_lab.CPU_frozen_cast_cache_v1 import CPUFrozenCastCache
from slm_lab.code_tasks import TASKS
from slm_lab.model import ModelConfig
from slm_lab.own_dense_actor_v1 import OwnDenseActor
from slm_lab.own_dense_growth_v1 import OwnDenseAssistant,new_backbone


def network(kind):
    config=ModelConfig(vocab_size=128,hidden_size=16,intermediate_size=32,num_layers=2,num_heads=4,num_kv_heads=2,
        max_text_length=32,vision_hidden_size=16,vision_intermediate_size=32,vision_layers=2,vision_heads=4,
        image_size=16,patch_size=4,pool_size=2)
    backbone=new_backbone(config,984113)
    if kind=='dense':return backbone,None
    own=OwnDenseAssistant(backbone,pointing_rank=4,reward_hidden=4,word_temporal=4,router_hidden=4,detector_hidden=4,word_expansion=1)
    actor=OwnDenseActor(own,rank=2)
    return actor.policy,actor


@pytest.mark.parametrize('kind',['dense','actor'])
@pytest.mark.parametrize('precision',['fp32','bf16'])
@pytest.mark.parametrize('image_side',[None,16,32])
@pytest.mark.parametrize('padded',[False,True])
def test_actual_training_path_has_no_future_text_or_masked_key_influence(kind,precision,image_side,padded):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    model,actor=network(kind);model.train();model.gradient_checkpointing=True
    cache=CPUFrozenCastCache() if kind=='actor' and precision=='bf16' else None
    if cache:cache.attach(model)
    ids=torch.tensor([[1,4,5,6,7,8,9,10],[1,11,12,13,14,15,16,17]])
    attention=torch.ones_like(ids,dtype=torch.bool)
    if padded:attention[0,5:]=False
    pixels=None if image_side is None else torch.randn((2,3,image_side,image_side),generator=torch.Generator().manual_seed(984217))
    def forward(value,mask):
        context=torch.autocast('cpu',dtype=torch.bfloat16) if precision=='bf16' else nullcontext()
        with context:return model(value,pixel_values=pixels,attention_mask=mask)['logits']
    before=[(id(p),p._version,p.data_ptr()) for p in model.parameters()]
    changed=ids.clone();changed[:,3:]+=30
    with torch.no_grad():original=forward(ids,attention);altered=forward(changed,attention)
    assert torch.equal(original[:,:3],altered[:,:3])
    # A masked key in the past must not contaminate any real token, even in
    # later blocks where the masked position itself has a hidden state.
    hole=attention.clone();hole[:,1]=False;changed=ids.clone();changed[:,1]+=60
    with torch.no_grad():original=forward(ids,hole);altered=forward(changed,hole)
    assert torch.equal(original[hole],altered[hole])
    recorded={}
    def leaf(_module,_args,value):
        result=value.detach().requires_grad_(True);recorded['input_embeddings']=result
        return result
    handle=model.embedding.register_forward_hook(leaf)
    try:
        result=forward(ids,attention)
        gradient=torch.autograd.grad(result[:,2,:5].float().sum(),recorded['input_embeddings'])[0]
        assert bool(gradient[:,:3].abs().sum()>0)
        assert not bool(gradient[:,3:].any())
    finally:handle.remove()
    assert before==[(id(p),p._version,p.data_ptr()) for p in model.parameters()]
    assert all(parameter.grad is None for parameter in model.parameters())
    if cache:cache.assert_current();cache.close()


def test_actual_pinned_training_and_evaluation_tokenizers_have_identical_explicit_framing():
    paths=[Path('data/public-reasoning-curriculum-v1/tokenizer.json'),Path('checkpoints/public-pretraining-v1/initial/tokenizer.json')]
    raw=[path.read_bytes() for path in paths]
    assert raw[0]==raw[1]
    for path in paths:
        payload=json.loads(path.read_text());tokenizer=Tokenizer.from_file(str(path))
        assert payload.get('post_processor') is None and payload.get('truncation') is None and payload.get('padding') is None
        assert [tokenizer.token_to_id(token) for token in ['<pad>','<bos>','<eos>','<unk>']]==[0,1,2,3]
        for task in TASKS:
            name,args,spec=task[:3]
            for language in ['Python','JavaScript']:
                prefix='def '+name+'('+args+'):\n    """'+spec+'"""\n' if language=='Python' else 'function '+name+'('+args+') {\n  // '+spec+'\n'
                instruction='User: Write a '+language+' function with signature '+name+'('+args+'). '+spec+'\nReturn only the complete function.\nAssistant:'
                for text in [prefix,instruction]:
                    default=tokenizer.encode(text).ids;explicit=tokenizer.encode(text,add_special_tokens=False).ids
                    assert default==explicit and all(4<=value<32768 for value in default)
                    assert tokenizer.decode(default,skip_special_tokens=False)==text
