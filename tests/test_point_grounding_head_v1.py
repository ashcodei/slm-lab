import copy

import pytest
import torch

from slm_lab.model import ModelConfig,TinyMultimodalLM
from slm_lab.point_grounding_head_v1 import PointGroundingHead


def small():
    torch.set_num_threads(1);torch.manual_seed(1931)
    config=ModelConfig(vocab_size=32,hidden_size=16,intermediate_size=32,num_layers=2,num_heads=4,num_kv_heads=2,
        max_text_length=64,vision_hidden_size=8,vision_intermediate_size=16,vision_layers=1,vision_heads=2,
        image_size=16,patch_size=4,pool_size=2)
    return TinyMultimodalLM(config)


def test_actual_own_model_gradients_and_text_only_generation_are_preserved():
    base=small();original={n:t.clone() for n,t in base.state_dict().items()}
    ids=torch.tensor([[1,4,5,2],[1,6,7,2]]);pixels=torch.randn(2,3,16,16)
    base.eval();before=base.generate(ids,max_new_tokens=6,eos_token_id=2)
    wrapped=PointGroundingHead(base,rank=8);wrapped.train()
    features=wrapped.query_features(ids).clone();inventory=wrapped.trainable_inventory()
    assert inventory['actual_additional_head_parameters']==2*16*8
    assert inventory['total_unique_parameters']==sum(p.numel() for p in base.parameters())+2*16*8
    optimizer=torch.optim.AdamW([p for p in wrapped.parameters() if p.requires_grad],lr=.003)
    target=torch.tensor([[3.,1.,0.,0.],[0.,0.,0.,5.]])
    for _ in range(4):
        output=wrapped(ids,pixels,targets=target)
        expected=-(target/target.sum(1,keepdim=True)*output['logits'].log_softmax(-1)).sum(1).mean()
        assert torch.equal(expected,output['loss']) and torch.isfinite(output['loss'])
        output['loss'].backward()
        assert any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in base.vision.parameters())
        assert all(p.grad is None for n,p in base.named_parameters() if not n.startswith('vision.'))
        optimizer.step();optimizer.zero_grad(set_to_none=True)
    assert any(not torch.equal(t,base.state_dict()[n]) for n,t in original.items() if n.startswith('vision.'))
    assert all(torch.equal(t,base.state_dict()[n]) for n,t in original.items() if not n.startswith('vision.'))
    assert torch.equal(features,wrapped.query_features(ids))
    base.eval();after=base.generate(ids,max_new_tokens=6,eos_token_id=2)
    assert torch.equal(before,after)


def test_invalid_or_masked_human_target_rejects_before_visual_gradients():
    wrapped=PointGroundingHead(small(),rank=8)
    ids=torch.tensor([[1,4,2]]);pixels=torch.randn(1,3,16,16)
    for target in [torch.zeros(1,4),torch.tensor([[float('nan'),0.,0.,1.]]),torch.tensor([[-1.,0.,0.,2.]])]:
        with pytest.raises(ValueError,match='finite original attention counts'):
            wrapped(ids,pixels,targets=target)
    with pytest.raises(ValueError,match='finite original attention counts'):
        wrapped(ids,pixels,targets=torch.tensor([[1.,0.,0.,0.]]),valid_cells=torch.tensor([[False,True,True,True]]))
    with pytest.raises(ValueError,match='actual content tokens'):
        wrapped(torch.tensor([[1,0,2]]),pixels)
    assert all(p.grad is None for p in wrapped.parameters())
