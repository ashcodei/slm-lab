import copy

import pytest
import torch

from slm_lab.code_residual_branch_v1 import RoutedCodeResidual
from slm_lab.model import ModelConfig,TinyMultimodalLM
from slm_lab.recurrent_ablation import state_digest


def config():
    return ModelConfig(vocab_size=64,hidden_size=32,intermediate_size=64,num_layers=4,
        num_heads=4,num_kv_heads=2,max_text_length=64,vision_hidden_size=16,
        vision_intermediate_size=32,vision_layers=1,vision_heads=2,image_size=16,patch_size=4,pool_size=2)


@pytest.mark.parametrize('image_size',[None,16,32])
def test_initial_routes_exactly_match_whole_original_text_image_masks_values_and_loss(image_size):
    torch.manual_seed(72);base=TinyMultimodalLM(config()).eval();ids=torch.tensor([[1,7,8,9,2],[1,5,2,0,0]])
    mask=ids!=0;labels=ids.clone();labels[~mask]=-100;labels[:,:2]=-100
    pixels=None if image_size is None else torch.randn(2,3,image_size,image_size)
    # The controlled reference is the frozen backbone used by this architecture.
    # Freezing alone can change a CPU image kernel's final rounding bits.
    for parameter in base.parameters():parameter.requires_grad_(False)
    with torch.no_grad():
        original=base(ids,pixels,attention_mask=mask,labels=labels,return_hidden=True)
    routed=RoutedCodeResidual(base,width=24).eval()
    assert routed.code.embedding is base.embedding and routed.code.vision is base.vision
    assert routed.code.lm_head.weight is base.embedding.weight
    for route in ['general','code']:
        with torch.no_grad():
            actual=routed(ids,pixels,attention_mask=mask,labels=labels,return_hidden=True,route=route)
        for name in original:torch.testing.assert_close(actual[name],original[name],rtol=0,atol=0)
    assert routed.parameter_counts()['total_unique']==routed.parameter_counts()['frozen_base']+2*32*24*4


def test_initial_training_and_inference_paths_agree_with_explicit_floating_tolerance():
    torch.manual_seed(72);base=TinyMultimodalLM(config()).eval();routed=RoutedCodeResidual(base,width=24).eval()
    ids=torch.tensor([[1,7,8,9,2]]);pixels=torch.randn(1,3,16,16)
    with torch.no_grad():original=base(ids,pixels,labels=ids)
    training=routed(ids,pixels,labels=ids,route='code')
    torch.testing.assert_close(training['logits'],original['logits'],rtol=1e-5,atol=1e-6)
    torch.testing.assert_close(training['loss'],original['loss'],rtol=1e-5,atol=1e-6)


def test_freezing_original_image_backbone_has_only_explicit_small_floating_difference():
    torch.manual_seed(72);base=TinyMultimodalLM(config()).eval();ids=torch.tensor([[1,7,8,9,2],[1,5,2,0,0]])
    pixels=torch.randn(2,3,16,16);mask=ids!=0
    with torch.no_grad():original=base(ids,pixels,attention_mask=mask)['logits']
    routed=RoutedCodeResidual(base,width=24).eval()
    with torch.no_grad():frozen=routed(ids,pixels,attention_mask=mask)['logits']
    torch.testing.assert_close(frozen,original,rtol=1e-5,atol=1e-6)
    assert torch.equal(frozen.argmax(-1),original.argmax(-1))


def trained(image=False):
    torch.manual_seed(14);routed=RoutedCodeResidual(TinyMultimodalLM(config()),width=24)
    routed.code.gradient_checkpointing=True;ids=torch.tensor([[1,7,9,11,13,2]])
    pixels=torch.randn(1,3,16,16) if image else None;labels=ids.clone();labels[:,:3]=-100
    routed.eval();before=state_digest(routed.base.state_dict());general=routed(ids,pixels)['logits'].detach().clone()
    code_before=routed(ids,pixels,route='code')['logits'].detach().clone()
    optimizer=torch.optim.AdamW(routed.adapter_parameters(),lr=.02,weight_decay=.01)
    for _ in range(3):
        routed.train();optimizer.zero_grad(set_to_none=True);routed(ids,pixels,labels=labels,route='code')['loss'].backward()
        for parameter in routed.residuals.parameters():assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert all(p.grad is None for p in routed.base.parameters());optimizer.step();routed.assert_partition()
    routed.eval()
    assert state_digest(routed.base.state_dict())==before
    torch.testing.assert_close(routed(ids,pixels)['logits'],general,rtol=0,atol=0)
    assert not torch.equal(routed(ids,pixels,route='code')['logits'],code_before)
    assert all(p.grad.abs().sum()>0 for p in routed.residuals.parameters())
    return routed,ids,pixels


@pytest.mark.parametrize('image',[False,True])
def test_actual_adam_updates_change_code_only_and_trained_cache_matches_whole_forward(image):
    routed,ids,pixels=trained(image);model=routed.code
    state=model.prefill(ids[:,:3],pixels)
    for position in range(3,6):
        state=model.decode(ids[:,position:position+1],state['cache'])
        expected=model(ids[:,:position+1],pixels)['logits'][:,-1:]
        torch.testing.assert_close(state['logits'],expected,rtol=1e-5,atol=1e-6)
    state=model.decode(ids[:,3:6],model.prefill(ids[:,:3],pixels)['cache'])
    torch.testing.assert_close(state['logits'],model(ids,pixels)['logits'][:,3:6],rtol=1e-5,atol=1e-6)


def test_cache_routes_and_parameter_versions_never_mix():
    routed,ids,pixels=trained();general=routed.base.prefill(ids[:,:3]);code=routed.code.prefill(ids[:,:3])
    with pytest.raises(ValueError,match='code route'):routed.code.decode(ids[:,3:4],general['cache'])
    with pytest.raises(ValueError,match='belong'):routed.base.decode(ids[:,3:4],code['cache'])
    with torch.no_grad():routed.residuals[0].up.weight.add_(.01)
    with pytest.raises(ValueError,match='parameters changed'):routed.code.decode(ids[:,3:4],code['cache'])
    state=routed.base.decode(ids[:,3:4],general['cache'])
    torch.testing.assert_close(state['logits'],routed.base(ids[:,:4])['logits'][:,-1:],rtol=1e-5,atol=1e-6)


def test_checkpointed_and_ordinary_real_gradients_agree():
    torch.manual_seed(17);first=RoutedCodeResidual(TinyMultimodalLM(config()),width=24)
    second=RoutedCodeResidual(copy.deepcopy(first.base),width=24);second.load_adapter_state(first.adapter_state())
    first.train();second.train();second.code.gradient_checkpointing=True;ids=torch.tensor([[1,7,9,11,2]])
    first(ids,labels=ids,route='code')['loss'].backward();second(ids,labels=ids,route='code')['loss'].backward()
    for a,b in zip(first.residuals.parameters(),second.residuals.parameters()):
        assert a.grad is not None and b.grad is not None
        torch.testing.assert_close(a.grad,b.grad,rtol=1e-5,atol=1e-7)


def test_adapter_only_roundtrip_and_incomplete_or_nonfinite_states_reject():
    first,ids,pixels=trained();second=RoutedCodeResidual(copy.deepcopy(first.base),width=24).eval()
    exported=first.adapter_state();second.load_adapter_state(exported)
    torch.testing.assert_close(first(ids,route='code')['logits'],second(ids,route='code')['logits'],rtol=0,atol=0)
    wrong=dict(exported);wrong.pop(next(iter(wrong)))
    with pytest.raises(ValueError,match='inventory'):second.load_adapter_state(wrong)
    wrong={k:v.clone() for k,v in exported.items()};next(iter(wrong.values())).flatten()[0]=float('nan')
    with pytest.raises(ValueError,match='finite'):second.load_adapter_state(wrong)
    with pytest.raises(ValueError,match='explicitly'):second(ids,route='unknown')


def test_full_production_unique_parameter_budget_and_oversize_rejection_before_allocation():
    with torch.device('meta'):
        base=TinyMultimodalLM(ModelConfig());routed=RoutedCodeResidual(base,width=2048)
        assert routed.parameter_counts()==dict(total_unique=857007616,frozen_base=706012672,trainable_code=150994944)
        with pytest.raises(ValueError,match='total model limit'):RoutedCodeResidual(base,width=10000)
        with pytest.raises(ValueError):RoutedCodeResidual(base,width=2048,parameter_limit=2*10**9)
