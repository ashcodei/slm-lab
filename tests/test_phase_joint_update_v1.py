import copy

import pytest
import torch

from slm_lab.fast_joint_stage_v1 import make_optimizer,parameter_groups
from slm_lab.model import ModelConfig
from slm_lab.own_dense_growth_v1 import OwnDenseAssistant,new_backbone
from slm_lab.phase_joint_update_v1 import backward_phase


def fixture():
    torch.set_num_threads(1)
    c=ModelConfig(vocab_size=96,hidden_size=32,intermediate_size=64,num_layers=2,num_heads=4,num_kv_heads=2,
        max_text_length=64,vision_hidden_size=8,vision_intermediate_size=16,vision_layers=1,vision_heads=2,image_size=16,patch_size=4,pool_size=2)
    model=OwnDenseAssistant(new_backbone(c,958219),reward_hidden=8,word_temporal=4,pointing_rank=8,router_hidden=8,detector_hidden=4)
    optimizer=make_optimizer(model,dict(learning_rates=dict(language=.001,vision=.001,word=.001),betas=[.9,.95],weight_decay=.1))
    batch=dict(input_ids=torch.tensor([[1,11,12,14,2]]),labels=torch.tensor([[1,11,12,14,2]]),attention_mask=torch.ones(1,5,dtype=torch.bool))
    return model,optimizer,batch


def test_raw_pretraining_uses_every_next_token_and_no_reference_or_vision_updates():
    model,opt,batch=fixture();before={n:p.detach().clone() for n,p in model.named_parameters()}
    with torch.no_grad():expected=model(**batch)['loss'].item()*4
    report=backward_phase(model,batch,phase='foundation',kind='text',global_targets=4,global_context_targets=4)
    assert report['CE_sum']==pytest.approx(expected) and report['reference_parameters']==0
    opt.step();groups=parameter_groups(model)
    assert any(not torch.equal(p,before[n]) for n,p in groups['language'])
    assert all(torch.equal(p,before[n]) and p.grad is None for g in ['vision','word'] for n,p in groups[g])
    with pytest.raises(ValueError,match='no reference'):
        backward_phase(model,batch,phase='foundation',kind='text',global_targets=4,global_context_targets=4,reference=copy.deepcopy(model))


def test_later_text_anchors_to_actual_trained_own_foundation_and_images_keep_decoder_exact():
    model,opt,batch=fixture()
    backward_phase(model,batch,phase='foundation',kind='text',global_targets=4,global_context_targets=4);opt.step();opt.zero_grad(set_to_none=True)
    reference=copy.deepcopy(model).eval().requires_grad_(False)
    # Move the policy after taking its own foundation snapshot.
    with torch.no_grad():model.network.embedding.weight.add_(.01)
    report=backward_phase(model,batch,phase='instruction_and_vision',kind='text',global_targets=4,global_context_targets=4,reference=reference)
    assert report['full_context_reference_KL_sum']>0 and all(p.grad is None for p in reference.parameters())
    opt.step();opt.zero_grad(set_to_none=True);before={n:p.detach().clone() for n,p in parameter_groups(model)['language']}
    batch['pixel_values']=torch.randn(1,3,16,16)
    backward_phase(model,batch,phase='instruction_and_vision',kind='image',global_targets=4,global_context_targets=4,reference=reference)
    opt.step()
    assert all(torch.equal(p,before[n]) and p.grad is None for n,p in parameter_groups(model)['language'])
