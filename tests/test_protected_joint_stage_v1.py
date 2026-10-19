import copy
import torch

from slm_lab.fast_joint_stage_v1 import make_optimizer,parameter_groups
from slm_lab.model import ModelConfig
from slm_lab.own_dense_growth_v1 import OwnDenseAssistant,new_backbone
from slm_lab.protected_joint_stage_v1 import backward_protected


def fixture():
    torch.manual_seed(957421)
    config=ModelConfig(vocab_size=96,hidden_size=32,intermediate_size=64,num_layers=2,num_heads=4,num_kv_heads=2,
        max_text_length=64,vision_hidden_size=8,vision_intermediate_size=16,vision_layers=1,vision_heads=2,
        image_size=16,patch_size=4,pool_size=2)
    model=OwnDenseAssistant(new_backbone(config,957421),reward_hidden=8,word_temporal=4,pointing_rank=8,router_hidden=8,detector_hidden=4)
    settings=dict(learning_rates=dict(language=.001,vision=.001,word=.001),betas=[.9,.95],weight_decay=.1)
    optimizer=make_optimizer(model,settings)
    batch=dict(input_ids=torch.tensor([[1,11,12,14,2]]),labels=torch.tensor([[-100,-100,12,14,2]]),attention_mask=torch.ones(1,5,dtype=torch.bool))
    return model,optimizer,batch


def test_visual_updates_cannot_change_language_even_through_Adam_decay_and_restore_flags():
    model,optimizer,batch=fixture();groups=parameter_groups(model);before={n:p.detach().clone() for n,p in model.named_parameters()}
    flags=[p.requires_grad for p in model.parameters()];batch['pixel_values']=torch.randn(1,3,16,16)
    report=backward_protected(model,batch,kind='image',global_targets=3,global_context_targets=4)
    assert report['active_groups']==['vision'] and [p.requires_grad for p in model.parameters()]==flags
    assert all(p.grad is None for n in ['language','word'] for _,p in groups[n])
    optimizer.step()
    assert all(torch.equal(p,before[n]) for n,p in groups['language'])
    assert any(not torch.equal(p,before[n]) for n,p in groups['vision'])


def test_text_anchoring_trains_only_language_and_leaves_reference_bytes_and_metadata_exact():
    model,optimizer,batch=fixture();reference=copy.deepcopy(model).eval().requires_grad_(False)
    with torch.no_grad():reference.network.embedding.weight.add_(.05)
    reference_before={n:p.detach().clone() for n,p in reference.named_parameters()}
    vision_before={n:p.detach().clone() for n,p in parameter_groups(model)['vision']}
    report=backward_protected(model,batch,kind='text',global_targets=3,global_context_targets=4,reference=reference)
    assert report['full_context_reference_KL_sum']>0 and report['active_groups']==['language']
    assert all(p.grad is None and not p.requires_grad and torch.equal(p,reference_before[n]) for n,p in reference.named_parameters())
    optimizer.step()
    assert all(torch.equal(p,vision_before[n]) for n,p in parameter_groups(model)['vision'])
