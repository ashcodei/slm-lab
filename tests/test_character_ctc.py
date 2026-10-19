import torch
import pytest
from slm_lab.character_ctc import ALPHABET,CharacterSequenceModel,collapse,ctc_loss,labels,minimum_frames
from slm_lab.model import ModelConfig,VisionEncoder


def test_repeat_labels_require_a_blank_separator_and_extra_time():
    a=labels('a')[0]
    assert collapse([0,a,a,0,a,0])=='aa'
    assert minimum_frames('book')==5
    assert minimum_frames('aaa')==5
    with pytest.raises(ValueError):ctc_loss(torch.randn(1,4,96),['aaa'])


def test_ctc_matches_a_known_alignment_and_has_finite_gradients():
    a,b=labels('ab');logits=torch.full((1,4,96),-10.)
    for t,value in enumerate([a,0,b,0]):logits[0,t,value]=10.
    logits.requires_grad_();loss=ctc_loss(logits,['ab']);loss.backward()
    assert loss<1e-4 and torch.isfinite(logits.grad).all()
    with pytest.raises(ValueError):labels('λ')


def test_sequence_loss_trains_existing_encoder_projector_and_not_prototypes():
    config=ModelConfig(vocab_size=260,hidden_size=32,intermediate_size=64,num_layers=1,num_heads=4,num_kv_heads=2,
                       vision_hidden_size=16,vision_intermediate_size=32,vision_layers=1,vision_heads=2,
                       image_size=16,patch_size=4,pool_size=2)
    vision=VisionEncoder(config);count=sum(p.numel() for p in vision.parameters())
    model=CharacterSequenceModel(vision,torch.randn(96,32),frames=8).train();before=model.prototypes.clone()
    result=model(torch.randn(2,3,16,16));loss=ctc_loss(result,['ab','book']);loss.backward()
    assert sum(p.numel() for p in model.parameters())==count
    assert vision.patch.weight.grad.norm()>0 and vision.project.weight.grad.norm()>0
    assert torch.isfinite(vision.project.weight.grad).all()
    assert model.prototypes.grad is None and torch.equal(model.prototypes,before)
    assert not vision.norm._forward_hooks
