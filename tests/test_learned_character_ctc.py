import torch
from slm_lab.character_ctc import ctc_loss
from slm_lab.learned_character_ctc import LearnedCharacterSequenceModel
from slm_lab.model import ModelConfig,VisionEncoder


def tiny():
    config=ModelConfig(vocab_size=260,hidden_size=32,intermediate_size=64,num_layers=1,num_heads=4,num_kv_heads=2,
                       vision_hidden_size=16,vision_intermediate_size=32,vision_layers=1,vision_heads=2,
                       image_size=16,patch_size=4,pool_size=2)
    return LearnedCharacterSequenceModel(VisionEncoder(config),frames=8)


def test_learned_readout_receives_gradients_and_keeps_unused_projector_frozen():
    model=tiny().train();loss=ctc_loss(model(torch.randn(2,3,16,16)),['ab','book']);loss.backward()
    for parameter in [model.height_attention.weight,model.classifier.weight,model.vision.patch.weight]:
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.norm()>0
    assert model.vision.project.weight.grad is None and not model.vision.project.weight.requires_grad
    assert not model.vision.norm._forward_hooks


def test_head_can_fit_a_fixed_feature_batch_before_training_the_encoder():
    torch.manual_seed(42);model=tiny().train();model.vision.requires_grad_(False)
    features=torch.randn(2,4,4,16);targets=['ab','ba']
    optimizer=torch.optim.Adam(model.parameters(),lr=.02)
    first=float(ctc_loss(model.readout(features),targets).detach())
    for _ in range(100):
        optimizer.zero_grad();loss=ctc_loss(model.readout(features),targets);loss.backward();optimizer.step()
    last=float(ctc_loss(model.readout(features),targets).detach())
    assert last<first*.1
    assert all(parameter.grad is None for parameter in model.vision.parameters())
