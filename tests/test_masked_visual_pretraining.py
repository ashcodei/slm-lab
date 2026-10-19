import torch
import pytest

from slm_lab.masked_visual_pretraining import MaskedVisualModel, masked_loss, patchify
from slm_lab.model import ModelConfig, VisionEncoder


def tiny():
    config = ModelConfig(vocab_size=260, hidden_size=32, intermediate_size=64, num_layers=1,
                         num_heads=4, num_kv_heads=2, vision_hidden_size=16, vision_intermediate_size=32,
                         vision_layers=1, vision_heads=2, image_size=16, patch_size=4, pool_size=2)
    return MaskedVisualModel(VisionEncoder(config))


def test_patch_order_and_masked_only_loss_match_known_pixels():
    image = torch.arange(3 * 4 * 4).reshape(1, 3, 4, 4).float()
    patches = patchify(image, 2)
    assert patches.shape == (1, 4, 12)
    assert patches[0, 0].tolist() == [0, 16, 32, 1, 17, 33, 4, 20, 36, 5, 21, 37]
    target = (patches - patches.mean(-1, keepdim=True)) / torch.sqrt(patches.var(-1, unbiased=False, keepdim=True) + 1e-6)
    predictions = target.clone(); predictions[:, 0] += 100
    assert masked_loss(predictions, image, torch.tensor([[0., 1., 1., 1.]]), 2) == 0
    assert masked_loss(predictions, image, torch.tensor([[1., 0., 0., 0.]]), 2) > 1000
    with pytest.raises(ValueError, match='binary'):
        masked_loss(predictions, image, torch.zeros(1, 4), 2)


def test_hidden_pixels_cannot_leak_into_encoder_predictions_and_projector_is_frozen():
    torch.manual_seed(3); model = tiny().eval()
    pixels = torch.randn(2, 3, 16, 16); noise = torch.arange(16).float().expand(2, -1)
    first = model(pixels, noise=noise)
    assert first['visible_indices'].tolist() == [[0, 1, 2, 3], [0, 1, 2, 3]]
    assert first['mask'].sum() == 24
    changed = pixels.clone(); changed[:, :, 4:] += 100
    second = model(changed, noise=noise)
    torch.testing.assert_close(first['predictions'], second['predictions'], rtol=0, atol=0)
    model.train(); result = model(pixels, noise=noise, gradient_checkpointing=True)
    result['loss'].backward()
    for value in [model.vision.patch.weight, model.vision.position, model.mask_token, model.predict.weight]:
        assert value.grad is not None and torch.isfinite(value.grad).all() and value.grad.norm() > 0
    assert model.vision.project.weight.grad is None and not model.vision.project.weight.requires_grad


def test_masked_visual_model_can_learn_a_fixed_reconstruction_batch():
    torch.manual_seed(42); model = tiny().train()
    horizontal = torch.linspace(-1, 1, 16).expand(16, -1)
    vertical = horizontal.T
    pixels = torch.stack([torch.stack([horizontal, vertical, .5 * (horizontal + vertical)]),
                          torch.stack([vertical, horizontal, -.5 * (horizontal + vertical)])])
    noise = torch.rand(2, 16); optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    first = float(model(pixels, noise=noise)['loss'].detach())
    for _ in range(80):
        optimizer.zero_grad(); loss = model(pixels, noise=noise)['loss']; loss.backward(); optimizer.step()
    assert float(model(pixels, noise=noise)['loss'].detach()) < first * .05
    # Structured patterns fit the small decoder's rank; this verifies input-dependent
    # learning mechanics, not natural image semantics or reconstruction of random noise.
