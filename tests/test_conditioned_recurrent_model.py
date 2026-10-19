import pytest
import torch

from slm_lab.model import ModelConfig, TinyMultimodalLM
from slm_lab.conditioned_recurrent_model import ConditionedRecurrentLM
from slm_lab.recurrent_ablation import frozen_digest, language_parameters, state_digest


def config():
    return ModelConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_layers=4,
        num_heads=4, num_kv_heads=2, max_text_length=64, vision_hidden_size=16,
        vision_intermediate_size=32, vision_layers=1, vision_heads=2,
        image_size=16, patch_size=4, pool_size=2)


@pytest.mark.parametrize('depth', [1, 2, 4])
@pytest.mark.parametrize('image', [False, True])
def test_conditioned_cache_matches_whole_causal_text_and_image_forward(depth, image):
    torch.manual_seed(11); model = ConditionedRecurrentLM(config(), recurrences=depth).eval()
    ids = torch.randint(0, 64, (1, 12)); pixels = torch.randn(1, 3, 16, 16) if image else None
    state = model.prefill(ids[:, :6], pixels)
    assert len(state['cache'].layers) == model.execution_layers(depth)
    for index in range(6, 12):
        state = model.decode(ids[:, index:index + 1], state['cache'])
        torch.testing.assert_close(state['logits'], model(ids[:, :index + 1], pixels)['logits'][:, -1:], rtol=1e-5, atol=1e-6)
    multiple = model.decode(ids[:, 6:9], model.prefill(ids[:, :6], pixels)['cache'])
    torch.testing.assert_close(multiple['logits'], model(ids[:, :9], pixels)['logits'][:, 6:9], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('image', [False, True])
def test_injected_input_never_exposes_future_text(image):
    model = ConditionedRecurrentLM(config(), recurrences=4).eval()
    ids = torch.randint(0, 64, (1, 12)); altered = ids.clone(); altered[:, 7:] = 0
    pixels = torch.randn(1, 3, 16, 16) if image else None
    torch.testing.assert_close(model(ids, pixels)['logits'][:, :7], model(altered, pixels)['logits'][:, :7], rtol=0, atol=0)


def test_exact_own_weight_transfer_and_training_only_norm_calibration():
    torch.manual_seed(4); base = TinyMultimodalLM(config()).eval(); model = ConditionedRecurrentLM(config()).eval()
    model.load_owned_base(base.state_dict())
    evidence = model.calibrate_owned_norms([torch.tensor([[1, 2, 3, 4]])])
    assert evidence['original_training_windows'] == 1
    for name, original in base.state_dict().items():
        torch.testing.assert_close(original, model.state_dict()[name], rtol=0, atol=0)
    with pytest.raises(ValueError): model.load_owned_base({'unrelated': torch.tensor([1])})
    with pytest.raises(ValueError): model.calibrate_owned_norms([])


def test_no_gate_can_ignore_recurring_passes_and_input_adapter_has_both_active_halves():
    model = ConditionedRecurrentLM(config()).eval(); ids = torch.tensor([[1, 2, 3, 4]])
    assert 'recurrence_gates' not in model.state_dict()
    assert not torch.allclose(model(ids, recurrences=1)['logits'], model(ids, recurrences=4)['logits'])
    width = model.config.hidden_size
    assert model.recurrence_adapter.weight[:, :width].abs().sum() > 0
    assert model.recurrence_adapter.weight[:, width:].abs().sum() > 0


def test_shared_backbone_adapter_norms_and_vision_have_finite_matching_checkpointed_gradients():
    torch.manual_seed(17); first = ConditionedRecurrentLM(config(), recurrences=4).train()
    second = ConditionedRecurrentLM(config(), recurrences=4).train(); second.load_state_dict(first.state_dict())
    second.gradient_checkpointing = True
    ids = torch.tensor([[1, 4, 3, 9, 2]]); pixels = torch.randn(1, 3, 16, 16)
    first(ids, pixels, labels=ids)['loss'].backward(); second(ids, pixels, labels=ids)['loss'].backward()
    assert first.recurrence_adapter.weight.grad.abs().sum() > 0
    assert first.recurrence_attn_post[0].weight.grad.abs().sum() > 0
    assert first.recurrence_ffn_post[0].weight.grad.abs().sum() > 0
    assert first.vision.patch.weight.grad.abs().sum() > 0
    for a, b in zip(first.parameters(), second.parameters()):
        if a.grad is not None:
            assert torch.isfinite(a.grad).all(); torch.testing.assert_close(a.grad, b.grad, rtol=1e-5, atol=1e-7)


def test_production_unique_parameter_count_and_all_original_layers_retained():
    with torch.device('meta'): model = ConditionedRecurrentLM(ModelConfig())
    assert model.parameter_counts()['total'] == 710769664 < 1_000_000_000
    assert len(model.blocks) == 24 and model.execution_layers(4) == 60
    assert model.lm_head.weight is model.embedding.weight


def test_padding_masks_and_cached_generation_agree():
    model = ConditionedRecurrentLM(config(), recurrences=4).eval(); ids = torch.tensor([[1, 2, 3, 4, 5]])
    mask = torch.tensor([[1, 1, 1, 0, 0]]); other = ids.clone(); other[:, 3:] = 0
    torch.testing.assert_close(model(ids, attention_mask=mask)['logits'][:, :3],
                               model(other, attention_mask=mask)['logits'][:, :3], rtol=0, atol=0)
    assert torch.equal(model.generate(ids[:, :2], max_new_tokens=5),
                       model.generate(ids[:, :2], max_new_tokens=5, use_cache=False))


def test_each_control_resets_to_same_own_calibrated_initial_policy_and_full_language_learns():
    torch.manual_seed(16); base = TinyMultimodalLM(config()).eval()
    model = ConditionedRecurrentLM(config(), recurrences=4)
    model.load_owned_base(base.state_dict()); ids = torch.tensor([[1, 2, 3, 4]])
    model.calibrate_owned_norms([ids])
    initial = {name: value.clone() for name, value in model.new_state().items()}
    expected = state_digest(model.state_dict()); frozen = frozen_digest(model)
    optimizer = torch.optim.AdamW(language_parameters(model, True), lr=.001)
    model.train(); model(ids, labels=ids)['loss'].backward(); optimizer.step()
    assert state_digest(model.state_dict()) != expected and frozen_digest(model) == frozen
    model.load_owned_base(base.state_dict())
    transfer = model.load_state_dict(initial, strict=False)
    assert not transfer.unexpected_keys and set(transfer.missing_keys) == set(base.state_dict())
    assert state_digest(model.state_dict()) == expected
