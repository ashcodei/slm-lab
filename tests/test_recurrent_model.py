from dataclasses import replace

import pytest
import torch

from slm_lab.model import ModelConfig, TinyMultimodalLM
from slm_lab.recurrent_model import RecurrentMultimodalLM


def config():
    return ModelConfig(vocab_size=260, hidden_size=32, intermediate_size=64, num_layers=4,
        num_heads=4, num_kv_heads=2, max_text_length=64, vision_hidden_size=16,
        vision_intermediate_size=32, vision_layers=1, vision_heads=2, image_size=16, patch_size=4, pool_size=2)


@pytest.mark.parametrize('depth', [1, 2, 4])
@pytest.mark.parametrize('image', [False, True])
def test_zero_gate_transfer_preserves_all_own_base_outputs(depth, image):
    torch.manual_seed(11); base = TinyMultimodalLM(config()).eval()
    recurrent = RecurrentMultimodalLM(config(), recurrences=depth).eval(); recurrent.load_owned_base(base.state_dict())
    ids = torch.randint(0, 260, (2, 12)); pixels = torch.randn(2, 3, 16, 16) if image else None
    first = base(ids, pixels, labels=ids); second = recurrent(ids, pixels, labels=ids)
    for key in ('logits', 'values', 'loss'): torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
    assert set(recurrent.state_dict()) == set(base.state_dict()) | {'recurrence_gates'}


@pytest.mark.parametrize('depth', [2, 4])
@pytest.mark.parametrize('image', [False, True])
def test_nonzero_recurrence_keeps_future_text_invisible(depth, image):
    torch.manual_seed(4); model = RecurrentMultimodalLM(config(), recurrences=depth).eval()
    with torch.no_grad(): model.recurrence_gates.fill_(.3)
    ids = torch.randint(0, 260, (1, 12)); altered = ids.clone(); altered[:, 8:] = 3
    pixels = torch.randn(1, 3, 16, 16) if image else None
    torch.testing.assert_close(model(ids, pixels)['logits'][:, :8], model(altered, pixels)['logits'][:, :8], rtol=0, atol=0)


@pytest.mark.parametrize('depth', [1, 2, 4])
@pytest.mark.parametrize('image', [False, True])
def test_every_shared_core_visit_has_its_own_cache_and_matches_full_decode(depth, image):
    torch.manual_seed(5); model = RecurrentMultimodalLM(config(), recurrences=depth).eval()
    with torch.no_grad(): model.recurrence_gates.fill_(.3)
    ids = torch.randint(0, 260, (1, 12)); pixels = torch.randn(1, 3, 16, 16) if image else None
    state = model.prefill(ids[:, :7], pixels)
    assert len(state['cache'].layers) == model.execution_layers(depth)
    for index in range(7, 12):
        state = model.decode(ids[:, index:index + 1], state['cache'])
        torch.testing.assert_close(state['logits'], model(ids[:, :index + 1], pixels)['logits'][:, -1:], atol=1e-6, rtol=1e-5)
    multi = model.decode(ids[:, 7:10], model.prefill(ids[:, :7], pixels)['cache'])
    torch.testing.assert_close(multi['logits'], model(ids[:, :10], pixels)['logits'][:, 7:10], atol=1e-6, rtol=1e-5)


def test_recurrence_increases_compute_and_cache_without_duplicating_model_parameters():
    with torch.device('meta'):
        base = TinyMultimodalLM(ModelConfig()); model = RecurrentMultimodalLM(ModelConfig())
    assert model.parameter_counts()['total'] == base.parameter_counts()['total'] + 3 < 1_000_000_000
    assert len(model.blocks) == 24 and model.execution_layers(4) == 60
    assert model.lm_head.weight is model.embedding.weight
    assert len(list(model.parameters())) == len(list(base.parameters())) + 1
    small = RecurrentMultimodalLM(config()).eval(); ids = torch.tensor([[1, 2, 3]])
    before = small.prefill(ids, recurrences=1)['cache']; after = small.prefill(ids, recurrences=4)['cache']
    assert after.bytes * 4 == before.bytes * 10


def test_gate_and_shared_backbone_receive_finite_gradients_and_checkpointing_matches():
    torch.manual_seed(7); first = RecurrentMultimodalLM(config(), recurrences=4).train()
    second = RecurrentMultimodalLM(config(), recurrences=4).train(); second.load_state_dict(first.state_dict())
    second.gradient_checkpointing = True; ids = torch.randint(0, 260, (1, 10)); pixels = torch.randn(1, 3, 16, 16)
    first(ids, pixels, labels=ids)['loss'].backward(); second(ids, pixels, labels=ids)['loss'].backward()
    assert first.recurrence_gates.grad.abs().sum() > 0
    assert first.vision.patch.weight.grad.norm() > 0
    for a, b in zip(first.parameters(), second.parameters()):
        if a.grad is not None:
            assert torch.isfinite(a.grad).all(); torch.testing.assert_close(a.grad, b.grad)


def test_nonzero_shared_reasoning_changes_predictions_and_preserves_padding_mask():
    model = RecurrentMultimodalLM(config(), recurrences=4).eval(); ids = torch.randint(0, 260, (2, 12))
    one = model(ids, recurrences=1)['logits']
    with torch.no_grad(): model.recurrence_gates.fill_(.4)
    assert not torch.allclose(one, model(ids)['logits'])
    mask = torch.ones_like(ids); mask[0, 7:] = 0; other = ids.clone(); other[0, 7:] = 9
    torch.testing.assert_close(model(ids, attention_mask=mask)['logits'][0, :7], model(other, attention_mask=mask)['logits'][0, :7])


def test_cache_retains_requested_depth_and_rejects_another_model_or_missing_stage():
    model = RecurrentMultimodalLM(config(), recurrences=4).eval(); ids = torch.tensor([[1, 2]])
    cache = model.prefill(ids)['cache']; model.recurrences = 1
    assert model.decode(torch.tensor([[3]]), cache)['cache'].recurrences == 4
    with pytest.raises(ValueError): model.decode(torch.tensor([[3]]), replace(cache, layers=cache.layers[:-1]))
    with pytest.raises(ValueError): RecurrentMultimodalLM(config()).eval().decode(torch.tensor([[3]]), cache)


@pytest.mark.parametrize('depth', [0, 5, True, 1.5])
def test_explicit_depth_budget_cannot_be_bypassed(depth):
    with pytest.raises(ValueError): RecurrentMultimodalLM(config())(torch.tensor([[1, 2]]), recurrences=depth)


def test_cached_and_full_generation_agree():
    model = RecurrentMultimodalLM(config(), recurrences=4).eval()
    with torch.no_grad(): model.recurrence_gates.fill_(.2)
    ids = torch.tensor([[1, 5]])
    assert torch.equal(model.generate(ids, max_new_tokens=5), model.generate(ids, max_new_tokens=5, use_cache=False))
