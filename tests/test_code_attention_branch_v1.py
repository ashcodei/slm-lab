import copy
import pytest
import torch
from torch.nn import functional as F

from slm_lab.code_attention_branch_v1 import RoutedCodeAttention, LowRankUpdate, _Projection, restore_attention_training_state
from slm_lab.code_residual_branch_v1 import RoutedCodeResidual
from slm_lab.code_residual_checkpoint_v1 import make_training_state
from slm_lab.model import ModelConfig, TinyMultimodalLM
from slm_lab.recurrent_ablation import state_digest


def config():
    return ModelConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_layers=2,
        num_heads=4, num_kv_heads=2, max_text_length=64, vision_hidden_size=16,
        vision_intermediate_size=32, vision_layers=1, vision_heads=2, image_size=16, patch_size=4, pool_size=2)


@pytest.mark.parametrize('size', [None, 16, 32])
def test_zero_adapters_match_original_all_outputs_for_text_padded_and_image_inputs(size):
    torch.manual_seed(190)
    base = TinyMultimodalLM(config()).eval()
    model = RoutedCodeAttention(base, width=12, rank=3).eval()
    original = copy.deepcopy(base)
    ids = torch.tensor([[1, 7, 9, 11, 2], [1, 5, 2, 0, 0]])
    pixels = None if size is None else torch.randn(2, 3, size, size)
    labels = ids.clone(); labels[:, :2] = -100; labels[ids == 0] = -100
    with torch.no_grad():
        expected = original(ids, pixels, labels=labels, attention_mask=ids != 0, return_hidden=True)
        actual = model(ids, pixels, labels=labels, attention_mask=ids != 0, return_hidden=True, route='code')
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
    assert state_digest(base.state_dict()) == state_digest(original.state_dict())
    assert model.code.embedding.weight is model.code.lm_head.weight is base.embedding.weight
    with torch.device('meta'):
        production = RoutedCodeAttention(TinyMultimodalLM(ModelConfig()))
        control = RoutedCodeResidual(TinyMultimodalLM(ModelConfig()), width=2048)
        assert production.parameter_counts() == control.parameter_counts() == dict(
            total_unique=857007616, frozen_base=706012672, trainable_code=150994944)


def test_projection_matches_independent_merged_matrix_values_and_all_gradients():
    torch.manual_seed(16)
    frozen = torch.nn.Linear(12, 8, bias=False).double().requires_grad_(False)
    update = LowRankUpdate(frozen, rank=3, scale=.7)
    with torch.no_grad(): update.up.weight.normal_(0, .1)
    projection = _Projection(frozen, update)
    x = torch.randn(2, 5, 12, dtype=torch.double, requires_grad=True)
    expected_x = x.detach().clone().requires_grad_()
    a = update.down.weight.detach().clone().requires_grad_()
    b = update.up.weight.detach().clone().requires_grad_()
    expected = F.linear(expected_x, frozen.weight + .7 * (b @ a))
    actual = projection(x)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    weights = torch.randn_like(expected)
    (actual * weights).sum().backward(); (expected * weights).sum().backward()
    for observed, reference in [(x.grad, expected_x.grad), (update.down.weight.grad, a.grad), (update.up.weight.grad, b.grad)]:
        torch.testing.assert_close(observed, reference, rtol=1e-12, atol=1e-12)
    assert frozen.weight.grad is None


def update(model, optimizer):
    ids = torch.tensor([[1, 7, 9, 11, 13, 2]])
    labels = ids.clone(); labels[:, :3] = -100
    model.train(); optimizer.zero_grad(set_to_none=True)
    loss = model(ids, labels=labels, route='code')['loss']
    loss.backward()
    assert all(p.grad is None for p in model.base.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.adapter_parameters())
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    return float(loss.detach())


def test_actual_checkpointed_training_preserves_base_and_cached_text_image_generation():
    torch.manual_seed(19)
    model = RoutedCodeAttention(TinyMultimodalLM(config()), width=12, rank=3)
    model.code.gradient_checkpointing = True
    optimizer = torch.optim.AdamW(model.adapter_parameters(), lr=.01, foreach=False)
    before = state_digest(model.base.state_dict())
    initial = {k: v.clone() for k, v in model.residuals.state_dict().items()}
    for _ in range(3): update(model, optimizer)
    assert state_digest(model.base.state_dict()) == before
    for name, value in model.residuals.state_dict().items():
        assert not torch.equal(value, initial[name]), name
    model.eval(); ids = torch.tensor([[1, 7, 9, 11, 13, 2]])
    for size in [None, 16, 32]:
        pixels = None if size is None else torch.randn(1, 3, size, size)
        state = model.code.prefill(ids[:, :3], pixels)
        for position in range(3, 6):
            state = model.code.decode(ids[:, position:position + 1], state['cache'])
            with torch.no_grad(): expected = model.code(ids[:, :position + 1], pixels)['logits'][:, -1:]
            torch.testing.assert_close(state['logits'], expected, rtol=1e-5, atol=1e-6)
        state = model.code.decode(ids[:, 3:], model.code.prefill(ids[:, :3], pixels)['cache'])
        with torch.no_grad(): expected = model.code(ids, pixels)['logits'][:, 3:]
        torch.testing.assert_close(state['logits'], expected, rtol=1e-5, atol=1e-6)
    old = model.code.prefill(ids[:, :3])
    with torch.no_grad(): model.residuals[0].attention['q_proj'].up.weight.add_(.01)
    with pytest.raises(ValueError, match='parameters changed'): model.code.decode(ids[:, 3:4], old['cache'])
    general = model.base.prefill(ids[:, :3])
    with pytest.raises(ValueError, match='code route'): model.code.decode(ids[:, 3:4], general['cache'])


def test_full_Adam_RNG_restore_reproduces_next_update_and_rejects_architecture_change():
    torch.manual_seed(31)
    first = RoutedCodeAttention(TinyMultimodalLM(config()), width=12, rank=3)
    optimizer = torch.optim.AdamW(first.adapter_parameters(), lr=.01, foreach=False)
    metrics = [update(first, optimizer), update(first, optimizer)]
    args = dict(base_checkpoint_sha256='a'*64, tokenizer_sha256='b'*64, run_plan_sha256='c'*64,
        planned_steps=128, device='cpu')
    saved = make_training_state(first, optimizer, completed_steps=2, metrics=metrics, **args)
    saved['architecture'] = first.architecture()
    continuation = update(first, optimizer)
    expected = make_training_state(first, optimizer, completed_steps=3, metrics=metrics + [continuation], **args)
    second = RoutedCodeAttention(copy.deepcopy(first.base), width=12, rank=3)
    restored_optimizer = torch.optim.AdamW(second.adapter_parameters(), lr=.01, foreach=False)
    restore_attention_training_state(second, restored_optimizer, saved, **args)
    resumed = update(second, restored_optimizer)
    actual = make_training_state(second, restored_optimizer, completed_steps=3, metrics=metrics + [resumed], **args)
    assert resumed == continuation
    assert state_digest(actual['residuals']) == state_digest(expected['residuals'])
    for key in ['optimizer', 'rng']:
        def equal(a, b):
            if isinstance(a, torch.Tensor): return torch.equal(a, b)
            if isinstance(a, dict): return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
            if isinstance(a, (list, tuple)): return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
            return a == b
        assert equal(actual[key], expected[key])
    before = state_digest(second.residuals.state_dict())
    bad = copy.deepcopy(saved); bad['architecture']['attention_scale'] = 2.
    with pytest.raises(ValueError, match='architecture'):
        restore_attention_training_state(second, restored_optimizer, bad, **args)
    assert state_digest(second.residuals.state_dict()) == before
