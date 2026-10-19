"""Actual own-small 35-layer graph tests, with CUDA deliberately unavailable."""
import copy
import json
from pathlib import Path

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from slm_lab.deterministic_positions_v1 import install
from slm_lab.frozen_linear_cast_cache_v1 import FrozenLinearCastCache, parameters
from slm_lab.model import ModelConfig
from slm_lab.own_dense_actor_v1 import OwnDenseActor, _FrozenHead, _FrozenLinear
from slm_lab.own_dense_growth_v1 import OwnDenseAssistant, new_backbone


def small():
    torch.set_num_threads(1)
    config = ModelConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_layers=35,
        num_heads=4, num_kv_heads=2, max_text_length=64, vision_hidden_size=8,
        vision_intermediate_size=16, vision_layers=1, vision_heads=2, image_size=16, patch_size=4, pool_size=2)
    base = install(OwnDenseAssistant(new_backbone(config, 988119), reward_hidden=8,
        word_temporal=4, pointing_rank=8, router_hidden=8, detector_hidden=4))
    actor = OwnDenseActor(base, rank=2, seed=988121).eval()
    # Nonzero own adapters avoid a zero-gradient parity test.
    with torch.no_grad():
        for name, p in actor.named_selected(('actor',)):
            if name.endswith('.up'):
                p.copy_(torch.linspace(-.01, .01, p.numel()).reshape(p.shape))
    return actor


class Casts(TorchDispatchMode):
    def __init__(self, ptrs):
        super().__init__()
        self.ptrs = ptrs
        self.count = 0
        self.elements = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        if (func is torch.ops.aten._to_copy.default and isinstance(args[0], torch.Tensor)
                and args[0].data_ptr() in self.ptrs and args[0].dtype == torch.float32
                and kwargs.get('dtype') == torch.bfloat16):
            self.count += 1
            self.elements += args[0].numel()
        return func(*args, **kwargs)


def frozen_ptrs(actor):
    return {p.data_ptr() for m in actor.policy.modules() if type(m) in {_FrozenLinear, _FrozenHead}
        for p in parameters(m) if p is not None}


def test_actual35layer_repeated_conversions_disappear_after_warmup_with_identical_logits_and_inventory(tmp_path):
    actor = small()
    ids = torch.tensor([[1, 5, 7, 8, 2]])
    state = {n: t.clone() for n, t in actor.state_dict().items()}
    old_parameters = [(n, id(p), p.requires_grad) for n, p in actor.named_parameters()]
    with torch.no_grad(), Casts(frozen_ptrs(actor)) as before, torch.autocast('cpu', dtype=torch.bfloat16):
        expected = [actor(ids)['logits'].clone() for _ in range(3)]
    cache = FrozenLinearCastCache(device='cpu', purpose='CPU-mechanics').attach(actor.policy)
    with torch.no_grad(), Casts(frozen_ptrs(actor)) as warm, torch.autocast('cpu', dtype=torch.bfloat16):
        actual = actor(ids)['logits']
    with torch.no_grad(), Casts(frozen_ptrs(actor)) as after, torch.autocast('cpu', dtype=torch.bfloat16):
        following = [actor(ids)['logits'].clone() for _ in range(3)]
    assert all(torch.equal(x, actual) for x in expected + following)
    inventory = cache.inventory()
    assert before.count == 3 * 246 and warm.count == 246 and after.count == 0
    assert inventory['cached_original_parameter_tensors'] == 246 and inventory['actual_conversions'] == 246
    assert [(n, id(p), p.requires_grad) for n, p in actor.named_parameters()] == old_parameters
    assert actor.state_dict().keys() == state.keys()
    assert all(torch.equal(t, state[n]) for n, t in actor.state_dict().items())
    cache.assert_current()
    proof = dict(actual_small_research_parameters=actor.counts()['actual_unique'], actual_layers=35,
        actual_unwarmed_forward_calls=3, actual_uncached_frozen_FP32_to_BF16_conversions=before.count,
        actual_warmup_conversions=warm.count, actual_warmed_forward_calls=3,
        actual_warmed_frozen_FP32_to_BF16_conversions=after.count,
        actual_logits_bit_exact=True, original_parameter_and_state_inventory_unchanged=True,
        **inventory, actual_GPUcontexts=0, measured_GPU_speedup=False, new_accuracy_result=False)
    (tmp_path / 'actual35layer-cast-observation.json').write_text(json.dumps(proof, indent=2))
    cache.close()


def step(actor, optimizer, ids):
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        loss = actor(ids, labels=ids)['loss']
    selected = [p for _, p in actor.named_selected(('actor',))]
    loss.backward(inputs=selected)
    gradients = [p.grad.clone() for p in selected]
    optimizer.step()
    return loss.detach(), gradients


def same_tree(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            same_tree(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            same_tree(x, y)
    else:
        assert a == b


def test_actual35layer_all561actor_gradients_and_two_Adam_updates_and_RNG_are_exact(tmp_path):
    plain, cached = small(), small()
    ids = torch.tensor([[1, 5, 7, 8, 2]])
    optimizers = [torch.optim.AdamW([p for _, p in a.named_selected(('actor',))],
        lr=.001, betas=(.9, .95), eps=1e-5) for a in [plain, cached]]
    cache = FrozenLinearCastCache(device='cpu', purpose='CPU-mechanics').attach(cached.policy)
    rng = torch.get_rng_state().clone()
    for _ in range(2):
        expected = step(plain, optimizers[0], ids)
        actual = step(cached, optimizers[1], ids)
        same_tree(expected, actual)
        same_tree(optimizers[0].state_dict(), optimizers[1].state_dict())
        same_tree(plain.state_dict(), cached.state_dict())
    assert len(optimizers[0].state) == 561
    assert any(bool(g.abs().sum()) for g in actual[1])
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(p.grad is None for p in plain.base.parameters())
    assert all(p.grad is None for p in cached.base.parameters())
    cache.assert_current()
    (tmp_path / 'actual35layer-two-Adam-updates.json').write_text(json.dumps(dict(
        actual_own_small_research_parameters=plain.counts()['actual_unique'], actual_35layers=True,
        actual_Adam_leaves=561, actual_updates=2, all_gradients_weights_Adam_and_RNG_bit_exact=True,
        original_dense_parameters_without_gradients=True, actual_GPUcontexts=0,
        actual_full_size_GPU_equivalence_pending=True), indent=2))
    cache.close()


@pytest.mark.parametrize('bf16', [False, True])
def test_actual35layer_cached_native_prefill_decode_greedy_actions_and_FP32_fallback_exact(bf16):
    actor = small()
    ids = torch.tensor([[1, 5, 7]])
    with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16, enabled=bf16):
        expected = actor.generate(ids, max_new_tokens=7, temperature=0.)
    cache = FrozenLinearCastCache(device='cpu', purpose='CPU-mechanics').attach(actor.policy)
    with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16, enabled=bf16):
        actual = actor.generate(ids, max_new_tokens=7, temperature=0.)
    assert torch.equal(actual, expected)
    assert bool(cache.inventory()['actual_conversions']) == bf16
    cache.close()


@pytest.mark.parametrize('change', ['inplace', 'replacement', 'trainability', 'bias'])
def test_source_change_rejects_before_linear_and_cannot_silently_refresh_a_stale_value(change):
    original = torch.nn.Linear(8, 4)
    layer = _FrozenLinear(original)
    cache = FrozenLinearCastCache(device='cpu', purpose='CPU-mechanics').attach(layer)
    hidden = torch.ones(1, 8)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        layer(hidden)
    conversions = cache.conversions
    if change == 'inplace':
        with torch.no_grad():
            original.weight.add_(1)
    elif change == 'replacement':
        original.weight = torch.nn.Parameter(original.weight.clone())
    elif change == 'trainability':
        original.weight.requires_grad_(False)
    else:
        original.bias = None
    with pytest.raises(ValueError, match='Frozen source'):
        cache.assert_current()
    with pytest.raises(ValueError, match='Frozen source'), torch.autocast('cpu', dtype=torch.bfloat16):
        layer(hidden)
    assert cache.conversions == conversions
    cache.close()


def test_shared_views_cast_each_original_only_once_and_close_restores_actual_forwards():
    original = torch.nn.Linear(8, 4)
    first, second = _FrozenLinear(original), _FrozenLinear(original)
    forwards = [first.forward, second.forward]
    cache = FrozenLinearCastCache(device='cpu', purpose='CPU-mechanics').attach(first).attach(second).attach(first)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        assert torch.equal(first(torch.ones(1, 8)), second(torch.ones(1, 8)))
    assert cache.inventory()['actual_conversions'] == 2
    assert cache.inventory()['patched_detach_only_wrappers'] == 2
    cache.close()
    for module, original in zip([first, second], forwards):
        assert module.forward.__self__ is module
        assert module.forward.__func__ is original.__func__
    assert cache.inventory()['additional_derived_bytes'] == 0


def test_second_cache_and_failed_mixed_device_attach_do_not_partially_patch_any_wrapper():
    model = torch.nn.ModuleList([_FrozenLinear(torch.nn.Linear(8, 4)), _FrozenLinear(torch.nn.Linear(8, 4).double())])
    originals = [m.forward for m in model]
    cache = FrozenLinearCastCache(device='cpu', purpose='CPU-mechanics')
    with pytest.raises(ValueError, match='FP32'):
        cache.attach(model)
    assert all(m.forward == f for m, f in zip(model, originals)) and not cache._patched
    cache.attach(model[0])
    another = FrozenLinearCastCache(device='cpu', purpose='CPU-mechanics')
    with pytest.raises(ValueError, match='existing forward'):
        another.attach(model[0])
    cache.close()


@pytest.mark.parametrize('device,purpose', [('cpu', 'GPU-runtime'), ('cuda', 'GPU-runtime'),
    ('cuda:0', 'CPU-mechanics'), ('cpu', 'unspecified')])
def test_CPU_control_cannot_be_presented_as_a_production_GPU_runtime(device, purpose):
    with pytest.raises(ValueError, match='Explicit'):
        FrozenLinearCastCache(device=device, purpose=purpose)


@pytest.mark.parametrize('bf16_input', [False, True])
def test_actual_biased_linear_hidden_derivatives_match_under_autocast(bf16_input):
    original = torch.nn.Linear(8, 4)
    layer = _FrozenLinear(original)
    hidden = torch.linspace(-1, 1, 24).reshape(3, 8).to(
        torch.bfloat16 if bf16_input else torch.float32).requires_grad_()
    with torch.autocast('cpu', dtype=torch.bfloat16):
        expected = layer(hidden)
        expected.float().square().sum().backward()
    gradient = hidden.grad.clone()
    hidden.grad = None
    cache = FrozenLinearCastCache(device='cpu', purpose='CPU-mechanics').attach(layer)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        actual = layer(hidden)
        actual.float().square().sum().backward()
    assert torch.equal(expected, actual) and torch.equal(gradient, hidden.grad)
    assert all(p.grad is None for p in original.parameters())
    cache.assert_current()
    cache.close()
