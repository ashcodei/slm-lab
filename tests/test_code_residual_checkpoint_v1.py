import copy
import random

import numpy as np
import pytest
import torch

from slm_lab.code_residual_branch_v1 import RoutedCodeResidual
from slm_lab.code_residual_checkpoint_v1 import make_training_state, restore_training_state
from slm_lab.codenet_checkpoint_v1 import atomic_checkpoint, capture_rng
from slm_lab.model import ModelConfig, TinyMultimodalLM
from slm_lab.recurrent_ablation import state_digest


BINDINGS = dict(base_checkpoint_sha256='a'*64, tokenizer_sha256='b'*64,
                run_plan_sha256='c'*64, planned_steps=3, device='cpu')


def model_and_optimizer():
    random.seed(17); np.random.seed(18); torch.manual_seed(19)
    config = ModelConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_layers=3,
                         num_heads=4, num_kv_heads=2, max_text_length=32, vision_hidden_size=8,
                         vision_intermediate_size=16, vision_layers=1, vision_heads=2,
                         image_size=8, patch_size=4, pool_size=2)
    model = RoutedCodeResidual(TinyMultimodalLM(config), width=12)
    model.code.gradient_checkpointing = True
    optimizer = torch.optim.AdamW(model.adapter_parameters(), lr=.003, foreach=False)
    return model, optimizer


def update(model, optimizer):
    # Actual stochastic training input depends on all three restored CPU RNGs.
    choice = [random.randrange(8), int(np.random.randint(8)), *torch.randint(0, 8, (2,)).tolist()]
    ids = torch.tensor([[1, *[3+x for x in choice], 2]])
    labels = ids.clone(); labels[:, :3] = -100
    model.train(); optimizer.zero_grad(set_to_none=True)
    loss = model(ids, labels=labels, route='code')['loss']; loss.backward(); optimizer.step()
    model.eval()
    return choice, float(loss.detach())


def prepared():
    model, optimizer = model_and_optimizer()
    history = [update(model, optimizer), update(model, optimizer)]
    state = make_training_state(model, optimizer, **BINDINGS, completed_steps=2, metrics=history)
    return model, optimizer, state


def assert_tree_equal(a, b):
    assert type(a) is type(b)
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a: assert_tree_equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b): assert_tree_equal(x, y)
    else:
        assert a == b


def test_serialized_adam_rng_resume_reproduces_real_next_update_exactly(tmp_path):
    model, optimizer, bundle = prepared()
    original_base = state_digest(model.base.state_dict())
    path = tmp_path/'resume.pt'
    atomic_checkpoint(path, bundle, maximum_bytes=4*1024**2, minimum_free_bytes=1)
    restored = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    expected_draws, expected_loss = update(model, optimizer)
    expected_weights = model.adapter_state()
    expected_adam = copy.deepcopy(optimizer.state_dict())
    expected_rng = capture_rng('cpu')
    # A new process would construct unrelated adapter/optimizer/RNG state first.
    with torch.no_grad():
        for parameter in model.residuals.parameters(): parameter.add_(.8)
    optimizer = torch.optim.AdamW(model.adapter_parameters(), lr=.1, foreach=False)
    random.seed(987); np.random.seed(654); torch.manual_seed(321)
    steps, history = restore_training_state(model, optimizer, restored, **BINDINGS)
    assert steps == 2 and history == bundle['metrics']
    actual_draws, actual_loss = update(model, optimizer)
    assert actual_draws == expected_draws and actual_loss == expected_loss
    assert_tree_equal(model.adapter_state(), expected_weights)
    assert_tree_equal(optimizer.state_dict(), expected_adam)
    assert_tree_equal(capture_rng('cpu'), expected_rng)
    assert state_digest(model.base.state_dict()) == original_base
    assert all(p.grad is None for p in model.base.parameters())


@pytest.mark.parametrize('fault', [
    'base_sha', 'tokenizer_sha', 'plan_sha', 'width', 'config', 'names', 'budget',
    'adapter_missing', 'adapter_nonfinite', 'adam_missing', 'adam_order', 'adam_nonfinite',
    'adam_squared_negative', 'adam_step', 'rng_missing', 'rng_corrupt', 'cuda_rng_on_cpu',
])
def test_mismatched_or_incomplete_state_rejects_before_mutation(fault):
    model, optimizer, state = prepared(); state = copy.deepcopy(state)
    if fault == 'base_sha': state['base_checkpoint_sha256'] = 'd'*64
    elif fault == 'tokenizer_sha': state['tokenizer_sha256'] = 'd'*64
    elif fault == 'plan_sha': state['run_plan_sha256'] = 'd'*64
    elif fault == 'width': state['residual_width'] += 1
    elif fault == 'config': state['config']['num_layers'] += 1
    elif fault == 'names': state['residual_parameter_names'].reverse()
    elif fault == 'budget': state['planned_steps'] += 1
    elif fault == 'adapter_missing': state['residuals'].pop(next(iter(state['residuals'])))
    elif fault == 'adapter_nonfinite': next(iter(state['residuals'].values())).flatten()[0] = float('nan')
    elif fault == 'adam_missing': state['optimizer']['state'].pop(0)
    elif fault == 'adam_order': state['optimizer']['param_groups'][0]['params'].reverse()
    elif fault == 'adam_nonfinite': state['optimizer']['state'][0]['exp_avg'].flatten()[0] = float('nan')
    elif fault == 'adam_squared_negative': state['optimizer']['state'][0]['exp_avg_sq'].flatten()[0] = -1
    elif fault == 'adam_step': state['optimizer']['state'][0]['step'].fill_(1)
    elif fault == 'rng_missing': state['rng'].pop('python')
    elif fault == 'rng_corrupt': state['rng']['torch'] = torch.zeros(3, dtype=torch.uint8)
    elif fault == 'cuda_rng_on_cpu': state['rng']['cuda'] = torch.zeros(3, dtype=torch.uint8)
    before = state_digest(model.residuals.state_dict()); rng_before = capture_rng('cpu')
    adam_before = copy.deepcopy(optimizer.state_dict())
    with pytest.raises(ValueError): restore_training_state(model, optimizer, state, **BINDINGS)
    assert state_digest(model.residuals.state_dict()) == before
    assert_tree_equal(optimizer.state_dict(), adam_before)
    assert_tree_equal(capture_rng('cpu'), rng_before)


def test_actual_frozen_base_difference_rejects_without_changing_adapter_or_rng():
    model, optimizer, state = prepared()
    with torch.no_grad(): model.base.embedding.weight.flatten()[0].add_(.01)
    before = state_digest(model.residuals.state_dict()); rng_before = capture_rng('cpu')
    with pytest.raises(ValueError, match='Every original frozen backbone'):
        restore_training_state(model, optimizer, state, **BINDINGS)
    assert state_digest(model.residuals.state_dict()) == before
    assert_tree_equal(capture_rng('cpu'), rng_before)


def test_restore_invalidates_old_code_cache_and_keeps_original_general_cache_valid():
    model, optimizer, state = prepared(); ids = torch.tensor([[1, 5, 6, 7]])
    model.eval(); code_cache = model.code.prefill(ids[:, :3])['cache']
    general_cache = model.base.prefill(ids[:, :3])['cache']
    restore_training_state(model, optimizer, state, **BINDINGS)
    with pytest.raises(ValueError, match='parameters changed'): model.code.decode(ids[:, 3:], code_cache)
    torch.testing.assert_close(model.base.decode(ids[:, 3:], general_cache)['logits'],
                               model.base(ids)['logits'][:, -1:], rtol=1e-5, atol=1e-6)


def test_partial_optimizer_and_foreign_parameter_groups_cannot_be_saved():
    model, optimizer = model_and_optimizer()
    with pytest.raises(ValueError, match='moment inventory'):
        make_training_state(model, optimizer, **BINDINGS, completed_steps=1, metrics=[])
    optimizer.add_param_group(dict(params=list(model.base.parameters())))
    with pytest.raises(ValueError, match='residual-only'):
        make_training_state(model, optimizer, **BINDINGS, completed_steps=1, metrics=[])
