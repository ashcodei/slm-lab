import pytest
import torch

from slm_lab.model import ModelConfig
from slm_lab.recurrent_model import RecurrentMultimodalLM
from slm_lab.recurrent_ablation import budgets, frozen_digest, language_parameters, state_digest, window_order


def model():
    return RecurrentMultimodalLM(ModelConfig(vocab_size=32, hidden_size=16,
        intermediate_size=32, num_layers=4, num_heads=2, num_kv_heads=1,
        max_text_length=32, vision_hidden_size=8, vision_intermediate_size=16,
        vision_layers=1, vision_heads=2, image_size=16, patch_size=4, pool_size=2), recurrences=4)


def test_compute_proxy_is_exact_and_separate_from_target_exposure():
    with torch.device('meta'): value = RecurrentMultimodalLM(ModelConfig())
    budget = budgets(value, 256, 4, 512)
    assert budget['equal_layer_control_steps'] == 640
    assert budget['recurrent_target_exposures'] == 130816
    assert 640 * value.execution_layers(1) * 512 == budget['matched_decoder_layer_token_executions']
    assert budget['equal_layer_control_target_exposures'] > budget['recurrent_target_exposures']


def test_comparison_uses_unique_shared_prefix_source_windows():
    long = window_order(1000, 640, 4); short = window_order(1000, 256, 4)
    assert long[:256] == short and len(set(long)) == 640
    with pytest.raises(ValueError): window_order(8, 9, 4)


@pytest.mark.parametrize('train_gates', [False, True])
def test_full_language_update_changes_backbone_and_keeps_vision_and_critic_exact(train_gates):
    torch.manual_seed(5); value = model()
    frozen = frozen_digest(value); original = state_digest(value.blocks[0].state_dict())
    parameters = language_parameters(value, train_gates)
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    value.gradient_checkpointing = True
    optimizer = torch.optim.AdamW(parameters, lr=.001)
    value(ids, labels=ids, recurrences=4 if train_gates else 1)['loss'].backward()
    assert value.blocks[1].attn.q_proj.weight.grad.abs().sum() > 0
    optimizer.step()
    assert frozen_digest(value) == frozen
    assert state_digest(value.blocks[0].state_dict()) != original
    assert (value.recurrence_gates.grad is not None) == train_gates
    assert not value.vision.patch.weight.requires_grad and not value.value_head.weight.requires_grad


def test_nonintegral_compute_budget_is_rejected():
    with pytest.raises(ValueError): budgets(model(), 1, 2, 8)


def test_state_identity_detects_values_and_tensor_shapes():
    first = {'a': torch.tensor([1., 2.])}
    assert state_digest(first) != state_digest({'a': torch.tensor([1., 3.])})
    assert state_digest(first) != state_digest({'a': torch.tensor([[1., 2.]])})
