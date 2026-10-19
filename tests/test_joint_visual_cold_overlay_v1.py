"""Actual small independent native recovery; no full-size capacity inference."""
import pytest
import torch

from slm_lab.joint_visual_cold_overlay_v1 import selected, fixed_digest, apply_selected, prove
from test_joint_visual_training_stage_v1 import stage_fixture, fixture, weights, equal_weights


def inputs(batch):
    text = torch.tensor([[1] + [5, 6, 7] * 42 + [2]])
    return dict(text=dict(input_ids=text, attention_mask=torch.ones_like(text, dtype=torch.bool)),
        photo=dict(input_ids=batch['input_ids'][:1], attention_mask=batch['attention_mask'][:1],
            pixel_values=batch['pixel_values'][:1]), word=dict(word_pixels=batch['word_pixels'][:1]))


@pytest.mark.parametrize('bf16', [False, True])
def test_actual_independent_native_plus_selected_values_preserves_every_weight_and_cold_output(bf16):
    stage, batch = stage_fixture(tail=True, bf16=bf16)
    for _ in range(3):
        stage.next_update()
    other, _ = fixture()
    original_fixed = fixed_digest(stage.view.actor.base)
    RNG = torch.get_rng_state().clone()
    result = prove(stage.view, other.actor.base, inputs(batch), mode='mechanics',
        precision='bf16' if bf16 else 'fp32', expected_fixed_sha256=original_fixed)
    assert torch.equal(RNG, torch.get_rng_state())
    assert result['every_original_fixed_and_selected_native_value_equal']
    assert result['actual_text_photo_logits_and_word_readout_bit_exact']
    assert result['independent_native_storage_without_aliasing'] and result['actual_text_context_tokens'] == 128
    assert result['actual_word_readout_dtype'] == 'torch.float32' and result['mechanics_only']
    assert result['final_complete_standalone_export_and_cold_load_still_required']
    assert fixed_digest(other.actor.base) == original_fixed
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('case', ['missing_last_selected_value', 'nonfinite_last_selected_value', 'wrong_fixed_native'])
def test_every_overlay_validates_before_any_cold_native_tensor_is_changed(case):
    stage, _ = stage_fixture(tail=True)
    stage.next_update()
    other, _ = fixture()
    before = weights(other.actor.base)
    values = {name: p.detach().clone() for name, p in selected(stage.view.actor.base).items()}
    expected = fixed_digest(stage.view.actor.base)
    if case == 'missing_last_selected_value':
        del values[next(reversed(values))]
    elif case == 'nonfinite_last_selected_value':
        values[next(reversed(values))].reshape(-1)[-1] = float('nan')
    else:
        expected = 'f' * 64
    with pytest.raises(ValueError):
        apply_selected(other.actor.base, values, expected_fixed_sha256=expected)
    equal_weights(before, other.actor.base)


def test_small_native_cold_proof_cannot_be_presented_as_full4096token_GPU_capacity():
    stage, batch = stage_fixture(tail=True)
    other, _ = fixture()
    before = weights(other.actor.base)
    with pytest.raises(ValueError, match='actual4096text448photo64x448word'):
        prove(stage.view, other.actor.base, inputs(batch), mode='capacity', precision='bf16',
            expected_fixed_sha256=fixed_digest(stage.view.actor.base))
    equal_weights(before, other.actor.base)
    assert not torch.cuda.is_initialized()
