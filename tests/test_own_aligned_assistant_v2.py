import copy

import pytest
import torch

from slm_lab.own_aligned_assistant_v1 import OwnAlignedAssistant, byte_ctc_loss
from slm_lab.own_aligned_assistant_v2 import OwnAlignedAssistantV2
from slm_lab.own_alignment_checkpoint_v1 import make_state, restore_state
from test_own_aligned_assistant_v1 import fixture, original_signature
from test_own_alignment_checkpoint_v1 import update, GROUPS, SOURCES, assert_nested_equal


def test_whole_64_repeated_bytes_fit_real_time_axis_without_extra_parameters():
    original = fixture(); signature = original_signature(original)
    old = OwnAlignedAssistant(original, rank=2, reward_hidden=8, word_temporal=4).eval()
    model = OwnAlignedAssistantV2(original, rank=2, reward_hidden=8, word_temporal=4).eval()
    assert model.counts() == old.counts()
    assert signature == original_signature(original)
    # 28 actual patch columns become140readout steps. Even64identicalbytes
    # need127steps and must remain complete, with finite actual gradients.
    patches = torch.randn(1, 2, 28, 8)
    logits = model.parts['word'](patches)
    assert logits.shape == (1, 140, 257)
    loss = byte_ctc_loss(logits, ['a'*64]); loss.backward()
    assert bool(torch.isfinite(loss))
    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.extension_parameters('word'))
    with pytest.raises(ValueError, match='time axis'):
        byte_ctc_loss(old.parts['word'](patches), ['a'*64])
    assert all(p.grad is None for p in original.parameters())


def test_v2_actual_update_recovery_and_versioned_reload_reject_v1_computation():
    model = OwnAlignedAssistantV2(fixture(), rank=2, reward_hidden=8, word_temporal=4).eval()
    optimizer = torch.optim.AdamW(model.extension_parameters(*GROUPS), lr=1e-3, foreach=False)
    first = update(model, optimizer)
    state = make_state(model, optimizer, groups=GROUPS, sources=SOURCES, completed_steps=1, planned_steps=3,
        sampler_state=dict(next_source_position=1), metrics=[first], device='cpu')
    expected = update(model, optimizer); weights = model.extension_state(); adam = copy.deepcopy(optimizer.state_dict())
    restore_state(model, optimizer, state, groups=GROUPS, expected_sources=SOURCES, planned_steps=3,
        expected_sampler_state=dict(next_source_position=1), device='cpu')
    assert update(model, optimizer) == expected
    assert_nested_equal(weights, model.extension_state()); assert_nested_equal(adam, optimizer.state_dict())
    payload = model.export(SOURCES)
    cold = OwnAlignedAssistantV2(model.original, rank=2, reward_hidden=8, word_temporal=4).eval()
    cold.import_export(payload, expected_sources=SOURCES)
    pixels = torch.randn(1, 3, 8, 32)
    with torch.no_grad():
        assert torch.equal(model.word_logits(pixels), cold.word_logits(pixels))
    old = OwnAlignedAssistant(model.original, rank=2, reward_hidden=8, word_temporal=4).eval()
    with pytest.raises(ValueError):
        old.import_export(payload, expected_sources=SOURCES)
    with pytest.raises(ValueError, match='v2 computation'):
        cold.import_export(old.export(SOURCES), expected_sources=SOURCES)
    wrong = OwnAlignedAssistantV2(model.original, rank=2, reward_hidden=8, word_temporal=4, word_frame_expansion=4).eval()
    with pytest.raises(ValueError):
        wrong.import_export(payload, expected_sources=SOURCES)
