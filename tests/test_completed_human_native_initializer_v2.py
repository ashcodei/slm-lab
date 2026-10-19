"""Actual nullable-image text input matching the full data delivery contract."""
from pathlib import Path

import pytest
import torch

from slm_lab.completed_human_native_initializer_v2 import materialize
from test_completed_human_native_initializer_v1 import completed, inspect, SOURCES
from test_joint_task_gradients_v1 import fixture, word_fixture


def test_actual_text_with_explicit_null_pixels_and_photo_and_FP32_word_all_cold_load(completed, tmp_path):
    actor = completed[0]; verified = inspect(completed)
    photo, _ = fixture(1); _, word, _ = word_fixture()
    code = dict(input_ids=torch.tensor([[1, 4, 7, 2]]), attention_mask=torch.ones(1, 4, dtype=torch.bool), pixel_values=None)
    token = Path(__file__).resolve().parents[1] / 'data/tokenizer-corpus/tokenizer.json'
    if not token.exists():pytest.skip('The original own native tokenizer is local and excluded from Git')
    result = materialize(actor, tmp_path / 'native', token, sources=SOURCES, verified=verified,
        probes=[('nullable_image_text_control', code), ('actual_photo_tensor_control', photo)], word_pixels=word['word_pixels'],
        working_budget_bytes=256 * 1024**2, maximum_checkpoint_bytes=8 * 1024**2,
        minimum_free_bytes=40 * 1024**3, preflight=lambda: None, allow_mechanics=True)
    assert result['all_native_values_and_actual_text_photo_word_cold_outputs_exact']
    assert result['actual_FP32_source_probes'][0]['actual_pixel_shape'] is None
    assert result['actual_FP32_source_probes'][1]['actual_pixel_shape'] == list(photo['pixel_values'].shape)
    assert result['actual_word_dtype'] == 'torch.float32'
    assert result['joint_optimizer_updates'] == 0 and not result['full_size_GPU_capacity_or_training_admitted']
