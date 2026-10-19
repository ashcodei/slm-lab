import hashlib

import pytest
import torch

from slm_lab.evaluation_provenance import pixels_sha256
from slm_lab.native_greedy_observer_v2 import observe, signature
from test_native_greedy_observer_v1 import model


@pytest.mark.parametrize('mode', ['text', 'positional_image', 'keyword_image'])
def test_real_own_neural_outputs_and_input_images_unchanged(mode):
    torch.manual_seed(314)
    m = model()
    ids = torch.tensor([[1, 4, 5]])
    pixels = torch.randn(1, 3, 16, 16)
    before_pixels = pixels.clone()
    args = (pixels,) if mode == 'positional_image' else ()
    kw = {'pixel_values': pixels} if mode == 'keyword_image' else {}
    kw.update(max_new_tokens=3, eos_token_id=2, use_cache=True)
    original = m.generate
    before = signature(m)
    expected = m.generate(ids, *args, **kw)
    seen = []
    with observe(m, seen.append):
        actual = m.generate(ids, *args, **kw)
    assert m.generate == original and signature(m) == before
    assert torch.equal(expected, actual) and torch.equal(pixels, before_pixels)
    assert seen[0]['actual_generated_token_ids'] == actual[0, 3:].tolist()
    assert seen[0]['original_input_token_ids'] == [1, 4, 5]
    assert seen[0]['complete_input_ids_sha256'] == hashlib.sha256(ids.numpy().astype('<u2').tobytes()).hexdigest()
    assert seen[0]['input_pixels_sha256'] == (None if mode == 'text' else pixels_sha256(pixels))
    assert seen[0]['actual_native_action_count'] == actual.shape[1] - 3
    assert seen[0]['actual_stopped_on_EOS'] == (actual[0, -1].item() == 2)
    assert seen[0]['greedy_native_output_tensor_returned_unchanged']
    assert seen[0]['selected_action_log_probabilities_not_retained']


def test_callback_failure_restores_method_and_nested_capture_rejects():
    m = model()
    original = m.generate
    def fail(event):
        raise RuntimeError('preserved callback failure')
    with pytest.raises(RuntimeError, match='preserved callback failure'):
        with observe(m, fail):
            m.generate(torch.tensor([[1, 4]]), torch.zeros(1, 3, 16, 16), max_new_tokens=2)
    assert m.generate == original and not m._native_greedy_observer_active_v2
    with observe(m, lambda event: None):
        with pytest.raises(ValueError):
            with observe(m, lambda event: None):
                pass


@pytest.mark.parametrize('kwargs', [dict(temperature=.5), dict(use_cache=False), dict(return_rollout=True)])
def test_non_native_profiles_reject_before_forward(kwargs):
    m = model()
    with observe(m, lambda event: None):
        with pytest.raises(ValueError):
            m.generate(torch.tensor([[1, 4]]), max_new_tokens=2, **kwargs)
