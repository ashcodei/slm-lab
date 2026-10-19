from PIL import Image
import pytest
import torch

from slm_lab.more_real_words import crop_tensor, word_geometry


def test_native_geometry_matches_source_frame_scale_and_fixed_training_margin():
    text, bounds = word_geometry({'utf8_string': ' word ', 'bbox': [20, 30, 40, 20]}, (200, 100), (100, 50))
    assert text == 'word' and bounds == [9, 14, 31, 26]


def test_unreadable_long_tiny_and_wrong_frame_labels_are_explicitly_excluded():
    for text, rectangle, original, native in [('.', [0, 0, 30, 30], (100, 100), (100, 100)),
                                             ('x' * 65, [0, 0, 30, 30], (100, 100), (100, 100)),
                                             ('text', [0, 0, 3, 30], (100, 100), (100, 100)),
                                             ('text', [0, 0, 8, 8], (1000, 1000), (100, 100)),
                                             ('text', [0, 0, 30, 30], (100, 100), (100, 50))]:
        with pytest.raises(ValueError): word_geometry({'utf8_string': text, 'bbox': rectangle}, original, native)


def test_unicode_public_transcripts_and_clipped_source_bounds_are_supported():
    text, bounds = word_geometry({'utf8_string': 'café', 'bbox': [-2, -2, 30, 30]}, (100, 100), (100, 100))
    assert text == 'café' and bounds[:2] == [0, 0]


def test_native_crop_resizing_has_no_transcript_input_and_rejects_outside_geometry():
    image = Image.new('RGB', (100, 100), 'red'); value = crop_tensor(image, [10, 20, 60, 70])
    assert value.shape == (3, 32, 512) and torch.all(value[0] == 1) and torch.all(value[2] == -1)
    with pytest.raises(ValueError): crop_tensor(image, [0, 0, 101, 100])
    with pytest.raises(ValueError): crop_tensor(image, [0., 0, 100, 100])
