import numpy as np
from PIL import Image
import pytest
import torch

from slm_lab import photo_word_reading as reading


def target(text, rectangle=(.1, .1, .4, .4), legible=True):
    return {'text': text, 'normalized_padded_box': list(rectangle), 'legible': legible}


def prediction(text, rectangle=(.1, .1, .4, .4), valid=True):
    return {'text': text, 'box': list(rectangle), 'valid_utf8': valid}


def test_correct_transcript_in_wrong_place_or_wrong_transcript_in_right_place_gets_no_word_credit():
    assert reading.reading_score([prediction('cat', (.6, .6, .9, .9))], [target('cat')])['true_positive'] == 0
    result = reading.reading_score([prediction('dog')], [target('cat')])
    assert result['true_positive'] == 0 and result['false_positive'] == result['false_negative'] == 1
    assert result['localization']['true_positive'] == 1


def test_duplicate_words_only_get_credit_once_and_each_distinct_annotation_can_match():
    result = reading.reading_score([prediction('cat'), prediction('cat')], [target('cat')])
    assert result['true_positive'] == 1 and result['false_positive'] == 1
    result = reading.reading_score([prediction('cat'), prediction('dog')], [target('cat'), target('dog')])
    assert result['true_positive'] == 2


def test_exact_text_matching_does_not_follow_greedy_geometry_assignment():
    targets = [target('cat', (.1, .1, .5, .5)), target('dog', (.26, .1, .66, .5))]
    result = reading.reading_score([prediction('dog', (.14, .1, .54, .5)), prediction('cat', (.02, .1, .42, .5))], targets)
    assert result['true_positive'] == 2


def test_normalization_is_unicode_whitespace_only_and_invalid_utf8_cannot_match():
    assert reading.reading_score([prediction(' café ')], [target('cafe\u0301')])['true_positive'] == 1
    assert reading.reading_score([prediction('CAT')], [target('cat')])['true_positive'] == 0
    assert reading.reading_score([prediction('cat', valid=False)], [target('cat')])['true_positive'] == 0


def test_ignore_regions_do_not_hide_missing_legible_words():
    result = reading.reading_score([prediction('nonsense', (.7, .7, .8, .8))], [target('cat'), target('.', (.6, .6, .9, .9), False)])
    assert result['ignored_predictions'] == 1 and result['false_negative'] == 1 and result['false_positive'] == 0


def test_normalized_photo_round_trip_preserves_every_rgb_byte():
    array = np.arange(768 * 768 * 3, dtype=np.int64).reshape(768, 768, 3).astype(np.uint8)
    pixels = torch.from_numpy(array.copy()).permute(2, 0, 1).float().div(127.5).sub(1)
    assert reading.padded_image(pixels).tobytes() == array.tobytes()
    pixels[0, 0, 0] = float('nan')
    with pytest.raises(ValueError): reading.padded_image(pixels)


def test_crops_use_only_predicted_pixel_bounds_and_resize_without_word_length():
    image = Image.new('RGB', (768, 768), 'red'); image.paste('blue', (384, 0, 768, 768))
    left, bounds = reading.crop_pixels(image, [.1, .1, .2, .2]); right, _ = reading.crop_pixels(image, [.7, .1, .8, .2])
    assert bounds == [72, 69, 158, 162]
    assert left.shape == right.shape == (3, 32, 512)
    assert torch.all(left[0] == 1) and torch.all(left[2] == -1)
    assert torch.all(right[2] == 1) and torch.all(right[0] == -1)


def test_pixels_only_pipeline_uses_detector_boxes_and_actual_reader_output(monkeypatch):
    seen = []
    class Detector:
        def eval(self): return self
        def __call__(self, pixels):
            assert pixels.shape == (1, 3, 768, 768)
            return {4: torch.zeros(1, 5, 1, 1)}
    class Reader:
        def eval(self): return self
        def __call__(self, pixels):
            seen.append(pixels.clone()); output = torch.zeros(len(pixels), 128, 257)
            output[:, :, 0] = 1
            for index, character in enumerate(b'OWN'): output[:, index * 2, character + 1] = 5
            return output
    monkeypatch.setattr(reading, 'decode', lambda output: [{'box': [.1, .1, .2, .2], 'score': .9, 'stride': 4}])
    result = reading.infer(torch.full((3, 768, 768), -1.), Detector(), Reader(), 'cpu')
    assert result['predicted_words'][0]['text'] == 'OWN' and result['predicted_words'][0]['valid_utf8']
    assert len(seen) == 1 and seen[0].shape == (1, 3, 32, 512) and torch.all(seen[0] == -1)
