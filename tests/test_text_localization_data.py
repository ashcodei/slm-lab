import pytest
from slm_lab.text_localization_data import projected_target


def test_original_word_box_maps_to_actual_padded_wide_photo_pixels():
    annotation = {'id': 'word-a', 'bbox': [100, 50, 200, 100], 'utf8_string': 'word'}
    target = projected_target(annotation, (1000, 500), (768, 384), 768)
    assert target['normalized_source_box'] == [.1, .1, .3, .3]
    assert target['normalized_padded_box'] == [.1, .3, .3, .4]
    assert target['legible'] and not target['clipped_to_image_bounds']


def test_unreadable_annotation_retains_geometry_without_supplying_transcript():
    annotation = {'id': 'word-a', 'bbox': [0, 0, 10, 10], 'utf8_string': '.'}
    target = projected_target(annotation, (100, 100), (768, 768))
    assert not target['legible'] and target['normalized_source_box'] == [0, 0, .1, .1]
    assert 'transcript' not in target


@pytest.mark.parametrize('box', [[0, 0, 0, 10], [200, 0, 10, 10], [float('nan'), 0, 10, 10]])
def test_invalid_or_outside_original_geometry_is_not_silently_fabricated(box):
    with pytest.raises(ValueError):
        projected_target({'id': 'bad', 'bbox': box, 'utf8_string': 'word'}, (100, 100), (768, 768))
