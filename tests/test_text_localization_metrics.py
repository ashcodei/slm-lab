import pytest
from slm_lab.text_localization_metrics import score


def target(rectangle, legible=True):
    return {'normalized_padded_box': rectangle, 'legible': legible}


def test_duplicate_detections_cannot_count_one_original_word_twice():
    rectangle = [.1, .1, .4, .4]
    result = score([rectangle, rectangle], [target(rectangle)])
    assert result['true_positive'] == 1 and result['false_positive'] == 1 and result['precision'] == .5


def test_neighboring_word_assignment_uses_two_distinct_valid_matches():
    truth = [target([.1, .1, .5, .5]), target([.26, .1, .66, .5])]
    result = score([[.14, .1, .54, .5], [.02, .1, .42, .5]], truth)
    assert result['true_positive'] == 2 and result['false_negative'] == 0


def test_unreadable_original_regions_are_ignored_without_hiding_valid_word_misses():
    truth = [target([.1, .1, .3, .3]), target([.6, .6, .9, .9], False)]
    result = score([[.65, .65, .8, .8]], truth)
    assert result['ignored_predictions'] == 1 and result['false_positive'] == 0
    assert result['false_negative'] == 1 and result['true_positive'] == 0


def test_invalid_prediction_geometry_cannot_get_detection_credit():
    with pytest.raises(ValueError): score([[0, 0, float('nan'), 1]], [])
