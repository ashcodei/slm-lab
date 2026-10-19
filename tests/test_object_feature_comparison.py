import pytest
from scripts.assess_masked_visual_features import paired_difference


def cases(prediction):
    return [{'id': str(index), 'group_id': 'photo-a' if index < 2 else 'photo-b',
             'expected': 'cat', 'predicted': prediction, 'input_pixels_sha256': 'pixels-' + str(index),
             'wrong_image_id': 'other-' + str(index), 'wrong_pixels_sha256': 'other-pixels-' + str(index)}
            for index in range(3)]


def test_perfect_paired_gain_and_no_change_are_bounded_by_photo_group_bootstrap():
    result = paired_difference(cases('dog'), cases('cat'))
    assert result['photo_groups'] == 2 and result['accuracy_change'] == 1
    assert result['photo_group_bootstrap_95_interval'] == [1, 1]
    same = paired_difference(cases('cat'), cases('cat'))
    assert same['accuracy_change'] == 0 and same['photo_group_bootstrap_95_interval'] == [0, 0]


def test_paired_comparison_rejects_changed_cases_or_image_controls():
    changed = cases('cat'); changed[0]['wrong_pixels_sha256'] = 'different-control'
    with pytest.raises(ValueError, match='identities differ'):
        paired_difference(cases('dog'), changed)
