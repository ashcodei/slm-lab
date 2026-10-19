import copy
import pytest

from slm_lab.refined_photo_evaluation import compare


def reports():
    metric = {'true_positive': 1}
    case = {'id': 'photo', 'group_id': 'parent', 'input_pixels_sha256_768': 'pixels', 'stored_image_sha256': 'bytes',
            'wrong_image_id': 'other', 'wrong_pixels_sha256_768': 'other-pixels',
            'proper_predictions': [{'box': [0, 0, 1, 1], 'score': .9, 'stride': 4, 'crop_pixel_bounds': [0, 0, 768, 768], 'reader_input_pixels_sha256': 'crop'}]}
    for control in ['proper', 'blank', 'wrong']: case[control + '_scores'] = {'localization': metric, 'matched_target_prediction_indices': [[0, 0]] if control == 'proper' else []}
    report = {'full_photos': 1, 'legible_original_words': 2, 'detector_sha256': 'detector', 'data_manifest_sha256': 'data',
              'crop_protocol': 'same', 'source_provenance': {'packages': {'torch': 'same'}, 'files_sha256': {'inference.py': 'same'}},
              'results': [case], 'totals': {}}
    return report, copy.deepcopy(report)


def test_reader_comparison_counts_gained_and_lost_original_word_targets():
    before, after = reports(); after['results'][0]['proper_scores']['matched_target_prediction_indices'] = [[1, 0]]
    result = compare(before, after)
    assert result['gained_correct_word_targets'] == [['photo', 1]]
    assert result['lost_previously_correct_word_targets'] == [['photo', 0]]


@pytest.mark.parametrize('field,value', [('detector_sha256', 'changed'), ('data_manifest_sha256', 'changed'), ('crop_protocol', 'changed')])
def test_other_component_or_protocol_changes_cannot_be_claimed_as_reader_improvement(field, value):
    before, after = reports(); after[field] = value
    with pytest.raises(ValueError): compare(before, after)


def test_changed_photo_crop_or_coverage_is_rejected():
    for field, value in [('input_pixels_sha256_768', 'other'), ('wrong_image_id', 'same')]:
        before, after = reports(); after['results'][0][field] = value
        with pytest.raises(ValueError): compare(before, after)
    before, after = reports(); after['results'][0]['proper_predictions'][0]['reader_input_pixels_sha256'] = 'changed'
    with pytest.raises(ValueError): compare(before, after)
    before, after = reports(); after['results'] = []
    with pytest.raises(ValueError): compare(before, after)


def test_changed_original_inference_code_or_runtime_is_rejected():
    before, after = reports(); after['source_provenance']['files_sha256']['inference.py'] = 'changed'
    with pytest.raises(ValueError): compare(before, after)
    before, after = reports(); after['source_provenance']['packages']['torch'] = 'other'
    with pytest.raises(ValueError): compare(before, after)
