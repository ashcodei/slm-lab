import torch

from slm_lab import word_localization_evaluation as evaluation


def test_detection_aggregate_uses_word_counts_instead_of_averaging_photo_scores():
    rows = [{'true_positive': 1, 'false_positive': 0, 'false_negative': 0, 'ignored_predictions': 2, 'legible_targets': 1},
            {'true_positive': 0, 'false_positive': 1, 'false_negative': 9, 'ignored_predictions': 0, 'legible_targets': 9}]
    result = evaluation.aggregate(rows)
    assert result['recall'] == .1 and result['precision'] == .5
    assert result['ignored_predictions'] == 2 and result['legible_targets'] == 10


def test_whole_photo_controls_reuse_actual_counterpart_predictions_and_keep_original_targets(tmp_path, monkeypatch):
    inputs = []; decoded = [[{'box': [.1, .1, .4, .4], 'score': .9}], [{'box': [.6, .6, .9, .9], 'score': .9}], []]
    class OnlyPixels:
        def eval(self): return self
        def __call__(self, value):
            inputs.append(value.clone()); return {4: torch.zeros(1, 5, 1, 1)}
    monkeypatch.setattr(evaluation, 'pixels', lambda row: torch.full((3, 8, 8), float(row['source_image_id'])))
    monkeypatch.setattr(evaluation, 'decode', lambda output: decoded.pop(0))
    rows = [{'id': str(index), 'group_id': 'parent' + str(index), 'source_image_id': index,
             'stored_image_sha256': str(index), 'model_pixel_sha256_768': str(index),
             'targets': [{'legible': True, 'normalized_padded_box': rectangle}]} for index, rectangle in
            enumerate([[.1, .1, .4, .4], [.6, .6, .9, .9]])]
    report = evaluation.evaluate(OnlyPixels(), rows, tmp_path / 'reading.json', 'cpu')
    assert len(inputs) == 3 and torch.all(inputs[-1] == -1)
    assert report['totals']['proper']['true_positive'] == 2
    assert report['totals']['blank']['true_positive'] == report['totals']['wrong']['true_positive'] == 0
    assert report['results'][0]['wrong_image_id'] == '1' and report['results'][1]['wrong_image_id'] == '0'
