import math
import pytest
import torch

from slm_lab.word_localizer import WordLocalizer, assignments, decode, encode_targets, loss
from slm_lab.text_localization_metrics import iou


def word(identifier, rectangle, legible=True):
    return {'original_annotation_id': identifier, 'normalized_padded_box': rectangle, 'legible': legible}


def batch(targets):
    return {stride: {key: value[None] for key, value in level.items()} for stride, level in targets.items()}


def test_same_center_words_fall_back_to_unique_levels_without_silent_loss():
    targets = [word(str(index), [.2, .2, .3, .3]) for index in range(3)]
    result, counts = assignments(targets, 64)
    assert len(result) == counts['assigned_legible_words'] == 3
    assert len({(row['stride'], row['x'], row['y']) for row in result}) == 3
    with pytest.raises(ValueError, match='Unrepresented original word'):
        assignments(targets + [word('overflow', [.2, .2, .3, .3])], 64)


def test_annotation_order_does_not_change_learning_targets():
    original = [word('tiny', [.2, .2, .22, .22]), word('larger', [.15, .15, .27, .27]), word('other', [.6, .2, .9, .25])]
    first, second = encode_targets(original, 64), encode_targets(list(reversed(original)), 64)
    for stride in first:
        for key in first[stride]: assert torch.equal(first[stride][key], second[stride][key])


def test_unreadable_regions_ignore_background_but_not_overlapping_legible_words():
    targets = [word('ignore', [.1, .1, .5, .5], False), word('readable', [.2, .2, .3, .3])]
    encoded = encode_targets(targets, 64)
    assert encoded[4]['valid'].sum() < encoded[4]['valid'].numel()
    for level in encoded.values(): assert level['valid'][level['positive']].all()
    assert sum(int(level['positive'].sum()) for level in encoded.values()) == 1


def test_center_size_predictions_decode_without_receiving_original_targets():
    targets = [word('first', [.10, .15, .25, .27]), word('second', [.6, .55, .9, .75])]
    assigned, _ = assignments(targets, 64)
    output = {stride: torch.full((5, 64 // stride, 64 // stride), -20.) for stride in [4, 8, 16]}
    for item in assigned:
        x, y = item['x'], item['y']; tensor = output[item['stride']]
        tensor[0, y, x] = 10
        tensor[1:3, y, x] = torch.tensor([math.log(max(1e-6, min(1 - 1e-6, value)) / (1 - max(1e-6, min(1 - 1e-6, value)))) for value in item['offset']])
        tensor[3:5, y, x] = torch.tensor(item['log_size'])
    predictions = decode(output, 64)
    assert len(predictions) == 2
    for target in targets: assert max(iou(target['normalized_padded_box'], prediction['box']) for prediction in predictions) > .9999


def test_duplicate_cross_level_boxes_are_suppressed_and_nonfinite_predictions_rejected():
    output = {stride: torch.full((5, 64 // stride, 64 // stride), -20.) for stride in [4, 8, 16]}
    for stride in [4, 8]:
        tensor = output[stride]; cell = 32 // stride
        tensor[:, cell, cell] = torch.tensor([10., -20., -20., math.log(16 / stride), math.log(16 / stride)])
    assert len(decode(output, 64)) == 1
    output[16][0, 0, 0] = float('nan')
    with pytest.raises(ValueError, match='Nonfinite'): decode(output, 64)


def test_scratch_detector_gets_finite_center_and_geometry_gradients():
    torch.manual_seed(42); model = WordLocalizer()
    encoded = batch(encode_targets([word('readable', [.2, .2, .7, .4])], 64))
    value, parts = loss(model(torch.rand(1, 3, 64, 64)), encoded)
    assert torch.isfinite(value) and parts['center_focal'] > 0 and parts['box_geometry'] > 0
    value.backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    assert model.heads[0][-1].weight.grad[0].abs().sum() > 0
    assert model.heads[0][-1].weight.grad[1:].abs().sum() > 0


def test_no_legible_words_has_finite_background_only_loss():
    model = WordLocalizer(); encoded = batch(encode_targets([word('ignore', [.1, .1, .5, .5], False)], 64))
    value, parts = loss(model(torch.zeros(1, 3, 64, 64)), encoded)
    assert torch.isfinite(value) and parts['box_geometry'] == 0
