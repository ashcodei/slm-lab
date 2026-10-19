"""Explicit one-to-one word-box detection scoring with unreadable-region ignores."""
import math


def box(value):
    if len(value) != 4 or any(not math.isfinite(float(x)) for x in value):
        raise ValueError('Detection box must have four finite normalized coordinates')
    result = tuple(map(float, value))
    if any(x < 0 or x > 1 for x in result) or result[2] <= result[0] or result[3] <= result[1]:
        raise ValueError('Detection box is empty/outside normalized photo')
    return result


def area(value):
    return (value[2] - value[0]) * (value[3] - value[1])


def intersection(first, second):
    return max(0., min(first[2], second[2]) - max(first[0], second[0])) * max(0., min(first[3], second[3]) - max(first[1], second[1]))


def iou(first, second):
    first, second = box(first), box(second); common = intersection(first, second)
    return common / (area(first) + area(second) - common)


def score(predictions, targets, threshold=.5, ignore_coverage=.5):
    if not 0 < threshold <= 1 or not 0 < ignore_coverage <= 1:
        raise ValueError('Invalid declared word matching/ignore threshold')
    predictions = [box(value) for value in predictions]
    valid = [box(target['normalized_padded_box']) for target in targets if target['legible']]
    ignored = [box(target['normalized_padded_box']) for target in targets if not target['legible']]
    edges = [[index for index, target in enumerate(valid) if iou(prediction, target) >= threshold] for prediction in predictions]
    # Maximum-cardinality bipartite assignment prevents counting one word twice
    # and resolves overlapping neighboring word boxes without greedy-order bias.
    assignments = {}
    def augment(prediction, visited):
        for target in edges[prediction]:
            if target in visited: continue
            visited.add(target)
            if target not in assignments or augment(assignments[target], visited):
                assignments[target] = prediction; return True
        return False
    for index in range(len(predictions)): augment(index, set())
    matched_predictions = set(assignments.values()); ignored_predictions = []
    for index, prediction in enumerate(predictions):
        if index in matched_predictions: continue
        if any(intersection(prediction, target) / area(prediction) >= ignore_coverage for target in ignored):
            ignored_predictions.append(index)
    true_positive = len(assignments); false_positive = len(predictions) - true_positive - len(ignored_predictions)
    false_negative = len(valid) - true_positive
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.
    recall = true_positive / len(valid) if valid else 0.
    return {'true_positive': true_positive, 'false_positive': false_positive, 'false_negative': false_negative,
            'ignored_predictions': len(ignored_predictions), 'legible_targets': len(valid),
            'precision': precision, 'recall': recall, 'f1': 2 * precision * recall / (precision + recall) if precision + recall else 0.,
            'matched_target_prediction_indices': sorted([target, prediction] for target, prediction in assignments.items()),
            'protocol': f'Fixed IoU>={threshold} maximum-cardinality one-to-one word-box matches; unmatched predictions with>={ignore_coverage} area in unreadable regions ignored. Explicit local protocol, not official ICDAR scoring.'}
