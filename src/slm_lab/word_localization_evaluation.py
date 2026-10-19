"""Whole-photo detection assessment with fixed real/blank/wrong-image controls."""
from datetime import datetime, timezone
import json
from pathlib import Path

import torch

from .diverse_images import atomic_json
from .text_localization_metrics import score
from .word_localizer import CONFIDENCE, MAX_PREDICTIONS, NMS_IOU, SIZE, decode, pixels


def aggregate(results):
    totals = {key: sum(row[key] for row in results) for key in ['true_positive', 'false_positive', 'false_negative', 'ignored_predictions', 'legible_targets']}
    tp, fp, fn = totals['true_positive'], totals['false_positive'], totals['false_negative']
    precision = tp / (tp + fp) if tp + fp else 0.
    recall = tp / (tp + fn) if tp + fn else 0.
    totals.update(precision=precision, recall=recall, f1=2 * precision * recall / (precision + recall) if precision + recall else 0.)
    return totals


@torch.inference_mode()
def evaluate(model, rows, output, device='cuda:3'):
    rows = sorted(rows, key=lambda row: row['id'])
    if len(rows) < 2 or len({row['group_id'] for row in rows}) != len(rows): raise ValueError('Whole-photo detection requires distinct validation parents')
    model.eval(); predictions = {}; cases = []
    for index, row in enumerate(rows):
        inputs = pixels(row)[None].to(device)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=str(device).startswith('cuda')): outputs = model(inputs)
        predictions[row['id']] = decode({stride: tensor[0] for stride, tensor in outputs.items()})
        if index % 50 == 0: print('Full-photo detector evaluation', index + 1, len(rows), flush=True)
    # Exactly identical all-black normalized inputs have one deterministic
    # prediction, reused explicitly. They are not gray letterbox-only images.
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=str(device).startswith('cuda')):
        blank_output = model(torch.full((1, 3, SIZE, SIZE), -1., device=device))
    blank_predictions = decode({stride: tensor[0] for stride, tensor in blank_output.items()})
    for index, row in enumerate(rows):
        other = rows[(index + 1) % len(rows)]
        case = {'id': row['id'], 'group_id': row['group_id'], 'source_image_id': row['source_image_id'],
                'stored_image_sha256': row['stored_image_sha256'], 'input_pixels_sha256_768': row['model_pixel_sha256_768'],
                'proper_predictions': predictions[row['id']], 'wrong_image_id': other['id'],
                'wrong_pixels_sha256_768': other['model_pixel_sha256_768'],
                'legible_targets': sum(target['legible'] for target in row['targets'])}
        for control, actual in [('proper', predictions[row['id']]), ('blank', blank_predictions), ('wrong', predictions[other['id']])]:
            case[control + '_scores'] = score([value['box'] for value in actual], row['targets'])
        cases.append(case)
    report = {'created_at': datetime.now(timezone.utc).isoformat(), 'full_photos': len(rows), 'input_size': SIZE,
              'fixed_confidence': CONFIDENCE, 'fixed_nms_iou': NMS_IOU, 'maximum_predictions_per_photo': MAX_PREDICTIONS,
              'totals': {control: aggregate([row[control + '_scores'] for row in cases]) for control in ['proper', 'blank', 'wrong']},
              'blank_predictions': blank_predictions, 'results': cases, 'goal_achieved': False,
              'control_generation': 'All proper full-photo inputs actually inferred. Deterministic blank predictions and each actual counterpart prediction reused for control scoring.',
              'limitations': ['Development photo split; thresholds fixed before learning; not an independent final benchmark.',
                              'Axis-aligned word localization only, not text recognition or native policy question answering.',
                              'Wrong-photo boxes scored against original geometry; counterpart photos are not newly annotated for original targets.',
                              'Explicit maximum-cardinality IoU0.5 and unreadable-area0.5 protocol; not official ICDAR scoring.']}
    atomic_json(Path(output), report)
    return report
