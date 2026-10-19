"""Compose our scratch detector and reader, then score complete-photo reading.

Inference accepts pixels and the two own trained components only. Original
human boxes and transcripts are joined separately for evaluation, never used
to crop model inputs or to correct the predicted text.
"""
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .byte_word_reader import HEIGHT, WIDTH, collapse
from .corpus import file_hash
from .ocr_evaluation import normalized, transcription_scores
from .text_localization_data import projected_target
from .text_localization_metrics import area, box, intersection, iou, score as detection_score
from .word_localizer import SIZE, decode

ROOT = Path(__file__).resolve().parents[2]
READER_BATCH_SIZE = 16


def padded_image(pixels):
    if pixels.shape != (3, SIZE, SIZE) or not torch.isfinite(pixels).all() or pixels.min() < -1 or pixels.max() > 1:
        raise ValueError('Full-photo reader expects actual finite normalized768px RGB pixels')
    # Recover the exact prepared RGB bytes. No original dimensions or target
    # annotations are available to this pixels-only inference function.
    array = pixels.detach().cpu().add(1).mul(127.5).round().to(torch.uint8).permute(1, 2, 0).contiguous().numpy()
    return Image.fromarray(array)


def crop_pixels(image, rectangle):
    rectangle = box(rectangle)
    if image.size != (SIZE, SIZE): raise ValueError('Predicted word coordinates require the declared padded photo')
    width, height = rectangle[2] - rectangle[0], rectangle[3] - rectangle[1]
    # Match the fixed5% horizontal/10% vertical margin used when the public
    # training crops were prepared, rather than introducing a tighter crop.
    left, top = max(0, math.floor((rectangle[0] - .05 * width) * SIZE)), max(0, math.floor((rectangle[1] - .1 * height) * SIZE))
    right, bottom = min(SIZE, math.ceil((rectangle[2] + .05 * width) * SIZE)), min(SIZE, math.ceil((rectangle[3] + .1 * height) * SIZE))
    if right <= left or bottom <= top: raise ValueError('Predicted word crop is empty')
    crop = image.crop((left, top, right, bottom)).convert('RGB').resize((WIDTH, HEIGHT), Image.Resampling.BICUBIC)
    array = np.asarray(crop, dtype=np.float32) / 127.5 - 1
    return torch.from_numpy(array.copy()).permute(2, 0, 1), [left, top, right, bottom]


@torch.inference_mode()
def infer(pixels, detector, reader, device='cuda:3'):
    """All crops come from predicted boxes; there is no target/question input."""
    image = padded_image(pixels); detector.eval(); reader.eval()
    amp = lambda: torch.autocast('cuda', dtype=torch.bfloat16) if str(device).startswith('cuda') else nullcontext()
    with amp(): output = detector(pixels[None].to(device))
    predictions = decode({stride: value[0] for stride, value in output.items()})
    words = []
    for start in range(0, len(predictions), READER_BATCH_SIZE):
        batch = predictions[start:start + READER_BATCH_SIZE]
        crops = [crop_pixels(image, prediction['box']) for prediction in batch]
        inputs = torch.stack([value[0] for value in crops]).to(device)
        with amp(): paths = reader(inputs).argmax(-1).cpu().tolist()
        for prediction, (crop, rectangle), path in zip(batch, crops, paths):
            text, valid = collapse(path)
            words.append({**prediction, 'text': text, 'valid_utf8': valid, 'crop_pixel_bounds': rectangle,
                          'reader_input_pixels_sha256': hashlib.sha256(crop.contiguous().numpy().tobytes()).hexdigest()})
    return {'predicted_words': words, 'padded_photo_rgb_sha256': hashlib.sha256(image.tobytes()).hexdigest(),
            'crop_protocol': 'Predicted normalized boxes only; fixed5% horizontal/10% vertical margin matched to training crops, clipped to frame; floor left/top and ceil right/bottom; bicubic resize32x512.'}


def original_validation_targets(rows, manifest):
    """Recover and bind the original text labels for scoring after inference."""
    path = ROOT / 'data/image-source-cache-v1/TextOCR_0.1_val.json'
    if file_hash(path) != manifest['original_annotation_sources']['valid']['sha256']:
        raise ValueError('Original-format validation word/transcript source changed')
    original = json.loads(path.read_text()); result = {}
    for row in rows:
        image_id = row['source_image_id']; image = original['imgs'][image_id]
        if image['set'] != 'val' or row['split'] != 'valid': raise ValueError('Whole-photo evaluation used a non-validation annotation')
        actual_ids = [str(value) for value in original['imgToAnns'][image_id]]
        if actual_ids != [target['original_annotation_id'] for target in row['targets']]:
            raise ValueError('Whole-photo targets omit/reorder original word annotations')
        targets = []
        for target in row['targets']:
            annotation = original['anns'][target['original_annotation_id']]
            if str(annotation['image_id']) != image_id: raise ValueError('Original word/photo identity differs')
            expected = projected_target(annotation, (image['width'], image['height']), row['stored_dimensions'])
            if expected != target: raise ValueError('Original word geometry/legibility differs from prepared detector targets')
            text = annotation.get('utf8_string', '').strip()
            targets.append({**target, 'text': text})
        result[row['id']] = targets
    return result


def reading_score(predictions, targets, threshold=.5, ignore_coverage=.5):
    """A word earns credit once only when both its box and text are correct."""
    if not 0 < threshold <= 1 or not 0 < ignore_coverage <= 1: raise ValueError('Invalid full-photo reading matching protocol')
    rectangles = [box(prediction['box']) for prediction in predictions]
    valid = [target for target in targets if target['legible']]
    ignored = [box(target['normalized_padded_box']) for target in targets if not target['legible']]
    for target in valid:
        if not normalized(target['text']): raise ValueError('Legible original word has an empty transcript')
        box(target['normalized_padded_box'])
    edges = [[index for index, target in enumerate(valid)
              if prediction['valid_utf8'] is True and normalized(prediction['text']) == normalized(target['text'])
              and iou(rectangles[number], target['normalized_padded_box']) >= threshold]
             for number, prediction in enumerate(predictions)]
    assignments = {}
    def augment(prediction, visited):
        for target in edges[prediction]:
            if target in visited: continue
            visited.add(target)
            if target not in assignments or augment(assignments[target], visited):
                assignments[target] = prediction; return True
        return False
    for index in range(len(predictions)): augment(index, set())
    matched = set(assignments.values())
    ignored_predictions = [index for index, rectangle in enumerate(rectangles) if index not in matched
                           and any(intersection(rectangle, target) / area(rectangle) >= ignore_coverage for target in ignored)]
    tp = len(assignments); fp = len(predictions) - tp - len(ignored_predictions); fn = len(valid) - tp
    precision = tp / (tp + fp) if tp + fp else 0.; recall = tp / len(valid) if valid else 0.
    localization = detection_score(rectangles, targets, threshold, ignore_coverage)
    # This secondary diagnostic aligns geometry before examining the text.
    # It is not a transcript of an unambiguous page reading order.
    localized = localization['matched_target_prediction_indices']
    localized_scores = [transcription_scores(valid[target]['text'], predictions[prediction]['text']) for target, prediction in localized]
    return {'true_positive': tp, 'false_positive': fp, 'false_negative': fn, 'ignored_predictions': len(ignored_predictions),
            'legible_targets': len(valid), 'precision': precision, 'recall': recall,
            'f1': 2 * precision * recall / (precision + recall) if precision + recall else 0.,
            'matched_target_prediction_indices': sorted([target, prediction] for target, prediction in assignments.items()),
            'localization': localization, 'localized_word_transcripts': {'matched_boxes': len(localized),
                'exact_words': sum(value['exact_match'] for value in localized_scores),
                'character_edits': sum(value['character_edits'] for value in localized_scores),
                'reference_characters': sum(value['reference_characters'] for value in localized_scores)},
            'predicted_words': len(predictions), 'valid_utf8_predictions': sum(bool(value['valid_utf8']) for value in predictions),
            'protocol': f'One-to-one maximum-cardinality IoU>={threshold} AND case-sensitive NFKC/whitespace-normalized exact transcript; invalid UTF8 gets no exact credit. Unmatched predictions covered>={ignore_coverage} by unreadable regions ignored.'}
