"""Repeat complete-photo reading with our refined reader and the same detector."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

import torch

from .byte_word_reader import ByteWordReader
from .corpus import file_hash
from .diverse_images import atomic_json
from .evaluation_provenance import capture
from .photo_word_reading import infer, original_validation_targets, reading_score
from .word_localization_evaluation import aggregate
from .word_localizer import SIZE, WordLocalizer, load_rows, pixels

ROOT = Path(__file__).resolve().parents[2]


def compare(before, after):
    for field in ['full_photos', 'legible_original_words', 'detector_sha256', 'data_manifest_sha256', 'crop_protocol']:
        if before[field] != after[field]: raise ValueError('Unmatched composed-photo comparison: ' + field)
    if before['source_provenance']['packages'] != after['source_provenance']['packages']: raise ValueError('Unmatched composed-photo inference runtime')
    for path, digest in before['source_provenance']['files_sha256'].items():
        if path not in after['source_provenance']['files_sha256'] or after['source_provenance']['files_sha256'][path] != digest:
            raise ValueError('Original composed-photo implementation changed: ' + path)
    key = lambda row: (row['id'], row['group_id'], row['input_pixels_sha256_768'], row['stored_image_sha256'], row['wrong_image_id'], row['wrong_pixels_sha256_768'])
    if len(before['results']) != before['full_photos'] or len(after['results']) != after['full_photos'] or len({row['id'] for row in after['results']}) != after['full_photos']:
        raise ValueError('Incomplete/duplicate composed-photo comparison cases')
    if [key(row) for row in before['results']] != [key(row) for row in after['results']]: raise ValueError('Original photo/pixel/counterpart identities changed')
    old_targets = set(); new_targets = set()
    for old, new in zip(before['results'], after['results']):
        for control in ['proper', 'blank', 'wrong']:
            if old[control + '_scores']['localization'] != new[control + '_scores']['localization']: raise ValueError('Original detector matching changed')
        select = lambda row: [{key: word[key] for key in ['box', 'score', 'stride', 'crop_pixel_bounds', 'reader_input_pixels_sha256']} for word in row['proper_predictions']]
        if select(old) != select(new): raise ValueError('Original predicted boxes/actual word-crop pixels changed')
        old_targets.update((old['id'], target) for target, _ in old['proper_scores']['matched_target_prediction_indices'])
        new_targets.update((new['id'], target) for target, _ in new['proper_scores']['matched_target_prediction_indices'])
    return {'before_totals': before['totals'], 'after_totals': after['totals'],
            'gained_correct_word_targets': [list(value) for value in sorted(new_targets - old_targets)],
            'lost_previously_correct_word_targets': [list(value) for value in sorted(old_targets - new_targets)],
            'all_case_pixel_box_crop_metric_contracts_matched': True, 'production_promoted': False, 'goal_achieved': False,
            'scope': 'Same400 development photos and detector/crops; only our own reader weights differ. Word target indices refer to original legible annotation order; native-policy question answering still unverified.'}


def evaluate(checkpoint, output, runtime, files, device='cuda:3'):
    checkpoint = Path(checkpoint); checkpoint_sha = file_hash(checkpoint)
    before_path = ROOT / 'reports/photo-word-reading-v1/reading.json'; before_sha = file_hash(before_path)
    before = json.loads(before_path.read_text()); data = ROOT / 'data/text-localization-v1'; data_sha = file_hash(data / 'manifest.json')
    detector_path = ROOT / 'checkpoints/word-localizer-v1/model.pt'; detector_sha = file_hash(detector_path)
    if detector_sha != before['detector_sha256'] or data_sha != before['data_manifest_sha256']: raise ValueError('Original composed detector/input data changed')
    rows, manifest = load_rows(data); rows = sorted(rows['valid'], key=lambda row: row['id'])
    if len(rows) != 400: raise ValueError('Entire original400-photo population is required')
    targets = original_validation_targets(rows, manifest); originals = {row['id']: row for row in before['results']}
    if len(originals) != 400 or sorted(originals) != [row['id'] for row in rows]: raise ValueError('Original composed reading cases are incomplete')
    model = ByteWordReader().to(device); state = torch.load(checkpoint, map_location='cpu', weights_only=True); model.load_state_dict(state['model']); del state
    detector = WordLocalizer().to(device); state = torch.load(detector_path, map_location='cpu', weights_only=True); detector.load_state_dict(state['model']); del state
    words = {}
    for index, row in enumerate(rows):
        actual = infer(pixels(row), detector, model, device)
        if actual['padded_photo_rgb_sha256'] != row['model_pixel_sha256_768']: raise ValueError('Changed actual complete-photo pixels')
        select = lambda values: [{key: value[key] for key in ['box', 'score', 'stride', 'crop_pixel_bounds', 'reader_input_pixels_sha256']} for value in values]
        if select(actual['predicted_words']) != select(originals[row['id']]['proper_predictions']): raise ValueError('Changed actual detector boxes/word-crop pixels')
        words[row['id']] = actual['predicted_words']
        if index % 25 == 0: print('Refined complete-photo reader inferred', index + 1, len(rows), flush=True)
    blank = infer(torch.full((3, SIZE, SIZE), -1.), detector, model, device); cases = []; correct_words = Counter()
    for index, row in enumerate(rows):
        other = rows[(index + 1) % len(rows)]
        case = {'id': row['id'], 'group_id': row['group_id'], 'input_pixels_sha256_768': row['model_pixel_sha256_768'],
                'stored_image_sha256': row['stored_image_sha256'], 'wrong_image_id': other['id'],
                'wrong_pixels_sha256_768': other['model_pixel_sha256_768'], 'proper_predictions': words[row['id']]}
        for control, predictions in [('proper', words[row['id']]), ('blank', blank['predicted_words']), ('wrong', words[other['id']])]:
            case[control + '_scores'] = reading_score(predictions, targets[row['id']])
        valid = [target for target in targets[row['id']] if target['legible']]
        for target, _ in case['proper_scores']['matched_target_prediction_indices']: correct_words[valid[target]['text']] += 1
        cases.append(case)
    if capture(files) != runtime or file_hash(before_path) != before_sha or file_hash(checkpoint) != checkpoint_sha or file_hash(detector_path) != detector_sha or file_hash(data / 'manifest.json') != data_sha:
        raise ValueError('Refined composed evaluation implementation/model/original input changed')
    report = {'completed_at': datetime.now(timezone.utc).isoformat(), 'full_photos': 400,
              'legible_original_words': sum(target['legible'] for items in targets.values() for target in items),
              'detector_sha256': detector_sha, 'reader_sha256': checkpoint_sha, 'data_manifest_sha256': data_sha,
              'baseline_report_sha256': before_sha, 'source_provenance': runtime, 'results': cases,
              'totals': {control: aggregate([row[control + '_scores'] for row in cases]) for control in ['proper', 'blank', 'wrong']},
              'correct_transcript_histogram': dict(correct_words.most_common()), 'blank_predictions': blank['predicted_words'],
              'crop_protocol': blank['crop_protocol'], 'automatic_policy_promotion': False, 'native_policy_question_answering_verified': False,
              'goal_achieved': False, 'scope': 'All original400 complete photos, fixed own detector and predicted-crop transform, proper/blank/actual counterpart controls. Reused development evidence, not independent final native multimodal capability.'}
    comparison = compare(before, report)
    atomic_json(Path(output), report); atomic_json(Path(output).with_name('comparison.json'), comparison)
    return report, comparison
