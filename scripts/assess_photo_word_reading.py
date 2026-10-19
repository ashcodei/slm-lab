"""Assess our completed own detector+reader on every original validation photo."""
from collections import Counter
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import time

import torch

from slm_lab.byte_word_reader import ByteWordReader
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.evaluation_provenance import capture
from slm_lab.masked_visual_pretraining import process_identity
from slm_lab.ocr_evaluation import normalized
from slm_lab.photo_word_reading import infer, original_validation_targets, reading_score
from slm_lab.word_localization_evaluation import aggregate
from slm_lab.word_localizer import SIZE, WordLocalizer, load_rows, pixels

ROOT = Path(__file__).resolve().parents[1]
DETECTOR_DONE = 'scratch full-photo word detector trained and assessed; integrated reliable reading unverified'
READER_DONE = 'scratch byte-word component trained and assessed; native full-photo reading unverified'


def verified_state(name, expected_status):
    report_dir = ROOT / 'reports' / name; path = ROOT / 'checkpoints' / name / 'model.pt'
    status = json.loads((report_dir / 'status.json').read_text())
    comparison = json.loads((report_dir / 'comparison.json').read_text()); plan = json.loads((report_dir / 'plan.json').read_text())
    digest = file_hash(path)
    if status['status'] != expected_status or digest != comparison['model_sha256']: raise ValueError('Photo-reader component is not the declared completed model: ' + name)
    snapshot = torch.load(path, map_location='cpu', weights_only=True)
    if snapshot['plan'] != plan or snapshot['step'] != plan['steps']: raise ValueError('Photo-reader component did not complete its fixed training plan: ' + name)
    files = [ROOT / item for item in plan['source_provenance']['files_sha256']]
    if capture(files) != plan['source_provenance']: raise ValueError('Photo-reader component implementation/runtime changed: ' + name)
    return snapshot, plan, digest


def matching_localization_cases(rows, report):
    if report['full_photos'] != len(rows) or len(report['results']) != len(rows): raise ValueError('Incomplete original full-photo detector coverage')
    locations = {row['id']: row for row in report['results']}
    if sorted(locations) != [row['id'] for row in rows]: raise ValueError('Unmatched original full-photo detector cases')
    for index, row in enumerate(rows):
        other = rows[(index + 1) % len(rows)]; actual = locations[row['id']]
        expected = {'group_id': row['group_id'], 'stored_image_sha256': row['stored_image_sha256'],
                    'input_pixels_sha256_768': row['model_pixel_sha256_768'], 'wrong_image_id': other['id'],
                    'wrong_pixels_sha256_768': other['model_pixel_sha256_768']}
        if any(actual[key] != value for key, value in expected.items()): raise ValueError('Unmatched actual full-photo bytes/pixels/counterpart')
    return locations


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--detector-pid', type=int, required=True)
    parser.add_argument('--detector-start-ticks', required=True); args = parser.parse_args()
    reports = ROOT / 'reports/photo-word-reading-v1'
    if reports.exists(): raise ValueError('Preserve existing whole-photo reading assessment; inspect before resuming')
    process = Path('/proc') / str(args.detector_pid)
    if process_identity(args.detector_pid) != args.detector_start_ticks or b'scripts/train_word_localizer.py' not in process.joinpath('cmdline').read_bytes().split(b'\0'):
        raise ValueError('Expected whole-photo detector predecessor is not verified live')
    torch.set_num_threads(1)
    files = [Path(__file__)] + [ROOT / 'src/slm_lab' / (name + '.py') for name in
             ['photo_word_reading', 'word_localizer', 'word_localization_evaluation', 'byte_word_reader', 'text_localization_data',
              'text_localization_metrics', 'ocr_evaluation', 'rendered_text', 'model', 'training', 'corpus',
              'diverse_images', 'evaluation_provenance', 'masked_visual_pretraining', 'vqa_pairs', 'public_sources']]
    runtime = capture(files); data = ROOT / 'data/text-localization-v1'; manifest_sha = file_hash(data / 'manifest.json')
    queue = ROOT / 'reports/word-localizer-v1/queued-plan.json'; queue_sha = file_hash(queue)
    reports.mkdir()
    atomic_json(reports / 'queued-plan.json', {'created_at': datetime.now(timezone.utc).isoformat(), 'detector_pid': args.detector_pid,
                'detector_start_ticks': args.detector_start_ticks, 'validation_full_photos': 400,
                'data_manifest_sha256': manifest_sha, 'detector_queued_plan_sha256': queue_sha,
                'source_provenance': runtime, 'device': 'cuda:3', 'training_updates_added': 0, 'automatic_policy_promotion': False,
                'scope': 'Own from-scratch detector+reader whole-photo composition; no target boxes/transcripts at inference; policy question answering still requires separate validation.',
                'goal_achieved': False})
    def status(message, **values):
        atomic_json(reports / 'status.json', {'status': message, 'updated_at': datetime.now(timezone.utc).isoformat(), 'goal_achieved': False, **values})
    def verify():
        if capture(files) != runtime or file_hash(data / 'manifest.json') != manifest_sha or file_hash(queue) != queue_sha:
            raise ValueError('Loaded composed-photo evaluation implementation/data/queue changed')
    try:
        status('waiting for own detector training and full-photo localization assessment', detector_pid=args.detector_pid, detector_start_ticks=args.detector_start_ticks)
        while process_identity(args.detector_pid) == args.detector_start_ticks:
            verify(); time.sleep(15)
        verify(); detector_state, detector_plan, detector_sha = verified_state('word-localizer-v1', DETECTOR_DONE)
        reader_state, reader_plan, reader_sha = verified_state('byte-word-reader-v1', READER_DONE)
        rows, manifest = load_rows(data); rows = sorted(rows['valid'], key=lambda row: row['id'])
        if len(rows) != 400: raise ValueError('Original complete400-photo evaluation scope changed')
        targets = original_validation_targets(rows, manifest)
        location_report_path = ROOT / 'reports/word-localizer-v1/after.json'; location_report_sha = file_hash(location_report_path)
        location_report = json.loads(location_report_path.read_text()); locations = matching_localization_cases(rows, location_report)
        device = 'cuda:3'; torch.cuda.set_per_process_memory_fraction(.2, 3)
        detector = WordLocalizer().to(device); reader = ByteWordReader().to(device)
        detector.load_state_dict(detector_state['model']); reader.load_state_dict(reader_state['model']); del detector_state, reader_state
        model_words = {}
        status('generating own word transcripts from predicted boxes on complete photographs')
        for index, row in enumerate(rows):
            result = infer(pixels(row), detector, reader, device)
            if result['padded_photo_rgb_sha256'] != row['model_pixel_sha256_768']: raise ValueError('Composed-photo input bytes differ from original geometry admission')
            predicted_boxes = [{key: word[key] for key in ['box', 'score', 'stride']} for word in result['predicted_words']]
            if predicted_boxes != locations[row['id']]['proper_predictions']: raise ValueError('Composed reading used different actual boxes than the completed localization assessment')
            model_words[row['id']] = result['predicted_words']
            if index % 25 == 0:
                verify(); print('Actual complete-photo words inferred', index + 1, len(rows), flush=True)
                status('generating own word transcripts from predicted boxes on complete photographs', full_photos_completed=index + 1, full_photos=len(rows))
        blank = infer(torch.full((3, SIZE, SIZE), -1.), detector, reader, device)
        if [{key: word[key] for key in ['box', 'score', 'stride']} for word in blank['predicted_words']] != location_report['blank_predictions']:
            raise ValueError('Different actual blank-photo predictions between localization and composed reading')
        cases = []; correct_texts = Counter(); correct_annotations = []
        for index, row in enumerate(rows):
            other = rows[(index + 1) % len(rows)]
            case = {'id': row['id'], 'group_id': row['group_id'], 'input_pixels_sha256_768': row['model_pixel_sha256_768'],
                    'stored_image_sha256': row['stored_image_sha256'], 'wrong_image_id': other['id'],
                    'wrong_pixels_sha256_768': other['model_pixel_sha256_768'], 'proper_predictions': model_words[row['id']]}
            for control, predictions in [('proper', model_words[row['id']]), ('blank', blank['predicted_words']), ('wrong', model_words[other['id']])]:
                case[control + '_scores'] = reading_score(predictions, targets[row['id']])
                if case[control + '_scores']['localization'] != locations[row['id']][control + '_scores']:
                    raise ValueError('Original detection metric/counterpart changed while scoring transcripts')
            valid = [target for target in targets[row['id']] if target['legible']]
            for target_index, prediction_index in case['proper_scores']['matched_target_prediction_indices']:
                target = valid[target_index]; word = model_words[row['id']][prediction_index]; correct_texts[normalized(target['text'])] += 1
                correct_annotations.append({'photo_id': row['id'], 'original_word_id': target['original_annotation_id'], 'expected': target['text'],
                                            'generated': word['text'], 'predicted_box': word['box'], 'original_box': target['normalized_padded_box']})
            cases.append(case)
        verify()
        if file_hash(location_report_path) != location_report_sha or file_hash(ROOT / 'checkpoints/word-localizer-v1/model.pt') != detector_sha or file_hash(ROOT / 'checkpoints/byte-word-reader-v1/model.pt') != reader_sha:
            raise ValueError('Completed source detector/reader/report changed during composed evaluation')
        report = {'completed_at': datetime.now(timezone.utc).isoformat(), 'full_photos': len(rows), 'legible_original_words': sum(target['legible'] for items in targets.values() for target in items),
                  'detector_sha256': detector_sha, 'reader_sha256': reader_sha, 'data_manifest_sha256': manifest_sha, 'source_provenance': runtime,
                  'totals': {control: aggregate([row[control + '_scores'] for row in cases]) for control in ['proper', 'blank', 'wrong']},
                  'detection_totals': {control: aggregate([row[control + '_scores']['localization'] for row in cases]) for control in ['proper', 'blank', 'wrong']},
                  'correct_transcript_histogram': dict(correct_texts.most_common()), 'blank_predictions': blank['predicted_words'], 'results': cases,
                  'crop_protocol': blank['crop_protocol'], 'automatic_policy_promotion': False, 'native_policy_question_answering_verified': False,
                  'control_generation': 'Every actual proper full photo inferred; deterministic blank/counterpart predictions reused explicitly without original-target correction.',
                  'hypothetical_policy_reader_detector_parameters': detector_plan['hypothetical_policy_reader_detector_parameters'],
                  'limitations': ['Existing400-photo development split; not independent final broad capability evidence.',
                                  'Entire original legible annotation set retained, including tiny/rotated/unfamiliar/long words.',
                                  'Axis-aligned localization and word transcription only; document ordering and native policy question answering unverified.',
                                  'Original-format mirrored TextOCR bytes remain independently unverified against blocked original endpoint.',
                                  'Composed modules stay separate from the production policy; no coding/language capability claim.'], 'goal_achieved': False}
        atomic_json(reports / 'reading.json', report)
        atomic_json(reports / 'correct-cases.json', {'cases': correct_annotations, 'count': len(correct_annotations), 'goal_achieved': False})
        status('own complete-photo localization/transcription assessed; reliable native multimodal policy unverified', full_photos=len(rows))
    except Exception as error:
        status('composed photo reading assessment failed; inspect actual evidence before resuming', error=str(error)); raise


if __name__ == '__main__': main()
