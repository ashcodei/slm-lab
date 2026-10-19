"""Matched policy assessments using the compact own-reader training interface."""
import argparse
from contextlib import nullcontext
import gzip
import hashlib
import json
import math
from pathlib import Path

import torch
from tokenizers import Tokenizer

from .byte_word_reader import ByteWordReader
from .corpus import file_hash
from .diverse_images import atomic_json, jsonl
from .evaluation_provenance import capture, pixels_sha256, seal
from .object_evaluation import summarize
from .own_ocr_context import prompt
from .own_ocr_question_admission import admit
from .own_ocr_question_evaluation import control_words, image_pixels
from .own_ocr_training_data import compact_predictions, variants
from .photo_word_reading import infer
from .textvqa_evaluation import HUMAN_PROTOCOL, answer_key, compare_reports as compare_questions, select
from .training import Encoder, load_checkpoint
from .vqa_pair_evaluation import human_agreement
from .word_localizer import SIZE, WordLocalizer

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ('Original128 hash-selected distinct validation photos; compact own pixel-predicted text, '
            'top64 words/four-decimal geometry/fixed256-token headroom;448px policy,greedy32tokens; '
            'proper/gray-blank/different-image controls each use their own predicted words.')


def conditioned_prompt(question, predictions, tokenizer, config):
    return prompt(question, compact_predictions(predictions), tokenizer,
                  max_text_length=config.max_text_length, generation_tokens=256, max_words=64)


def compare_reports(before, after):
    for key in ['own_context_manifest_sha256', 'detector_sha256', 'reader_sha256',
                'context_headroom_tokens', 'maximum_context_words', 'image_size_own_reader']:
        if before[key] != after[key]: raise ValueError('Unmatched own-reader conditioning protocol: ' + key)
    for report in [before, after]:
        for row in report['results']:
            for control in ['proper', 'blank', 'wrong']:
                expected = answer_key(row[control + '_generated']) == answer_key(row['expected'])
                if row[control + '_exact'] != expected or not math.isfinite(row[control + '_nll']):
                    raise ValueError('Own-context case score disagrees with generated answer or loss')
        actual = summarize(report['results'])
        if any(report['overall'][key] != value for key, value in actual.items()):
            raise ValueError('Own-context aggregate differs from generated case evidence')
    guard = compare_questions(before, after)
    for first, second in zip(before['results'], after['results']):
        for control in ['proper', 'blank', 'wrong']:
            for suffix in ['_prompt_sha256', '_target_tokens', '_selected_context']:
                if first[control + suffix] != second[control + suffix]:
                    raise ValueError('Unmatched actual own-context prompt/answer budget: ' + control + suffix)
        for key in ['own_OCR_input_rgb_sha256_768', 'wrong_OCR_rgb_sha256_768']:
            if first[key] != second[key]: raise ValueError('Unmatched own-reader actual image pixels')
    guard['scope'] = 'Matched compact own-reader native answers; reused original development photos, not independent broad visual capability.'
    return guard


@torch.inference_mode()
def evaluate(checkpoint, output, device='cuda', max_examples=128):
    output = Path(output)
    if output.exists(): raise ValueError('Preserve existing own-context policy assessment')
    _, admission = admit(ROOT)
    data = ROOT / 'data/textvqa-human-reading-v1'; own_data = ROOT / 'data/own-ocr-human-questions-v1'
    manifest_path = data / 'manifest.json'; manifest_sha = file_hash(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    own_manifest = json.loads((own_data / 'manifest.json').read_text())
    data_path = data / 'valid.jsonl'; data_sha = file_hash(data_path)
    checkpoint_sha = file_hash(checkpoint)
    detector_path = ROOT / 'checkpoints/word-localizer-v1/model.pt'
    reader_path = ROOT / 'checkpoints/native-word-refinement-v1/model.pt'
    predictions = {}
    with gzip.open(own_data / 'valid.predictions.jsonl.gz', 'rt') as stream:
        for line in stream:
            item = json.loads(line); predictions[item['group_id']] = item
    rows = select(list(jsonl(data_path)), max_examples)
    files = [ROOT / name for name in capture()['files_sha256']] + [Path(__file__)]
    files += [ROOT / 'src/slm_lab' / (name + '.py') for name in
              ['own_ocr_question_admission', 'own_ocr_training_data', 'own_ocr_context', 'own_ocr_question_evaluation',
               'photo_word_reading', 'word_localizer', 'byte_word_reader', 'text_localization_metrics',
               'textvqa_admission', 'textvqa_input_audit', 'textvqa_data', 'textvqa_evaluation',
               'vqa_pairs', 'vqa_pair_evaluation', 'clevr_evaluation']]
    provenance = capture(files); torch.set_num_threads(1)
    model, metadata = load_checkpoint(checkpoint, device, training_state=False); model.eval()
    tokenizer_path = Path(checkpoint).parent / 'tokenizer.json'
    if metadata['tokenizer_sha256'] != manifest['tokenizer_sha256'] or file_hash(tokenizer_path) != manifest['tokenizer_sha256']:
        raise ValueError('Own policy tokenizer differs from original human questions')
    tokenizer = Tokenizer.from_file(str(tokenizer_path)); encoder = Encoder(tokenizer, model.config, data, 448)
    detector = WordLocalizer().to(device)
    detector.load_state_dict(torch.load(detector_path, map_location='cpu', weights_only=True)['model'])
    reader = ByteWordReader().to(device)
    reader.load_state_dict(torch.load(reader_path, map_location='cpu', weights_only=True)['model'])
    combined_parameters = model.parameter_counts()['total'] + sum(item.numel() for item in detector.parameters()) + sum(item.numel() for item in reader.parameters())
    if combined_parameters >= 1_000_000_000: raise ValueError('Own conditioned text/image system exceeds parameter limit')
    blank_words = infer(torch.zeros(3, SIZE, SIZE), detector, reader, device)['predicted_words']
    del detector, reader
    if device.startswith('cuda'): torch.cuda.empty_cache()
    inventory = {group: item['predicted_words'] for group, item in predictions.items()}
    results = []; amp = lambda: torch.autocast('cuda', dtype=torch.bfloat16) if device.startswith('cuda') else nullcontext()
    for index, row in enumerate(rows):
        alternatives = [other for other in rows if other['group_id'] != row['group_id'] and answer_key(other['response']) != answer_key(row['response'])]
        other = alternatives[index % len(alternatives)]
        rgb_hashes = {}
        for item in [row, other]:
            pixels = image_pixels(item)
            raw_rgb = pixels.add(1).mul(127.5).round().clamp(0, 255).to(torch.uint8).permute(1, 2, 0).contiguous().numpy()
            rgb_hashes[item['group_id']] = hashlib.sha256(raw_rgb.tobytes()).hexdigest()
            if rgb_hashes[item['group_id']] != predictions[item['group_id']]['reader_rgb_pixels_sha256_768']:
                raise ValueError('Recorded own predicted text refers to different actual reader pixels')
        words = control_words(row['group_id'], other['group_id'], inventory, blank_words)
        proper = encoder.pixels(row)[None].to(device); wrong = encoder.pixels(other)[None].to(device)
        answers = [{'answer_id': number, 'answer': value} for number, value in enumerate(row['original_textvqa_question']['answers'], 1)]
        result = {'id': row['id'], 'group_id': row['group_id'], 'task': 'own_context_reading_question', 'expected': row['response'],
                  'wrong_image_id': other['id'], 'input_pixels_sha256': pixels_sha256(proper), 'wrong_pixels_sha256': pixels_sha256(wrong),
                  'own_OCR_input_rgb_sha256_768': rgb_hashes[row['group_id']], 'wrong_OCR_rgb_sha256_768': rgb_hashes[other['group_id']],
                  'original_answers_sha256': hashlib.sha256(json.dumps(answers).encode()).hexdigest()}
        for control, pixels in [('proper', proper), ('blank', torch.zeros_like(proper)), ('wrong', wrong)]:
            context = conditioned_prompt(row['original_textvqa_question']['question'], words[control], tokenizer, model.config)
            conditioned = {**row, 'prompt': context['prompt']}
            if control == 'proper' and context['prompt'] != variants(row, words[control], tokenizer, model.config)[1]['prompt']:
                raise ValueError('Assessment context differs from actual training interface')
            actual_prompt = encoder.prompt(conditioned); tokens = [encoder.bos] + encoder.tokens(actual_prompt)
            if len(tokens) + 32 > model.config.max_text_length: raise ValueError('Complete conditioned question exceeds generation budget')
            ids = torch.tensor([tokens], device=device); batch = encoder.batch([conditioned], 'response', device)
            if encoder.truncated: raise ValueError('Complete original human answer truncated in conditioning assessment')
            with amp():
                generated = model.generate(ids, pixels, max_new_tokens=32, eos_token_id=encoder.eos)
                loss = float(model(**{**batch, 'pixel_values': pixels})['loss'])
            raw = tokenizer.decode(generated[0, len(tokens):].tolist())
            result.update({control + '_generated': raw, control + '_exact': answer_key(raw) == answer_key(row['response']),
                           control + '_human_agreement': human_agreement(raw, answers), control + '_nll': loss,
                           control + '_prompt_sha256': hashlib.sha256(actual_prompt.encode()).hexdigest(),
                           control + '_target_tokens': int((batch['labels'][:, 1:] != -100).sum()),
                           control + '_selected_context': context['selected_context']})
        result['prompt_sha256'] = result['proper_prompt_sha256']; result['target_tokens'] = result['proper_target_tokens']
        results.append(result)
        if len(results) % 16 == 0: print('Compact own-context native policy questions assessed', len(results), flush=True)
    overall = summarize(results)
    for control in ['proper', 'blank', 'wrong']:
        overall[control + '_human_agreement'] = sum(row[control + '_human_agreement'] for row in results) / len(results)
    if (file_hash(checkpoint) != checkpoint_sha or file_hash(manifest_path) != manifest_sha
            or file_hash(own_data / 'manifest.json') != admission['manifest_sha256']): raise ValueError('Own-context policy/data changed during assessment')
    for path, digest in own_manifest['own_component_sha256'].items():
        if file_hash(path) != digest: raise ValueError('Own fixed reading component changed during assessment')
    for name, digest in own_manifest['files_sha256'].items():
        if file_hash(own_data / name) != digest: raise ValueError('Own predicted context inventory changed during assessment')
    for row in rows:
        if file_hash(row['image']) != row['stored_image_sha256']: raise ValueError('Original assessed photo changed during generation')
    report = {'checkpoint_sha256': checkpoint_sha, 'data_sha256': data_sha, 'manifest_sha256': manifest_sha,
              'own_context_manifest_sha256': admission['manifest_sha256'], 'tokenizer_sha256': metadata['tokenizer_sha256'],
              'detector_sha256': own_manifest['own_component_sha256'][str(detector_path)],
              'reader_sha256': own_manifest['own_component_sha256'][str(reader_path)],
              'image_size': 448, 'image_size_own_reader': SIZE, 'max_examples': max_examples, 'max_new_tokens': 32,
              'context_headroom_tokens': 256, 'maximum_context_words': 64, 'overall': overall, 'results': results,
              'total_inference_parameters': combined_parameters,
              'human_score_protocol': HUMAN_PROTOCOL, 'protocol': PROTOCOL, 'goal_achieved': False,
              'limitations': ['Reused strict-consensus development subset, not full official TextVQA or independent final evaluation.',
                              'Proper/counterpart own predictions are reused from the hash-bound finalized inventory; blank predictions come from normalized-gray pixels.',
                              'Changed-image controls keep the original target; OCR predictions may be wrong.', HUMAN_PROTOCOL]}
    seal(report, provenance, files, data=data_path, digest=data_sha, device=device, tokenizer_sha256=metadata['tokenizer_sha256'])
    atomic_json(output, report); return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    print(json.dumps(evaluate(args.checkpoint, args.output, args.device)['overall']), flush=True)
