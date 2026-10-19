"""Fixed-policy real-photo questions with plain/empty/own-predicted text contexts."""
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
from tokenizers import Tokenizer

from .byte_word_reader import ByteWordReader
from .corpus import file_hash
from .diverse_images import atomic_json, jsonl, pixel_hash
from .evaluation_provenance import capture, pixels_sha256
from .ocr_evaluation import normalized
from .own_ocr_context import prompt as context_prompt
from .photo_word_reading import infer
from .textvqa_data import SOURCE_SHA
from .textvqa_input_audit import validate_question
from .training import Encoder, load_checkpoint
from .vqa_pair_evaluation import human_agreement
from .word_localizer import SIZE, WordLocalizer

ROOT = Path(__file__).resolve().parents[2]
MODES = ('plain', 'empty_context', 'own_context')
CONTROLS = ('proper', 'blank', 'wrong')


def image_pixels(row):
    path = Path(row['image']).resolve()
    if not path.is_relative_to(ROOT): raise ValueError('Own-OCR question photo path leaves project')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != row['stored_image_sha256']: raise ValueError('Own-OCR question actual photo bytes changed')
    with Image.open(io.BytesIO(raw)) as opened:
        image = opened.convert('RGB')
        for size, key in [(224, 'model_pixel_sha256'), (448, 'model_pixel_sha256_448')]:
            if pixel_hash(image, size) != row[key]: raise ValueError('Original native question image pixels changed')
        image = ImageOps.pad(image, (SIZE, SIZE), method=Image.Resampling.BICUBIC, color=(128, 128, 128))
        array = np.asarray(image, dtype=np.float32) / 127.5 - 1
    return torch.from_numpy(array.copy()).permute(2, 0, 1)


def control_words(proper_id, wrong_id, predictions, blank):
    if proper_id == wrong_id or proper_id not in predictions or wrong_id not in predictions: raise ValueError('Invalid original photo/counterpart prediction binding')
    return {'proper': predictions[proper_id], 'blank': blank, 'wrong': predictions[wrong_id]}


def aggregate(results):
    if not results: raise ValueError('Empty own-OCR native question assessment')
    output = {'examples': len(results), 'original_photos': len({row['group_id'] for row in results})}
    for mode in MODES:
        output[mode] = {}
        for control in CONTROLS:
            cases = [row['modes'][mode][control] for row in results]
            output[mode][control] = {'correct': sum(row['exact'] for row in cases), 'accuracy': sum(row['exact'] for row in cases) / len(cases),
                                     'mean_human_agreement': sum(row['human_agreement'] for row in cases) / len(cases)}
    return output


@torch.inference_mode()
def evaluate(output, device='cuda:3'):
    output = Path(output)
    if output.exists(): raise ValueError('Preserve existing own-OCR native question assessment')
    baseline_path = ROOT / 'reports/textvqa-joint-transfer-v1/before.json'; baseline_sha = file_hash(baseline_path)
    baseline = json.loads(baseline_path.read_text())
    checkpoint = ROOT / 'checkpoints/clevr-dense-learning-v1/vision/model.pt'
    detector_path = ROOT / 'checkpoints/word-localizer-v1/model.pt'; reader_path = ROOT / 'checkpoints/native-word-refinement-v1/model.pt'
    digests = {str(path): file_hash(path) for path in [checkpoint, detector_path, reader_path]}
    if digests[str(checkpoint)] != baseline['checkpoint_sha256']: raise ValueError('Question experiment policy differs from original own baseline')
    for name, path, done in [('word-localizer-v1', detector_path, 'scratch full-photo word detector trained and assessed; integrated reliable reading unverified'),
                             ('native-word-refinement-v1', reader_path, 'native-word refinement trained and assessed; full-photo/native-policy benefit unverified')]:
        status = json.loads((ROOT / 'reports' / name / 'status.json').read_text())
        comparison = json.loads((ROOT / 'reports' / name / 'comparison.json').read_text())
        if status['status'] != done or comparison['model_sha256'] != digests[str(path)]: raise ValueError('Own image component is not its completed declared model')
    data = ROOT / 'data/textvqa-human-reading-v1'; data_path = data / 'valid.jsonl'; manifest_path = data / 'manifest.json'
    data_sha = file_hash(data_path); manifest_sha = file_hash(manifest_path); manifest = json.loads(manifest_path.read_text())
    if data_sha != baseline['data_sha256'] or manifest_sha != baseline['manifest_sha256'] or data_sha != manifest['files_sha256']['valid.jsonl']:
        raise ValueError('Original complete native-question input scope changed')
    original_path = ROOT / 'data/textvqa-source-cache-v1/TextVQA_0.5.1_val.json'
    if file_hash(original_path) != SOURCE_SHA['val']: raise ValueError('Original human question/answer release changed')
    originals = {row['question_id']: row for row in json.loads(original_path.read_text())['data']}
    all_rows = {row['id']: row for row in jsonl(data_path)}; selected = [all_rows[row['id']] for row in baseline['results']]
    if len(selected) != 128 or len({row['group_id'] for row in selected}) != 128: raise ValueError('All original128 unique native-question photos are required')
    for row in selected: validate_question(row, originals[row['original_textvqa_question']['question_id']], 'val')
    files = [ROOT / path for path in baseline['evaluation_provenance']['files_sha256']] + [Path(__file__)]
    files += [ROOT / 'src/slm_lab' / (name + '.py') for name in ['own_ocr_context', 'photo_word_reading', 'word_localizer', 'byte_word_reader', 'textvqa_input_audit']]
    runtime = capture(files)
    for name, digest in baseline['evaluation_provenance']['files_sha256'].items():
        if runtime['files_sha256'][name] != digest: raise ValueError('Original native-question inference implementation changed')
    torch.set_num_threads(1); torch.cuda.set_per_process_memory_fraction(.2, int(str(device).split(':')[-1]))
    model, metadata = load_checkpoint(checkpoint, device, training_state=False); model.eval()
    tokenizer_path = checkpoint.parent / 'tokenizer.json'; tokenizer_sha = file_hash(tokenizer_path)
    if tokenizer_sha != manifest['tokenizer_sha256'] or tokenizer_sha != metadata['tokenizer_sha256']: raise ValueError('Own policy tokenizer differs from original questions')
    tokenizer = Tokenizer.from_file(str(tokenizer_path)); encoder = Encoder(tokenizer, model.config, data, 448)
    detector = WordLocalizer().to(device); detector.load_state_dict(torch.load(detector_path, map_location='cpu', weights_only=True)['model'])
    reader = ByteWordReader().to(device); reader.load_state_dict(torch.load(reader_path, map_location='cpu', weights_only=True)['model'])
    combined = model.parameter_counts()['total'] + sum(parameter.numel() for parameter in detector.parameters()) + sum(parameter.numel() for parameter in reader.parameters())
    if combined >= 1_000_000_000: raise ValueError('Own question/image/word system exceeds parameter request')
    predicted = {}; ocr_pixels = {}
    for index, row in enumerate(selected):
        actual = image_pixels(row); result = infer(actual, detector, reader, device)
        predicted[row['id']] = result['predicted_words']; ocr_pixels[row['id']] = result['padded_photo_rgb_sha256']
        if index % 16 == 0: print('Own predicted question-photo text prepared', index + 1, len(selected), flush=True)
    # Original native baseline uses normalized gray-zero blank input. Regenerate
    # its own text context; never carry proper-photo words into a control.
    blank = infer(torch.zeros(3, SIZE, SIZE), detector, reader, device)['predicted_words']
    by_id = {row['id']: row for row in selected}; results = []; baseline_differences = []
    for index, (row, old) in enumerate(zip(selected, baseline['results'])):
        other = by_id[old['wrong_image_id']]; contexts = control_words(row['id'], other['id'], predicted, blank)
        proper = encoder.pixels(row)[None].to(device); wrong = encoder.pixels(other)[None].to(device)
        if pixels_sha256(proper) != old['input_pixels_sha256'] or pixels_sha256(wrong) != old['wrong_pixels_sha256']: raise ValueError('Original actual native policy image inputs changed')
        image_controls = {'proper': proper, 'blank': torch.zeros_like(proper), 'wrong': wrong}
        answers = [{'answer_id': number, 'answer': value} for number, value in enumerate(row['original_textvqa_question']['answers'], 1)]
        case = {'id': row['id'], 'group_id': row['group_id'], 'expected': row['response'], 'wrong_image_id': other['id'],
                'input_pixels_sha256': old['input_pixels_sha256'], 'wrong_pixels_sha256': old['wrong_pixels_sha256'],
                'own_OCR_input_rgb_sha256_768': ocr_pixels[row['id']], 'wrong_OCR_rgb_sha256_768': ocr_pixels[other['id']],
                'original_answers_sha256': hashlib.sha256(json.dumps(answers).encode()).hexdigest(), 'modes': {}}
        for mode in MODES:
            case['modes'][mode] = {}
            for control in CONTROLS:
                if mode == 'plain':
                    raw_prompt = row['prompt']; budget = None
                else:
                    budget = context_prompt(row['original_textvqa_question']['question'], contexts[control] if mode == 'own_context' else [],
                                            tokenizer, max_text_length=model.config.max_text_length, generation_tokens=32, max_words=64)
                    raw_prompt = budget['prompt']
                actual_prompt = encoder.prompt({**row, 'prompt': raw_prompt}); ids_list = [encoder.bos] + encoder.tokens(actual_prompt)
                if len(ids_list) + 32 > model.config.max_text_length: raise ValueError('Complete native question/context exceeds fixed generation budget')
                if mode == 'plain' and hashlib.sha256(actual_prompt.encode()).hexdigest() != old['prompt_sha256']: raise ValueError('Original plain-question prompt changed')
                ids = torch.tensor([ids_list], device=device)
                with torch.autocast('cuda', dtype=torch.bfloat16): output_ids = model.generate(ids, image_controls[control], max_new_tokens=32, eos_token_id=encoder.eos)
                text = tokenizer.decode(output_ids[0, len(ids_list):].tolist())
                result = {'generated': text, 'exact': normalized(text).casefold() == normalized(row['response']).casefold(),
                          'human_agreement': human_agreement(text, answers), 'prompt_sha256': hashlib.sha256(actual_prompt.encode()).hexdigest(),
                          'prompt_tokens': len(ids_list), 'predicted_words_supplied': budget['predicted_words_supplied'] if budget else 0,
                          'supplied_predicted_words': budget['selected_context'] if budget else []}
                case['modes'][mode][control] = result
                if mode == 'plain' and text != old[control + '_generated']:
                    baseline_differences.append({'id': row['id'], 'control': control, 'previous': old[control + '_generated'], 'fresh': text})
        results.append(case)
        if index % 8 == 0: print('Own-OCR native questions actually generated', index + 1, len(selected), flush=True)
    if capture(files) != runtime or file_hash(baseline_path) != baseline_sha or file_hash(data_path) != data_sha or file_hash(manifest_path) != manifest_sha:
        raise ValueError('Own native-question experiment implementation/input/baseline changed')
    if any(file_hash(path) != digest for path, digest in digests.items()) or file_hash(original_path) != SOURCE_SHA['val']: raise ValueError('Own native question model/original answers changed')
    report = {'completed_at': datetime.now(timezone.utc).isoformat(), 'overall': aggregate(results), 'results': results,
              'policy_sha256': digests[str(checkpoint)], 'detector_sha256': digests[str(detector_path)], 'reader_sha256': digests[str(reader_path)],
              'data_sha256': data_sha, 'manifest_sha256': manifest_sha, 'tokenizer_sha256': tokenizer_sha, 'source_provenance': runtime,
              'image_size_policy': 448, 'image_size_own_reader': 768, 'max_new_tokens': 32, 'maximum_predicted_context_words': 64,
              'total_inference_parameters': combined, 'fresh_plain_baseline_transcript_differences': baseline_differences,
              'automatic_policy_promotion': False, 'goal_achieved': False,
              'scope': 'Same frozen own policy/original128 development questions; plain versus empty-schema versus own pixel-predicted text context; each proper/gray-blank/wrong-photo context regenerated from its own input.',
              'limitations': ['Frozen policy has not been trained with this new context; conditioning success/skill requires measured answers.',
                              'Reused strict-consensus development subset, not full independent official TextVQA or broad capability.',
                              'Changed-image controls retain original answer; no newly annotated counterpart question.',
                              'Human agreement uses existing NFKC/case/whitespace normalization, not full official punctuation/number scoring.',
                              'Own reader crop/full-photo metrics and parameter count do not establish frontier equivalence.']}
    atomic_json(output, report); atomic_json(output.with_name('summary.json'), {key: value for key, value in report.items() if key not in ['results', 'source_provenance']})
    return report
