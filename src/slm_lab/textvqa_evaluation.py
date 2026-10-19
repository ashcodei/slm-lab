"""Native reading questions on real photos, without external OCR or text hints."""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from .clevr_evaluation import compare_reports as compare_questions
from .corpus import file_hash
from .diverse_images import atomic_json, jsonl
from .evaluation_provenance import capture, pixels_sha256, seal
from .object_evaluation import summarize
from .ocr_evaluation import normalized
from .textvqa_data import SOURCE_SHA, agreed_target, same_photo
from .training import Encoder, load_checkpoint
from .vqa_pair_evaluation import human_agreement

ROOT = Path(__file__).resolve().parents[2]
HUMAN_PROTOCOL = ('Ten original answers, leave-one-out mean min(other matches/3,1); '
                  'NFKC/case/whitespace normalization only, not the complete official punctuation/number scorer.')


def answer_key(value):
    return normalized(value).casefold()


def select(rows, count=128):
    if count < 2 or len({row['id'] for row in rows}) != len(rows):
        raise ValueError('Need distinct original real-reading question identities and at least two cases')
    selected = []; groups = set()
    for row in sorted(rows, key=lambda item: hashlib.sha256(item['id'].encode()).digest()):
        if row['group_id'] not in groups:
            selected.append(row); groups.add(row['group_id'])
        if len(selected) == count: break
    if len(selected) < 2:
        raise ValueError('Need at least two separate original full photographs')
    for row in selected:
        if not any(other['group_id'] != row['group_id'] and answer_key(other['response']) != answer_key(row['response']) for other in selected):
            raise ValueError('Missing different-image, different-answer control')
    return selected


def compare_reports(before, after):
    if before['human_score_protocol'] != after['human_score_protocol']:
        raise ValueError('Unmatched human answer scoring')
    guard = compare_questions(before, after)
    if [row['original_answers_sha256'] for row in before['results']] != [row['original_answers_sha256'] for row in after['results']]:
        raise ValueError('Unmatched original human answer records')
    guard['scope'] = 'Matched reused real-photo reading development subset; not full TextVQA or independent final broad evaluation.'
    return guard


@torch.no_grad()
def evaluate(checkpoint, output, data=None, device='cuda:2', max_examples=128, max_new_tokens=32):
    data = ROOT / 'data/textvqa-human-reading-v1' if data is None else Path(data)
    data_path = data / 'valid.jsonl'; digest = file_hash(data_path)
    manifest_path = data / 'manifest.json'; manifest_digest = file_hash(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if digest != manifest['files_sha256']['valid.jsonl'] or manifest['validation_group_overlap'] != 0:
        raise ValueError('Finalized split-safe real-reading validation records differ')
    source_path = ROOT / 'data/textvqa-source-cache-v1/TextVQA_0.5.1_val.json'
    if file_hash(source_path) != SOURCE_SHA['val']:
        raise ValueError('Original publisher validation answers changed')
    originals = {row['question_id']: row for row in json.loads(source_path.read_text())['data']}
    rows = select(list(jsonl(data_path)), max_examples)
    for row in rows:
        original = row['original_textvqa_question']; source = originals[original['question_id']]
        if (original != source or row['id'] != 'textvqa:val:' + str(source['question_id'])
                or row['prompt'] != source['question'] + '\nAnswer briefly.'
                or agreed_target(source) != (row['response'], row['identical_original_answer_votes'])
                or not same_photo(source, row)):
            raise ValueError('Stored real-reading case differs from its actual original question/photo/human answers')
    files = [ROOT / name for name in capture()['files_sha256']] + [Path(__file__),
             ROOT / 'src/slm_lab/textvqa_data.py', ROOT / 'src/slm_lab/vqa_pairs.py',
             ROOT / 'src/slm_lab/vqa_pair_evaluation.py', ROOT / 'src/slm_lab/clevr_evaluation.py']
    provenance = capture(files)
    torch.set_num_threads(1)
    model, metadata = load_checkpoint(checkpoint, device, training_state=False); model.eval()
    tokenizer = Path(checkpoint).parent / 'tokenizer.json'
    if file_hash(tokenizer) != manifest['tokenizer_sha256'] or metadata['tokenizer_sha256'] != manifest['tokenizer_sha256']:
        raise ValueError('Own model and public reading records use different tokenizers')
    encoder = Encoder(Tokenizer.from_file(str(tokenizer)), model.config, data, 448)
    amp = lambda: torch.autocast('cuda', dtype=torch.bfloat16) if device.startswith('cuda') else nullcontext()
    results = []
    for index, row in enumerate(rows):
        alternatives = [other for other in rows if other['group_id'] != row['group_id'] and answer_key(other['response']) != answer_key(row['response'])]
        other = alternatives[index % len(alternatives)]
        for item in [row, other]:
            if file_hash(Path(item['image'])) != item['stored_image_sha256']:
                raise ValueError('Actual credited full-photo bytes changed')
        prompt = encoder.prompt(row); tokens = [encoder.bos] + encoder.tokens(prompt)
        if len(tokens) + max_new_tokens > model.config.max_text_length:
            raise ValueError('Complete real-reading prompt/generation budget exceeds context')
        ids = torch.tensor([tokens], device=device)
        proper = encoder.pixels(row)[None].to(device); wrong = encoder.pixels(other)[None].to(device)
        batch = encoder.batch([row], 'response', device)
        answers = [{'answer_id': number, 'answer': value} for number, value in enumerate(row['original_textvqa_question']['answers'], 1)]
        result = {'id': row['id'], 'group_id': row['group_id'], 'task': 'text_reading_question', 'expected': row['response'],
                  'wrong_image_id': other['id'], 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                  'input_pixels_sha256': pixels_sha256(proper), 'wrong_pixels_sha256': pixels_sha256(wrong),
                  'original_answers_sha256': hashlib.sha256(json.dumps(answers).encode()).hexdigest(),
                  'target_tokens': int((batch['labels'][:, 1:] != -100).sum())}
        for control, pixels in [('proper', proper), ('blank', torch.zeros_like(proper)), ('wrong', wrong)]:
            with amp():
                generated = model.generate(ids, pixels, max_new_tokens=max_new_tokens, eos_token_id=encoder.eos)
                loss = float(model(**{**batch, 'pixel_values': pixels})['loss'])
            raw = encoder.tokenizer.decode(generated[0, len(tokens):].tolist())
            result[control + '_generated'] = raw
            result[control + '_exact'] = normalized(raw).casefold() == normalized(row['response']).casefold()
            result[control + '_human_agreement'] = human_agreement(raw, answers)
            result[control + '_nll'] = loss
        results.append(result)
        if len(results) % 16 == 0: print('Original real-photo reading evaluated', len(results), flush=True)
    overall = summarize(results)
    for control in ['proper', 'blank', 'wrong']:
        overall[control + '_human_agreement'] = sum(row[control + '_human_agreement'] for row in results) / len(results)
    report = {'checkpoint_sha256': file_hash(checkpoint), 'data_sha256': digest, 'manifest_sha256': manifest_digest,
              'tokenizer_sha256': metadata['tokenizer_sha256'], 'image_size': 448, 'max_examples': max_examples,
              'max_new_tokens': max_new_tokens, 'overall': overall, 'results': results,
              'human_score_protocol': HUMAN_PROTOCOL, 'goal_achieved': False,
              'protocol': 'Hash-selected distinct real validation photos; original question only;448px,greedy32tokens,proper/blank/different-image controls; no supplied OCR/class labels.',
              'limitations': ['Reused held-out development photos, strict six-of-ten exact consensus subset; not full official TextVQA or final independent evaluation.',
                              'Controls retain the original target; a changed image is not newly annotated for the current question.',
                              HUMAN_PROTOCOL]}
    if encoder.truncated: raise ValueError('Original public reading answers were truncated')
    seal(report, provenance, files=files, data=data_path, digest=digest, device=device,
         tokenizer_sha256=metadata['tokenizer_sha256'])
    if file_hash(manifest_path) != manifest_digest or file_hash(source_path) != SOURCE_SHA['val']:
        raise ValueError('Original real-reading source/manifest changed during evaluation')
    atomic_json(Path(output), report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True); parser.add_argument('--output', required=True)
    parser.add_argument('--data'); parser.add_argument('--device', default='cuda:2')
    args = parser.parse_args(); evaluate(args.checkpoint, args.output, data=args.data, device=args.device)


if __name__ == '__main__':
    main()
