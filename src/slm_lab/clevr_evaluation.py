"""Original visual reasoning questions with actual, blank and wrong-image controls."""
import argparse
from collections import defaultdict, deque
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer
from .corpus import file_hash
from .diverse_images import atomic_json, jsonl
from .evaluation_provenance import capture, seal, pixels_sha256
from .object_evaluation import summarize
from .ocr_evaluation import normalized
from .training import Encoder, load_checkpoint

ROOT = Path(__file__).resolve().parents[2]


def select(rows, count=256):
    groups = defaultdict(lambda: defaultdict(deque))
    for row in sorted(rows, key=lambda value: hashlib.sha256(value['id'].encode()).hexdigest()):
        groups[row['reasoning_type']][row['response']].append(row)
    result = []
    for task, answers in sorted(groups.items()):
        seen = set(); chosen = []; budget = max(1, count // len(groups))
        while answers and len(chosen) < budget:
            for label in sorted(list(answers)):
                queue = answers[label]
                while queue and queue[0]['group_id'] in seen: queue.popleft()
                if queue:
                    row = queue.popleft(); chosen.append(row); seen.add(row['group_id'])
                if not queue: del answers[label]
                if len(chosen) >= budget: break
        result.extend(chosen)
    return result[:count]


def compare_reports(before, after):
    for key in ['evaluation_provenance', 'data_sha256', 'manifest_sha256', 'tokenizer_sha256', 'image_size',
                'inference_precision', 'inference_device_type', 'max_new_tokens', 'max_examples', 'protocol']:
        if before[key] != after[key]: raise ValueError('Unmatched visual reasoning assessment: ' + key)
    def cases(report):
        rows = report['results']
        if not rows or len(rows) != report['overall']['examples'] or len({row['id'] for row in rows}) != len(rows):
            raise ValueError('Visual reasoning case evidence is empty, incomplete or repeated')
        accuracy = sum(row['proper_exact'] for row in rows) / len(rows)
        if report['overall']['correct_image_exact_accuracy'] != accuracy:
            raise ValueError('Visual reasoning aggregate differs from actual case evidence')
        return [(row['id'], row['group_id'], row['expected'], row['task'], row['prompt_sha256'],
                 row['input_pixels_sha256'], row['wrong_pixels_sha256'], row['wrong_image_id'], row['target_tokens'])
                for row in rows]
    if cases(before) != cases(after): raise ValueError('Unmatched visual reasoning cases/pixels/controls')
    lost = [first['id'] for first, second in zip(before['results'], after['results']) if first['proper_exact'] and not second['proper_exact']]
    passed = not lost and after['overall']['correct_image_exact_accuracy'] >= before['overall']['correct_image_exact_accuracy']
    return {'passed': passed, 'previously_correct_questions_lost': lost,
            'before_exact_accuracy': before['overall']['correct_image_exact_accuracy'],
            'after_exact_accuracy': after['overall']['correct_image_exact_accuracy'],
            'base_checkpoint_sha256': before['checkpoint_sha256'], 'candidate_checkpoint_sha256': after['checkpoint_sha256'],
            'goal_achieved': False, 'scope': 'Matched reused original CLEVR subset; no generic image/high-end capability claim.'}


@torch.no_grad()
def evaluate(checkpoint, data, output, device='cuda', max_examples=256, max_new_tokens=16):
    files = [ROOT / path for path in capture()['files_sha256']] + [Path(__file__)]
    provenance = capture(files); data = Path(data); digest = file_hash(data)
    manifest = json.loads((data.parent / 'manifest.json').read_text())
    if data.name != 'valid.jsonl' or digest != manifest['files_sha256'][data.name]: raise ValueError('Use finalized original validation questions')
    torch.set_num_threads(1); model, meta = load_checkpoint(checkpoint, device, training_state=False); model.eval()
    tokenizer = Path(checkpoint).parent / 'tokenizer.json'
    if file_hash(tokenizer) != manifest['tokenizer_sha256'] or meta['tokenizer_sha256'] != manifest['tokenizer_sha256']:
        raise ValueError('Own tokenizer differs from original reasoning data')
    encoder = Encoder(Tokenizer.from_file(str(tokenizer)), model.config, data.parent, 448)
    rows = select(list(jsonl(data)), max_examples)
    if not rows: raise ValueError('No original held-out visual reasoning questions')
    amp = lambda: torch.autocast('cuda', dtype=torch.bfloat16) if device.startswith('cuda') else nullcontext()
    results = []
    for index, row in enumerate(rows):
        alternatives = [other for other in rows if other['reasoning_type'] == row['reasoning_type'] and
                        other['group_id'] != row['group_id'] and other['response'] != row['response']]
        if not alternatives: raise ValueError('No independent differing-target image control')
        other = alternatives[index % len(alternatives)]
        for sample in [row, other]:
            if file_hash(data.parent / sample['image']) != sample['stored_image_sha256']: raise ValueError('Original stored image changed')
        prompt = encoder.prompt(row); tokens = [encoder.bos] + encoder.tokens(prompt)
        if len(tokens) + max_new_tokens > model.config.max_text_length: raise ValueError('Complete visual prompt exceeds context')
        ids = torch.tensor([tokens], device=device); pixels = encoder.pixels(row)[None].to(device)
        wrong = encoder.pixels(other)[None].to(device); batch = encoder.batch([row], 'response', device)
        result = {'id': row['id'], 'group_id': row['group_id'], 'task': row['reasoning_type'], 'expected': row['response'],
                  'wrong_image_id': other['id'], 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                  'input_pixels_sha256': pixels_sha256(pixels), 'wrong_pixels_sha256': pixels_sha256(wrong),
                  'target_tokens': int((batch['labels'][:, 1:] != -100).sum())}
        for name, image in [('proper', pixels), ('blank', torch.zeros_like(pixels)), ('wrong', wrong)]:
            with amp():
                generated = model.generate(ids, image, max_new_tokens=max_new_tokens, eos_token_id=encoder.eos)
                nll = float(model(**{**batch, 'pixel_values': image})['loss'])
            raw = encoder.tokenizer.decode(generated[0, len(tokens):].tolist())
            result[name + '_generated'] = raw; result[name + '_exact'] = normalized(raw).casefold() == normalized(row['response']).casefold()
            result[name + '_nll'] = nll
        results.append(result)
        if len(results) % 32 == 0: print('Original visual reasoning evaluated', len(results), flush=True)
    report = {'checkpoint_sha256': file_hash(checkpoint), 'data_sha256': digest, 'manifest_sha256': file_hash(data.parent / 'manifest.json'),
              'image_size': 448, 'max_examples': max_examples, 'max_new_tokens': max_new_tokens, 'overall': summarize(results),
              'by_task': {task: summarize([r for r in results if r['task'] == task]) for task in sorted({r['task'] for r in results})},
              'results': results, 'protocol': 'Answer-balanced unique-image selection within original program terminal types; greedy BF16 CUDA,448px,16tokens; original questions only, no scene/program/gold in model input; correct/blank/wrong controls.',
              'goal_achieved': False, 'limitations': ['Original synthetic subset, not full CLEVR/CoGenT, real photos or independent final generic visual reasoning.',
                          'Wrong/blank matches retain the original target, not correctness for another newly labelled scene.',
                          'Answer-mode baseline and positive ablation gains alone do not establish reliable reasoning.']}
    if encoder.truncated: raise ValueError('Original visual reasoning source targets truncated')
    seal(report, provenance, files=files, data=data, digest=digest, device=device, tokenizer_sha256=meta['tokenizer_sha256'])
    atomic_json(Path(output), report); return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True); parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True); parser.add_argument('--device', default='cuda')
    args = parser.parse_args(); evaluate(args.checkpoint, args.data, args.output, args.device)


if __name__ == '__main__': main()
