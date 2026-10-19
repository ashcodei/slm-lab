"""Object identification/counting/spatial QA with blank and wrong-image controls."""
import argparse
from collections import Counter, defaultdict, deque
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from .corpus import file_hash
from .data import read_jsonl
from .object_qa import TASKS
from .ocr_evaluation import normalized
from .training import Encoder, load_checkpoint


def select(rows, count):
    tasks = defaultdict(list)
    for row in rows:
        if row.get('task') in TASKS:
            tasks[row['task']].append(row)
    selected = []
    for task, records in sorted(tasks.items()):
        answers = defaultdict(deque)
        for row in sorted(records, key=lambda row: hashlib.sha256(row['id'].encode()).digest()):
            answers[normalized(row['response']).casefold()].append(row)
        seen = set(); chosen = []
        while answers and len(chosen) < max(1, count // len(tasks)):
            for answer in sorted(list(answers),key=lambda value:hashlib.sha256(value.encode()).digest()):
                queue = answers[answer]
                while queue and queue[0]['group_id'] in seen:
                    queue.popleft()
                if queue:
                    row = queue.popleft(); chosen.append(row); seen.add(row['group_id'])
                if not queue:
                    del answers[answer]
                if len(chosen) >= max(1, count // len(tasks)):
                    break
        selected.extend(chosen)
    return selected[:count]


def summarize(rows):
    targets = sum(row['target_tokens'] for row in rows)
    answers = Counter(normalized(row['expected']).casefold() for row in rows)
    return {'examples': len(rows), 'original_images': len({row['group_id'] for row in rows}),
            'correct_image_exact_accuracy': sum(row['proper_exact'] for row in rows) / len(rows),
            'blank_original_target_match': sum(row['blank_exact'] for row in rows) / len(rows),
            'wrong_image_original_target_match': sum(row['wrong_exact'] for row in rows) / len(rows),
            'answer_mode_baseline': max(answers.values()) / len(rows),
            'proper_nll': sum(row['proper_nll'] * row['target_tokens'] for row in rows) / targets,
            'blank_nll_gain': sum((row['blank_nll'] - row['proper_nll']) * row['target_tokens'] for row in rows) / targets,
            'mismatch_nll_gain': sum((row['wrong_nll'] - row['proper_nll']) * row['target_tokens'] for row in rows) / targets}


@torch.no_grad()
def evaluate(checkpoint, data, output, device='cuda', max_examples=180, max_new_tokens=32):
    from .evaluation_provenance import capture,seal,pixels_sha256
    provenance=capture();data_digest=file_hash(data)
    torch.set_num_threads(1)
    model, meta = load_checkpoint(checkpoint, device, training_state=False); model.eval()
    tokenizer_path = Path(checkpoint).parent / 'tokenizer.json'
    if file_hash(tokenizer_path) != meta['tokenizer_sha256']:
        raise ValueError('Tokenizer mismatch')
    size = meta.get('training_options', {}).get('image_size', model.config.image_size)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    encoder = Encoder(tokenizer, model.config, Path(data).parent, size)
    rows = select(read_jsonl(data), max_examples)
    if not rows:
        raise ValueError('No object questions found')
    amp = lambda: torch.autocast('cuda', dtype=torch.bfloat16) if device.startswith('cuda') else nullcontext()
    results = []
    for i, row in enumerate(rows):
        alternatives = [other for other in rows if other['task'] == row['task'] and other['group_id'] != row['group_id']
                        and normalized(other['response']).casefold() != normalized(row['response']).casefold()]
        if not alternatives:
            raise ValueError('Object image intervention requires differing targets on independent photos')
        other = alternatives[i % len(alternatives)]
        body = encoder.tokens(encoder.prompt(row))
        if len(body) + max_new_tokens + 1 > model.config.max_text_length:
            raise ValueError('Object prompt exceeds context budget')
        ids = torch.tensor([[encoder.bos] + body], device=device)
        pixels = encoder.pixels(row)[None].to(device)
        wrong = encoder.pixels(other)[None].to(device)
        interventions = {'proper': pixels, 'blank': torch.zeros_like(pixels), 'wrong': wrong}
        batch = encoder.batch([row], 'response', device)
        result = {'id': row['id'], 'group_id': row['group_id'], 'task': row['task'],
                  'expected': row['response'], 'wrong_image_id': other['id'],
                  'input_pixels_sha256':pixels_sha256(pixels),'wrong_pixels_sha256':pixels_sha256(wrong),
                  'target_tokens': int((batch['labels'][:, 1:] != -100).sum())}
        for name, image in interventions.items():
            with amp():
                tokens = model.generate(ids, image, max_new_tokens=max_new_tokens, eos_token_id=encoder.eos)
                loss = float(model(**{**batch, 'pixel_values': image})['loss'])
            generated = tokenizer.decode(tokens[0, ids.shape[1]:].tolist())
            result[name + '_generated'] = generated
            result[name + '_exact'] = normalized(generated).casefold() == normalized(row['response']).casefold()
            result[name + '_nll'] = loss
        results.append(result)
        if len(results) % 30 == 0:
            print('object questions evaluated', len(results), flush=True)
    report = {'checkpoint_sha256': file_hash(checkpoint), 'data_sha256': file_hash(data),
              'image_size': size, 'max_new_tokens': max_new_tokens,
              'overall': summarize(results),
              'by_task': {task: summarize([row for row in results if row['task'] == task]) for task in sorted({row['task'] for row in results})},
              'protocol': 'Fixed greedy generation. NFKC/whitespace/case normalized exact target match; answer-balanced unique-photo sampling within tasks.',
              'results': results,
              'limitations': ['Derived public annotation questions; this is an introductory tuning assessment, not a standardized broad VQA benchmark.',
                             'Blank/wrong-image target matches measure retention of the original answer, not correctness on a new unlabelled scene.',
                             'The empirical answer-mode baseline is text-prior context, not a complete no-image model baseline.',
                             'Positive image-ablation gains alone do not establish general visual reasoning.']}
    seal(report,provenance,data=data,digest=data_digest,device=device,tokenizer_sha256=meta['tokenizer_sha256'])
    Path(output).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'results'}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True); parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True); parser.add_argument('--device', default='cuda')
    parser.add_argument('--max-examples', type=int, default=180); parser.add_argument('--max-new-tokens', type=int, default=32)
    args = parser.parse_args()
    if min(args.max_examples, args.max_new_tokens) < 1:
        parser.error('Evaluation budgets must be positive')
    evaluate(args.checkpoint, args.data, args.output, args.device, args.max_examples, args.max_new_tokens)


if __name__ == '__main__':
    main()
