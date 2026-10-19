"""Seen-example learning versus free generation for our completed code SFT probe."""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from .code_quality import execute, instruction_code
from .corpus import file_hash
from .diverse_images import atomic_json, jsonl
from .evaluation_provenance import capture, seal
from .training import Encoder, load_checkpoint
from .vision_transfer import language_digest

ROOT = Path(__file__).resolve().parents[2]


def select(rows, tokenizer, count=32, response_budget=256):
    """Hash-select one original-name, short demonstration per source algorithm."""
    eligible = []
    seen = set()
    for row in sorted(rows, key=lambda r: hashlib.sha256(r['id'].encode()).digest()):
        if row['variant_index'] != 0 or row['original_program_sha256'] in seen:
            continue
        if len(tokenizer.encode(row['response']).ids) + 1 > response_budget:
            continue
        if not row.get('verification_cases'):
            raise ValueError('Published verification examples are missing')
        seen.add(row['original_program_sha256']); eligible.append(row)
    if len(eligible) < count:
        raise ValueError('Not enough distinct short training programs for the declared diagnostic')
    return eligible[:count]


@torch.no_grad()
def evaluate(checkpoint, data, output, device='cuda', count=128, max_new_tokens=512, exposures=None):
    files = [ROOT / name for name in capture()['files_sha256']] + [Path(__file__), ROOT / 'src/slm_lab/vision_transfer.py']
    provenance = capture(files)
    data = Path(data); data_digest = file_hash(data)
    source_manifest = json.loads((data.parent / 'manifest.json').read_text())
    if data_digest != source_manifest['files_sha256'][data.name]:
        raise ValueError('Expected the exact admitted instruction-trial source pool')
    if exposures is None:
        raise ValueError('Actual completed source exposure is required for a seen-example claim')
    exposures = Path(exposures)
    if file_hash(exposures) != source_manifest['files_sha256'][exposures.name]:
        raise ValueError('Actual completed instruction exposure plan differs')
    exposed = {row['id'] for row in jsonl(exposures) if row['pool'] == 'verified_python'}
    torch.set_num_threads(1)
    model, metadata = load_checkpoint(checkpoint, device, training_state=False); model.eval()
    tokenizer_path = Path(checkpoint).parent / 'tokenizer.json'
    if file_hash(tokenizer_path) != metadata['tokenizer_sha256']:
        raise ValueError('Own tokenizer identity changed')
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    rows = select([row for row in jsonl(data) if row['id'] in exposed], tokenizer, count)
    if max_new_tokens < 256:
        raise ValueError('Generation must accommodate every selected source response')
    encoder = Encoder(tokenizer, model.config, data.parent)
    before = language_digest(model.state_dict())
    amp = lambda: torch.autocast('cuda', dtype=torch.bfloat16) if device.startswith('cuda') else nullcontext()
    results = []
    for row in rows:
        gold = execute({'language': 'Python', 'code': instruction_code(row['response']), 'function': row['function'],
                        'cases': row['verification_cases'], 'stress': []})
        if not gold.get('functional_passed'):
            raise RuntimeError('Previously verified demonstration no longer passes: ' + row['id'])
        batch = encoder.batch([row], 'response', device)
        with amp():
            out = model(**batch)
            predictions = out['logits'][:, :-1].argmax(-1)
            labels = batch['labels'][:, 1:]; valid = labels != -100
            targets = int(valid.sum()); correct = int(((predictions == labels) & valid).sum())
            nll = float(out['loss'])
            prompt = [encoder.bos] + encoder.tokens(encoder.prompt(row))
            if len(prompt) + max_new_tokens > model.config.max_text_length:
                raise ValueError('Seen-example generation would exceed model context')
            ids = torch.tensor([prompt], device=device)
            generated = model.generate(ids, max_new_tokens=max_new_tokens, eos_token_id=encoder.eos)
        raw = tokenizer.decode(generated[0, len(prompt):].tolist())
        code = instruction_code(raw)
        candidate = execute({'language': 'Python', 'code': code, 'function': row['function'],
                             'cases': row['verification_cases'], 'stress': []})
        results.append({'id': row['id'], 'source': row['source'], 'source_url': row['source_url'],
                        'function': row['function'], 'original_program_sha256': row['original_program_sha256'],
                        'prompt_sha256': hashlib.sha256(encoder.prompt(row).encode()).hexdigest(),
                        'source_response_tokens': len(tokenizer.encode(row['response']).ids),
                        'published_cases': len(row['verification_cases']),
                        'target_tokens': targets, 'teacher_forced_correct_tokens': correct, 'teacher_forced_nll': nll,
                        'raw_response': raw, 'code': code, 'candidate': candidate})
        print(json.dumps({'seen_examples': len(results), 'functional': candidate['functional_passed'],
                          'teacher_forced_nll': nll}), flush=True)
    if before != language_digest(model.state_dict()):
        raise ValueError('Diagnostic modified own language/critic tensors')
    if encoder.truncated:
        raise ValueError('Seen-example diagnostic truncated source text')
    targets = sum(row['target_tokens'] for row in results)
    report = {'checkpoint_sha256': file_hash(checkpoint), 'checkpoint_stage': metadata['stage'],
              'checkpoint_steps': metadata['steps'], 'data_sha256': data_digest,
              'source_manifest_sha256': file_hash(data.parent / 'manifest.json'),
              'examples': len(results), 'generation_budget': max_new_tokens,
              'selection': 'Actual exposed training IDs only; SHA256(id), original-name variants, unique source algorithms, gold response <=255 own tokens',
              'actual_exposures_sha256': file_hash(exposures),
              'target_tokens': targets,
              'teacher_forced_nll': sum(row['teacher_forced_nll'] * row['target_tokens'] for row in results) / targets,
              'teacher_forced_token_accuracy': sum(row['teacher_forced_correct_tokens'] for row in results) / targets,
              'syntax_pass_rate': sum(row['candidate']['syntax_valid'] for row in results) / len(results),
              'published_example_functional_rate': sum(row['candidate']['functional_passed'] for row in results) / len(results),
              'exact_language_and_critic_preserved': True, 'results': results,
              'goal_achieved': False, 'production_promoted': False,
              'limitations': ['Training prompts and published source examples: learning/memorization diagnostic only.',
                             'Examples are visible in prompts; no held-out generalization or efficiency claim.',
                             'Short original-name Python programs only; renamed variants and other languages are not assessed.',
                             'Teacher forcing supplies preceding gold tokens; its accuracy does not measure free generation correctness.']}
    seal(report, provenance, files=files, data=data, digest=data_digest, device=device,
         tokenizer_sha256=metadata['tokenizer_sha256'])
    atomic_json(Path(output), report)
    print(json.dumps({key: value for key, value in report.items() if key not in {'results', 'evaluation_provenance'}}, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True); parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True); parser.add_argument('--device', default='cuda')
    parser.add_argument('--examples', type=int, default=32); parser.add_argument('--max-new-tokens', type=int, default=512)
    args = parser.parse_args()
    if args.examples < 1:
        parser.error('Positive diagnostic example count required')
    evaluate(args.checkpoint, args.data, args.output, args.device, args.examples, args.max_new_tokens)


if __name__ == '__main__':
    main()
