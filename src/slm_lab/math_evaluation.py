"""Tool-free, fixed greedy numeric-answer assessment on original GSM8K test data."""
import argparse
from contextlib import nullcontext
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from .corpus import file_hash
from .diverse_images import atomic_json, jsonl
from .knowledge_evaluation import interval
from .math_corpus import final_answer, prompt
from .training import load_checkpoint
from .evaluation_provenance import capture, seal


@torch.no_grad()
def evaluate(checkpoint, data, output, device='cuda', max_new_tokens=512, max_examples=None):
    provenance = capture(); data_digest = file_hash(data)
    if max_new_tokens < 1 or (max_examples is not None and max_examples < 1):
        raise ValueError('Evaluation budgets must be positive')
    data = Path(data); manifest = json.loads((data.parent/'manifest.json').read_text())
    if data.name != 'test.jsonl' or file_hash(data) != manifest['files_sha256']['test.jsonl']:
        raise ValueError('Use the frozen original test split')
    rows = list(jsonl(data))
    if len(rows) != manifest['official_test_examples']:
        raise ValueError('Test count differs from the prepared manifest')
    # Stable question-hash order permits honest matched prefix comparisons for
    # bounded pilot runs; the full assessment still includes every official row.
    rows.sort(key=lambda r: hashlib.sha256(r['question'].encode()).digest())
    if max_examples is not None: rows = rows[:max_examples]
    torch.set_num_threads(1); model, meta = load_checkpoint(checkpoint, device, training_state=False); model.eval()
    tokenizer_path = Path(checkpoint).parent/'tokenizer.json'
    if file_hash(tokenizer_path) != meta['tokenizer_sha256']: raise ValueError('Tokenizer mismatch')
    tokenizer = Tokenizer.from_file(str(tokenizer_path)); results = []
    amp = lambda: torch.autocast('cuda', dtype=torch.bfloat16) if device.startswith('cuda') else nullcontext()
    for row in rows:
        text = 'User: '+prompt(row['question'])+'\nAssistant:'
        ids = torch.tensor([[tokenizer.token_to_id('<bos>')]+tokenizer.encode(text).ids], device=device)
        if ids.shape[1]+max_new_tokens > model.config.max_text_length:
            raise ValueError('Math prompt exceeds fixed context budget')
        with amp():
            tokens = model.generate(ids, max_new_tokens=max_new_tokens, eos_token_id=tokenizer.token_to_id('<eos>'), use_cache=True)
        raw = tokenizer.decode(tokens[0, ids.shape[1]:].tolist()); answer = final_answer(raw)
        gold = Fraction(row['answer_numerator'], row['answer_denominator'])
        results.append({'id': row['id'], 'raw_response': raw,
                        'answer_numerator': answer.numerator if answer is not None else None,
                        'answer_denominator': answer.denominator if answer is not None else None,
                        'expected_numerator': gold.numerator, 'expected_denominator': gold.denominator,
                        'correct': answer == gold, 'format_compliant': answer is not None,
                        'context_tokens': ids.shape[1]})
        if len(results)%20 == 0: print('math problems evaluated', len(results), flush=True)
    correct = sum(row['correct'] for row in results); total = len(results)
    report = {'checkpoint_sha256': file_hash(checkpoint), 'data_sha256': file_hash(data),
              'manifest_sha256': file_hash(data.parent/'manifest.json'),
              'official_test_examples': manifest['official_test_examples'], 'evaluated_examples': total,
              'full_official_test': total == manifest['official_test_examples'],
              'correct': correct, 'numeric_exact_accuracy': correct/total if total else None,
              'accuracy_95ci': interval(correct, total),
              'format_compliance': sum(row['format_compliant'] for row in results)/total if total else None,
              'generation_budget': max_new_tokens,
              'protocol': 'Zero-shot native User/Assistant prompt, greedy cached generation, no tools/calculator/repair; last complete #### numeric line, exact rational comparison. Stable question-hash order.',
              'limitations': manifest['limitations'] + ['Protocol differs from publisher few-shot/calculator experiments; a bounded prefix is not a full benchmark score.',
                                                       'Numeric final-answer matches do not prove the generated reasoning is valid.'],
              'results': results}
    seal(report, provenance, data=data, digest=data_digest, device=device, tokenizer_sha256=meta['tokenizer_sha256'])
    atomic_json(Path(output), report)
    print(json.dumps({key: value for key, value in report.items() if key != 'results'}, indent=2)); return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True); parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True); parser.add_argument('--device', default='cuda')
    parser.add_argument('--max-new-tokens', type=int, default=512); parser.add_argument('--max-examples', type=int)
    args = parser.parse_args(); evaluate(args.checkpoint, args.data, args.output, args.device, args.max_new_tokens, args.max_examples)


if __name__ == '__main__': main()
