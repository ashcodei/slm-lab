"""Generated Python modules against supported held-out Exercism literal tests."""
import argparse
import ast
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

ROOT = Path(__file__).resolve().parents[2]


def compare_reports(before, after):
    for key in ['evaluation_provenance', 'data_sha256', 'manifest_sha256', 'tokenizer_sha256',
                'inference_precision', 'inference_device_type', 'declared_generation_budget', 'protocol', 'examples']:
        if before[key] != after[key]: raise ValueError('Unmatched exercise-code assessment: ' + key)
    def cases(report):
        return [(r['id'], r['group_id'], r['prompt_sha256'], r['generation_budget'],
                 [(c['function'], c['published_cases']) for c in r['checks']]) for r in report['results']]
    if cases(before) != cases(after): raise ValueError('Unmatched exercise module cases')
    lost = [old['id'] for old, new in zip(before['results'], after['results'])
            if old['all_supported_literal_tests_passed'] and not new['all_supported_literal_tests_passed']]
    return {'passed': not lost and after['python_syntax_accuracy'] >= before['python_syntax_accuracy']
            and after['module_literal_test_pass_rate'] >= before['module_literal_test_pass_rate'],
            'previously_correct_modules_lost': lost,
            'before_literal_module_rate': before['module_literal_test_pass_rate'],
            'after_literal_module_rate': after['module_literal_test_pass_rate'],
            'base_checkpoint_sha256': before['checkpoint_sha256'],
            'candidate_checkpoint_sha256': after['checkpoint_sha256'], 'goal_achieved': False,
            'scope': 'Additional supported Python literal module cases; no other-language or broad correctness/efficiency claim.'}


@torch.no_grad()
def evaluate(checkpoint, data, output, device='cuda', max_new_tokens=2048):
    files = [ROOT / name for name in capture()['files_sha256']] + [Path(__file__)]
    provenance = capture(files); data = Path(data); digest = file_hash(data)
    manifest = json.loads((data.parent / 'manifest.json').read_text())
    if data.name != 'valid.jsonl' or manifest['files_sha256'][data.name] != digest:
        raise ValueError('Use the finalized exercise-held-out validation data')
    torch.set_num_threads(1)
    model, metadata = load_checkpoint(checkpoint, device, training_state=False); model.eval()
    tokenizer_path = Path(checkpoint).parent / 'tokenizer.json'
    if file_hash(tokenizer_path) != manifest['tokenizer_sha256'] or metadata['tokenizer_sha256'] != manifest['tokenizer_sha256']:
        raise ValueError('Exercise model/tokenizer mismatch')
    encoder = Encoder(Tokenizer.from_file(str(tokenizer_path)), model.config, data.parent)
    amp = lambda: torch.autocast('cuda', dtype=torch.bfloat16) if device.startswith('cuda') else nullcontext()
    results = []; skipped = []
    for row in jsonl(data):
        if row['language'] != 'Python' or not row['reference_checks']:
            skipped.append({'id': row['id'], 'language': row['language'], 'reason': 'No supported literal Python function tests'}); continue
        gold = instruction_code(row['response'])
        for check in row['reference_checks']:
            verified = execute({'language': 'Python', 'code': gold, 'function': check['function'],
                                'cases': check['published_literal_cases'], 'stress': []})
            if not verified.get('functional_passed'): raise ValueError('Publisher reference no longer passes its supported tests')
        prompt = encoder.prompt(row); tokens = [encoder.bos] + encoder.tokens(prompt)
        budget = min(max_new_tokens, model.config.max_text_length - len(tokens))
        if budget < len(encoder.tokens(row['response'])) + 1:
            raise ValueError('Generation budget cannot accommodate the complete held-out reference module')
        ids = torch.tensor([tokens], device=device)
        with amp(): generated = model.generate(ids, max_new_tokens=budget, eos_token_id=encoder.eos)
        response = encoder.tokenizer.decode(generated[0, len(tokens):].tolist()); code = instruction_code(response)
        try: syntax = bool(ast.parse(code).body)
        except SyntaxError: syntax = False
        checks = []
        for check in row['reference_checks']:
            result = execute({'language': 'Python', 'code': code, 'function': check['function'],
                              'cases': check['published_literal_cases'], 'stress': []})
            checks.append({'function': check['function'], 'published_cases': check['published_literal_cases'], 'candidate': result})
        item = {'id': row['id'], 'group_id': row['group_id'], 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                'context_tokens': len(tokens), 'generation_budget': budget, 'source_files': row['source_files'],
                'raw_response': response, 'code': code, 'python_syntax_valid': syntax,
                'all_supported_literal_tests_passed': all(c['candidate'].get('functional_passed') for c in checks),
                'checks': checks}
        results.append(item)
        print('exercise modules evaluated', len(results), 'passed', item['all_supported_literal_tests_passed'], flush=True)
    if not results: raise ValueError('No supported held-out exercise modules')
    report = {'checkpoint_sha256': file_hash(checkpoint), 'data_sha256': digest,
              'manifest_sha256': file_hash(data.parent / 'manifest.json'), 'declared_generation_budget': max_new_tokens,
              'examples': len(results), 'supported_function_tests': sum(len(r['checks']) for r in results),
              'python_syntax_accuracy': sum(r['python_syntax_valid'] for r in results) / len(results),
              'module_literal_test_pass_rate': sum(r['all_supported_literal_tests_passed'] for r in results) / len(results),
              'results': results, 'skipped': skipped, 'truncated_sequences': 0,
              'protocol': 'Original task and starter interfaces; greedy BF16; full prompts; per-case budget min(2048, remaining own context); no tests/reference body in model input.',
              'goal_achieved': False,
              'limitations': ['Supported literal Python function assertions only; other languages/classes/error interfaces are not executed.',
                              'Examples explicitly described by the original task may be visible; this is not all publisher tests, efficiency or an independent final benchmark.',
                              'All prior pretraining and semantic exercise overlap are not exhaustively audited.',
                              'Restricted-worker import/attribute limits can reject syntactically valid candidates; raw Python syntax is reported separately.']}
    seal(report, provenance, files=files, data=data, digest=digest, device=device, tokenizer_sha256=metadata['tokenizer_sha256'])
    atomic_json(Path(output), report)
    print(json.dumps({key: value for key, value in report.items() if key not in {'results', 'evaluation_provenance', 'skipped'}}, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True); parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True); parser.add_argument('--device', default='cuda')
    args = parser.parse_args(); evaluate(args.checkpoint, args.data, args.output, args.device)


if __name__ == '__main__': main()
