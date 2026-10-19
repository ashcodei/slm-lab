"""Assess complete generated C++ exercise modules with original published tests."""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from .corpus import file_hash
from .diverse_images import atomic_json, jsonl
from .evaluation_provenance import capture, seal
from .exercise_cpp_runner import PublishedSuites, extract_files
from .training import Encoder, load_checkpoint

ROOT = Path(__file__).resolve().parents[2]
EXTRA_FILES = [Path(__file__), ROOT / 'src/slm_lab/exercise_cpp_runner.py']
PROTOCOL = ('Original task/starter interfaces plus exact labeled-file output instruction; greedy BF16 on CUDA; '
            'full prompts, min(2048, remaining own context) new tokens; ISO C++17; all publisher tests enabled; '
            'pre-exec Landlock/seccomp compilation/runtime; exact reference JUnit case/assertion inventory required.')


def inputs(data, cache):
    data = Path(data); manifest = json.loads((data.parent / 'manifest.json').read_text())
    if data.name != 'valid.jsonl' or file_hash(data) != manifest['files_sha256'][data.name]:
        raise ValueError('Use finalized held-out exercise data')
    rows = [row for row in jsonl(data) if row['language'] == 'C++']
    if not rows: raise ValueError('No C++ exercise modules')
    files = [ROOT / name for name in capture()['files_sha256']] + EXTRA_FILES
    files += [Path(cache) / 'source.json', data.parent / 'manifest.json']
    return data, manifest, rows, files


def solution(row, suites):
    files = extract_files(row['response'])
    source_paths = {Path(record['path']).name for record in row['source_files'] if '/.meta/' not in record['path']}
    if not set(files).issubset(source_paths): raise ValueError('Publisher response contains an undeclared solution file')
    for record in row['source_files']:
        if record.get('repository') == 'exercism/problem-specifications': continue
        if hashlib.sha256(suites.read(record['path'])).hexdigest() != record['sha256']:
            raise ValueError('Original publisher solution/interface bytes changed')
    return files


def verify_references(data, output, cache):
    data, manifest, rows, files = inputs(data, cache); provenance = capture(files)
    digest = file_hash(data); suites = PublishedSuites(cache); results = []
    try:
        for row in rows:
            reference = solution(row, suites); suite = suites.suite(row)
            result = suites.run(reference, suite)
            results.append({'id': row['id'], 'group_id': row['group_id'], 'exercise': row['exercise'],
                            'solution_files': sorted(reference), 'response_sha256': hashlib.sha256(row['response'].encode()).hexdigest(),
                            'test_sha256': suite['test_sha256'], 'catch_sha256': suite['catch_sha256'],
                            'main_sha256': suite['main_sha256'], 'reference': result})
            print('C++ publisher reference', row['exercise'], result['published_tests_passed'], flush=True)
        report = {'data_sha256': digest, 'manifest_sha256': file_hash(data.parent / 'manifest.json'),
                  'toolchain': suites.tools, 'source': suites.source, 'compiler_options': suites.common[1:],
                  'examples': len(rows), 'references_passed': sum(r['reference']['published_tests_passed'] for r in results),
                  'results': results, 'goal_achieved': False,
                  'scope': 'Published reference/runner verification, not a trained-model capability score.'}
        seal(report, provenance, files=files, data=data, digest=digest)
        atomic_json(Path(output), report)
        if report['references_passed'] != len(rows): raise ValueError('Some original references failed; inspect the complete report')
        return report
    finally: suites.close()


@torch.no_grad()
def evaluate(checkpoint, data, output, cache, reference_report, device='cuda:2'):
    data, manifest, rows, files = inputs(data, cache); provenance = capture(files)
    digest = file_hash(data); receipt = json.loads(Path(reference_report).read_text())
    if (receipt['evaluation_provenance'] != provenance or receipt['data_sha256'] != digest or
        receipt['references_passed'] != len(rows) or receipt['examples'] != len(rows)):
        raise ValueError('Publisher reference receipt is incomplete or belongs to another evaluator/data')
    references = {r['id']: r for r in receipt['results']}
    if set(references) != {r['id'] for r in rows}: raise ValueError('Publisher reference case IDs mismatch')
    checkpoint = Path(checkpoint); checkpoint_sha = file_hash(checkpoint)
    torch.set_num_threads(1)
    model, metadata = load_checkpoint(checkpoint, device, training_state=False); model.eval()
    tokenizer = checkpoint.parent / 'tokenizer.json'
    if file_hash(tokenizer) != manifest['tokenizer_sha256'] or metadata['tokenizer_sha256'] != manifest['tokenizer_sha256']:
        raise ValueError('Exercise/model tokenizer mismatch')
    encoder = Encoder(Tokenizer.from_file(str(tokenizer)), model.config, data.parent)
    amp = lambda: torch.autocast('cuda', dtype=torch.bfloat16) if device.startswith('cuda') else nullcontext()
    suites = PublishedSuites(cache); results = []
    try:
        if suites.tools != receipt['toolchain'] or suites.source != receipt['source']:
            raise ValueError('Reference toolchain/source receipt mismatch')
        for row in rows:
            gold = solution(row, suites); suite = suites.suite(row); reference = references[row['id']]
            if (suite['test_sha256'] != reference['test_sha256'] or suite['catch_sha256'] != reference['catch_sha256'] or
                suite['main_sha256'] != reference['main_sha256'] or
                hashlib.sha256(row['response'].encode()).hexdigest() != reference['response_sha256']):
                raise ValueError('Publisher reference/test harness changed')
            names = sorted(gold)
            prompt = encoder.prompt({**row, 'prompt': row['prompt'] + '\nReturn all complete solution files: ' + ', '.join(names) +
                                     '. Format each as File: filename followed by a cpp fenced code block. Return library code only.'})
            tokens = [encoder.bos] + encoder.tokens(prompt); budget = min(2048, model.config.max_text_length - len(tokens))
            if budget < len(encoder.tokens(row['response'])) + 1: raise ValueError('Complete reference cannot fit generation budget')
            ids = torch.tensor([tokens], device=device)
            with amp(): generated = model.generate(ids, max_new_tokens=budget, eos_token_id=encoder.eos)
            response = encoder.tokenizer.decode(generated[0, len(tokens):].tolist())
            try:
                candidate = extract_files(response, names)
                assessment = suites.run(candidate, suite, reference['reference']['inventory'])
            except ValueError as error:
                assessment = {'compilation_succeeded': False, 'published_tests_passed': False, 'format_error': str(error)}
            item = {'id': row['id'], 'group_id': row['group_id'], 'exercise': row['exercise'],
                    'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(), 'generation_budget': budget,
                    'context_tokens': len(tokens), 'raw_response': response, 'solution_files': names,
                    'test_sha256': suite['test_sha256'], 'reference_inventory': reference['reference']['inventory'],
                    'candidate': assessment}
            results.append(item)
            atomic_json(Path(output).with_suffix('.progress.json'), {'completed': len(results), 'examples': len(rows),
                        'checkpoint_sha256': checkpoint_sha, 'results': results, 'goal_achieved': False})
            print('C++ model module', row['exercise'], assessment['published_tests_passed'], flush=True)
        report = {'checkpoint_sha256': checkpoint_sha, 'data_sha256': digest,
                  'manifest_sha256': file_hash(data.parent / 'manifest.json'),
                  'reference_report_sha256': file_hash(reference_report), 'toolchain': suites.tools,
                  'examples': len(results), 'compilation_success_rate': sum(r['candidate']['compilation_succeeded'] for r in results) / len(results),
                  'published_suite_pass_rate': sum(r['candidate']['published_tests_passed'] for r in results) / len(results),
                  'protocol': PROTOCOL, 'results': results, 'goal_achieved': False,
                  'limitations': ['Eighteen reused validation exercises, not independent final repository/API or all-language coverage.',
                                  'No exhaustive prior pretraining/semantic-overlap audit, no efficiency or universal correctness proof.',
                                  'Allowed library/header/preprocessor restrictions may reject otherwise valid implementations.',
                                  'JUnit inventory checks reject early exit/empty tests, but do not prove adversarial same-process score-forgery resistance.',
                                  'Reference targets/test bodies are used for grading only; original task examples and starter interfaces are visible.']}
        if file_hash(checkpoint) != checkpoint_sha: raise ValueError('Model checkpoint changed during assessment')
        seal(report, provenance, files=files, data=data, digest=digest, device=device, tokenizer_sha256=metadata['tokenizer_sha256'])
        atomic_json(Path(output), report); return report
    finally: suites.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='data/exercism-code-v1/valid.jsonl')
    parser.add_argument('--cache', default='data/exercism-source-cache-v1/cpp')
    parser.add_argument('--output', required=True); parser.add_argument('--verify-references', action='store_true')
    parser.add_argument('--checkpoint'); parser.add_argument('--reference-report'); parser.add_argument('--device', default='cuda:2')
    args = parser.parse_args()
    if args.verify_references: verify_references(args.data, args.output, args.cache)
    else:
        if not args.checkpoint or not args.reference_report: parser.error('Specify --checkpoint and --reference-report')
        evaluate(args.checkpoint, args.data, args.output, args.cache, args.reference_report, args.device)


if __name__ == '__main__': main()
