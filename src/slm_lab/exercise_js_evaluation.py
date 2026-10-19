"""Generate complete held JavaScript modules, then run original publisher suites."""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import torch
from tokenizers import Tokenizer

from .corpus import file_hash
from .diverse_images import atomic_json, jsonl
from .evaluation_provenance import capture, seal
from .exercise_js_runner import PublishedSuites, extract_files, policy, LAUNCHER
from .native_code import NativeBackendError, stop
from .training import Encoder, load_checkpoint

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = 'All37 held JavaScript modules; complete named files,greedy generation up to2048tokens; exact original Jest test bytes with all skipped/focused cases enabled, kernel-confined Node and reference-matched complete test inventories.'


def syntax(files, tools):
    with tempfile.TemporaryDirectory(prefix='slm-js-parse-') as directory:
        work = Path(directory); (work / 'cache').mkdir(); (work / 'package.json').write_text('{"type":"module"}\n')
        for name, source in files.items(): (work / name).write_text(source)
        for name in files:
            command = [tools['node_path'], '--openssl-config=/dev/null', '--max-old-space-size=64', '--check', str(work / name)]
            process = subprocess.Popen([sys.executable, '-I', str(LAUNCHER), json.dumps(policy(work, tools, command))],
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                env={'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8'}, close_fds=True, start_new_session=True)
            try: _, errors = process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                stop(process); return False, 'JavaScript parse deadline exceeded'
            diagnostic = errors.decode(errors='replace')[-1200:]
            if 'SANDBOX_ERROR:' in diagnostic: raise NativeBackendError(diagnostic)
            if process.returncode: return False, diagnostic
    return True, None


def compare_reports(before, after):
    for key in ['evaluation_provenance', 'data_sha256', 'manifest_sha256', 'tokenizer_sha256', 'source_archive_sha256',
                'toolchain_manifest_sha256', 'reference_report_sha256', 'declared_generation_budget', 'protocol',
                'inference_precision', 'inference_device_type']:
        if before[key] != after[key]: raise ValueError('Unmatched JavaScript module assessment: ' + key)
    identities = []
    for report in [before, after]:
        rows = report['results']
        if len(rows) != 37 or len({row['id'] for row in rows}) != 37: raise ValueError('All37 original held JavaScript modules required')
        for key, field in [('full_file_delivery_rate', 'full_file_delivery'), ('javascript_syntax_rate', 'syntax_valid'),
                           ('published_module_pass_rate', 'published_tests_passed')]:
            if report[key] != sum(row[field] for row in rows) / 37: raise ValueError('JavaScript aggregate differs from case evidence')
        identities.append([(row['id'], row['group_id'], row['prompt_sha256'], row['generation_budget'],
                            row['original_test_sha256'], row['expected_solution_files']) for row in rows])
    if identities[0] != identities[1]: raise ValueError('Unmatched JavaScript questions/interfaces/budgets/test bytes')
    lost = [first['id'] for first, second in zip(before['results'], after['results']) if first['published_tests_passed'] and not second['published_tests_passed']]
    passed = not lost and all(after[key] >= before[key] for key in ['full_file_delivery_rate', 'javascript_syntax_rate', 'published_module_pass_rate'])
    return {'passed': passed, 'previously_correct_modules_lost': lost,
            'base_checkpoint_sha256': before['checkpoint_sha256'], 'candidate_checkpoint_sha256': after['checkpoint_sha256'],
            'before_module_pass_rate': before['published_module_pass_rate'], 'after_module_pass_rate': after['published_module_pass_rate'],
            'goal_achieved': False, 'scope': 'Full matched original JavaScript module development suites; no all-language correctness, optimal complexity or independent final claim.'}


@torch.inference_mode()
def evaluate(checkpoint, data, output, device='cuda:2'):
    data, output = Path(data), Path(output)
    if output.exists(): raise ValueError('Preserve existing JavaScript policy assessment')
    data_sha = file_hash(data); manifest_path = data.parent / 'manifest.json'; manifest_sha = file_hash(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if data.name != 'valid.jsonl' or manifest['files_sha256'][data.name] != data_sha: raise ValueError('Use finalized original held exercise data')
    reference_path = ROOT / 'reports/exercism-javascript-publisher-references-v1.json'; reference_sha = file_hash(reference_path)
    references = json.loads(reference_path.read_text())
    if (references['complete_all37_reference_suites_passed'] is not True or references['data_sha256'] != data_sha
            or references['manifest_sha256'] != manifest_sha): raise ValueError('All original JavaScript references must first pass complete publisher suites')
    for name, digest in references['evaluation_provenance']['files_sha256'].items():
        if file_hash(ROOT / name) != digest: raise ValueError('Verified JavaScript reference implementation changed')
    files = list(dict.fromkeys([ROOT / name for name in capture()['files_sha256']] + [Path(__file__),
            *[ROOT / name for name in references['evaluation_provenance']['files_sha256']]]))
    runtime = capture(files); torch.set_num_threads(1)
    if device.startswith('cuda'): torch.cuda.set_per_process_memory_fraction(.2, int(device.split(':')[-1]) if ':' in device else 0)
    checkpoint_sha = file_hash(checkpoint); model, metadata = load_checkpoint(checkpoint, device, training_state=False); model.eval()
    tokenizer_path = Path(checkpoint).parent / 'tokenizer.json'
    if file_hash(tokenizer_path) != manifest['tokenizer_sha256'] or metadata['tokenizer_sha256'] != manifest['tokenizer_sha256']:
        raise ValueError('Own JavaScript policy/tokenizer differs from original dataset')
    encoder = Encoder(Tokenizer.from_file(str(tokenizer_path)), model.config, data.parent, 448)
    rows = [row for row in jsonl(data) if row['language'] == 'JavaScript']; expected = {row['id']: row for row in references['results']}
    if len(rows) != 37 or set(expected) != {row['id'] for row in rows}: raise ValueError('Complete held JavaScript population differs')
    suites = PublishedSuites(ROOT / 'data/exercism-source-cache-v1/javascript'); results = []
    if file_hash(ROOT / '.tools/javascript-toolchain-v1.json') != references['toolchain_manifest_sha256']: raise ValueError('Verified JavaScript toolchain manifest changed')
    amp = lambda: torch.autocast('cuda', dtype=torch.bfloat16) if device.startswith('cuda') else nullcontext()
    try:
        for row in rows:
            original = suites.reference(row); suite = suites.suite(row); reference = expected[row['id']]
            if suite['test_sha256'] != reference['original_test_sha256']: raise ValueError('Original held JavaScript tests differ')
            prompt = encoder.prompt(row); tokens = [encoder.bos] + encoder.tokens(prompt)
            budget = min(2048, model.config.max_text_length - len(tokens))
            if budget < len(encoder.tokens(row['response'])) + 1: raise ValueError('Complete reference module cannot fit fixed generation budget')
            ids = torch.tensor([tokens], device=device)
            with amp(): generated = model.generate(ids, max_new_tokens=budget, eos_token_id=encoder.eos)
            raw = encoder.tokenizer.decode(generated[0, len(tokens):].tolist())
            result = {'id': row['id'], 'group_id': row['group_id'], 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                      'generation_budget': budget, 'original_test_sha256': suite['test_sha256'], 'expected_solution_files': sorted(original),
                      'raw_response': raw, 'full_file_delivery': False, 'syntax_valid': False, 'published_tests_passed': False}
            try: delivered = extract_files(raw, original)
            except ValueError as error: result['delivery_error'] = str(error)
            else:
                result['full_file_delivery'] = True; result['syntax_valid'], result['syntax_error'] = syntax(delivered, suites.tools)
                if result['syntax_valid']:
                    measured = suites.run(delivered, suite, reference['result']['inventory'])
                    result['execution'] = measured; result['published_tests_passed'] = measured['published_tests_passed']
            results.append(result)
            print('Own JavaScript module evaluated', row['exercise'], result['full_file_delivery'], result['syntax_valid'], result['published_tests_passed'], flush=True)
        if file_hash(checkpoint) != checkpoint_sha or file_hash(manifest_path) != manifest_sha or file_hash(reference_path) != reference_sha:
            raise ValueError('Own JavaScript policy/data/reference evidence changed')
        report = {'checkpoint_sha256': checkpoint_sha, 'data_sha256': data_sha, 'manifest_sha256': manifest_sha,
                  'tokenizer_sha256': metadata['tokenizer_sha256'], 'source_archive_sha256': references['source_archive_sha256'],
                  'toolchain_manifest_sha256': references['toolchain_manifest_sha256'], 'reference_report_sha256': reference_sha,
                  'declared_generation_budget': 2048, 'protocol': PROTOCOL, 'results': results,
                  'full_file_delivery_rate': sum(row['full_file_delivery'] for row in results) / 37,
                  'javascript_syntax_rate': sum(row['syntax_valid'] for row in results) / 37,
                  'published_module_pass_rate': sum(row['published_tests_passed'] for row in results) / 37,
                  'goal_achieved': False, 'limitations': ['Reused source-group-held development modules; prior semantic pretraining overlap is not exhaustively audited.',
                    'Published behavior and a bounded heavy case do not prove optimal asymptotic complexity or support all JavaScript.',
                    'Same-process Jest case inventories are not adversarial score-forgery proof; temporary toolchain must be restored exactly if removed.']}
        seal(report, runtime, files, data=data, digest=data_sha, device=device, tokenizer_sha256=metadata['tokenizer_sha256'])
        atomic_json(output, report); return report
    finally: suites.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=ROOT / 'data/exercism-code-v1/valid.jsonl')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:2')
    args = parser.parse_args(); report = evaluate(args.checkpoint, args.data, args.output, args.device)
    print(json.dumps({key: report[key] for key in ['full_file_delivery_rate', 'javascript_syntax_rate', 'published_module_pass_rate']}), flush=True)
