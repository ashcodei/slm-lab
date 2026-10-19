"""Generate complete original held Rust modules and execute publisher tests."""
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
from .exercise_rust_runner import PublishedSuites, extract_files
from .training import Encoder, load_checkpoint

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = 'All24 original held Rust modules; complete named files,greedy generation up to2048tokens; exact complete publisher Rust tests/module files, Rust2024/musl pre-exec confined compilation/runtime, included ignored/expected-panic cases and reference-matched full test inventory; no benchmark claim.'


def compare_reports(before, after):
    keys = ['evaluation_provenance', 'data_sha256', 'manifest_sha256', 'tokenizer_sha256', 'source_archive_sha256',
            'toolchain_manifest_sha256', 'reference_report_sha256', 'kernel_report_sha256', 'declared_generation_budget', 'protocol',
            'inference_precision', 'inference_device_type']
    for key in keys:
        if before[key] != after[key]: raise ValueError('Unmatched full Rust module assessment: ' + key)
    identity = []
    for report in (before, after):
        rows = report['results']
        if len(rows) != 24 or len({row['id'] for row in rows}) != 24: raise ValueError('All24 original held Rust modules required')
        for row in rows:
            flags = [row[field] for field in ('full_file_delivery', 'compilation_passed', 'published_tests_passed')]
            if any(type(flag) is not bool for flag in flags) or (flags[2] and not all(flags)) or (flags[1] and not flags[0]):
                raise ValueError('Rust delivery/compilation/behavior evidence is inconsistent')
            if flags[2]:
                execution = row.get('execution', {})
                inventory = execution.get('inventory', {})
                if (execution.get('published_tests_passed') is not True
                        or execution.get('reference_inventory_matched') is not True
                        or inventory.get('all_passed') is not True
                        or not inventory.get('cases') or inventory.get('cases') != inventory.get('passed_cases')
                        or inventory.get('skipped') != []
                        or any(item.get('returncode') != 0 or item.get('limit') for item in execution.get('execution', []))
                        or not execution.get('execution')):
                    raise ValueError('Successful Rust module lacks a complete native test inventory')
        for key, field in [('full_file_delivery_rate', 'full_file_delivery'), ('compilation_success_rate', 'compilation_passed'),
                           ('published_module_pass_rate', 'published_tests_passed')]:
            if report[key] != sum(row[field] for row in rows) / 24: raise ValueError('Rust aggregate differs from native case evidence')
        identity.append([(row['id'], row['group_id'], row['prompt_sha256'], row['generation_budget'],
                          row['original_test_sha256'], row['expected_solution_files']) for row in rows])
    if identity[0] != identity[1]: raise ValueError('Unmatched original Rust prompt/interface/test/budget identities')
    lost = [first['id'] for first, second in zip(before['results'], after['results'])
            if first['published_tests_passed'] and not second['published_tests_passed']]
    passed = not lost and all(after[key] >= before[key] for key in
        ('full_file_delivery_rate', 'compilation_success_rate', 'published_module_pass_rate'))
    return {'passed': passed, 'previously_correct_modules_lost': lost,
        'base_checkpoint_sha256': before['checkpoint_sha256'], 'candidate_checkpoint_sha256': after['checkpoint_sha256'],
        'before_module_pass_rate': before['published_module_pass_rate'], 'after_module_pass_rate': after['published_module_pass_rate'],
        'goal_achieved': False, 'scope': 'Matched full Rust publisher development suites; no all-code or efficiency optimality claim.'}


@torch.inference_mode()
def evaluate(checkpoint, data, output, device='cuda:2'):
    checkpoint, data, output = Path(checkpoint), Path(data), Path(output)
    if output.exists(): raise ValueError('Preserve an existing Rust policy assessment')
    torch.set_num_threads(1)
    manifest_path = data.parent / 'manifest.json'; manifest_sha = file_hash(manifest_path); data_sha = file_hash(data)
    manifest = json.loads(manifest_path.read_text())
    if data.name != 'valid.jsonl' or manifest['files_sha256'][data.name] != data_sha: raise ValueError('Finalized held Rust source data required')
    kernel_path = ROOT / 'reports/rust-kernel-and-report-verification-v1.json'
    kernel = json.loads(kernel_path.read_text()); kernel_sha = file_hash(kernel_path)
    if kernel.get('passed') is not True or kernel.get('actual_probe_count') != 11:
        raise ValueError('All11 actual Rust kernel/report probes must pass first')
    for name, digest in kernel['evaluation_provenance']['files_sha256'].items():
        if file_hash(ROOT / name) != digest: raise ValueError('Verified Rust kernel runtime changed')
    ref_path = ROOT / 'reports/exercism-rust-publisher-references-v2.json'; ref_sha = file_hash(ref_path)
    refs = json.loads(ref_path.read_text())
    if (refs['complete_all24_reference_suites_passed'] is not True or refs['passed'] != 24
            or refs['data_sha256'] != data_sha or refs['manifest_sha256'] != manifest_sha):
        raise ValueError('All original Rust references must first pass complete publisher suites')
    for name, digest in refs['evaluation_provenance']['files_sha256'].items():
        if file_hash(ROOT / name) != digest: raise ValueError('Verified Rust reference runtime changed')
    if file_hash(ROOT / '.tools/rust-toolchain-v1.json') != refs['toolchain_manifest_sha256']:
        raise ValueError('Verified Rust toolchain changed')
    files = list(dict.fromkeys([ROOT / name for name in capture()['files_sha256']]
        + [Path(__file__)] + [ROOT / name for name in refs['evaluation_provenance']['files_sha256']]
        + [ROOT / name for name in kernel['evaluation_provenance']['files_sha256']]))
    runtime = capture(files)
    if device.startswith('cuda'): torch.cuda.set_per_process_memory_fraction(.2, int(device.split(':')[-1]) if ':' in device else 0)
    checkpoint_sha = file_hash(checkpoint); model, metadata = load_checkpoint(checkpoint, device, training_state=False); model.eval()
    tokenizer_path = checkpoint.parent / 'tokenizer.json'
    if file_hash(tokenizer_path) != manifest['tokenizer_sha256'] or metadata['tokenizer_sha256'] != manifest['tokenizer_sha256']:
        raise ValueError('Own Rust policy/tokenizer differs from original dataset')
    encoder = Encoder(Tokenizer.from_file(str(tokenizer_path)), model.config, data.parent, 448)
    rows = [row for row in jsonl(data) if row['language'] == 'Rust']; expected = {row['id']: row for row in refs['results']}
    if len(rows) != 24 or set(expected) != {row['id'] for row in rows}: raise ValueError('Complete original Rust module population differs')
    suites = PublishedSuites(ROOT / 'data/exercism-source-cache-v1/rust'); results = []
    amp = lambda: torch.autocast('cuda', dtype=torch.bfloat16) if device.startswith('cuda') else nullcontext()
    try:
        for row in rows:
            original = suites.reference(row); suite = suites.suite(row); reference = expected[row['id']]
            if suite['test_sha256'] != reference['original_test_sha256']: raise ValueError('Complete original Rust test/module files differ')
            prompt = encoder.prompt(row); tokens = [encoder.bos] + encoder.tokens(prompt)
            budget = min(2048, model.config.max_text_length - len(tokens))
            if budget < len(encoder.tokens(row['response'])) + 1: raise ValueError('Complete Rust reference cannot fit fixed generation allocation')
            ids = torch.tensor([tokens], device=device)
            with amp(): generated = model.generate(ids, max_new_tokens=budget, eos_token_id=encoder.eos)
            raw = encoder.tokenizer.decode(generated[0, len(tokens):].tolist())
            result = {'id': row['id'], 'group_id': row['group_id'], 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                'generation_budget': budget, 'original_test_sha256': suite['test_sha256'], 'expected_solution_files': sorted(original),
                'raw_response': raw, 'full_file_delivery': False, 'compilation_passed': False, 'published_tests_passed': False}
            try: delivered = extract_files(raw, original)
            except ValueError as error: result['delivery_error'] = str(error)
            else:
                result['full_file_delivery'] = True
                measured = suites.run(delivered, suite, reference['result']['inventory'])
                result['execution'] = measured; result['compilation_passed'] = measured['compilation_passed']
                result['published_tests_passed'] = measured['published_tests_passed']
            results.append(result)
            print('Own Rust module evaluated', row['exercise'], result['full_file_delivery'], result['compilation_passed'], result['published_tests_passed'], flush=True)
        if file_hash(checkpoint) != checkpoint_sha or file_hash(manifest_path) != manifest_sha or file_hash(ref_path) != ref_sha or file_hash(kernel_path) != kernel_sha:
            raise ValueError('Own Rust policy or source/reference evidence changed')
        report = {'checkpoint_sha256': checkpoint_sha, 'data_sha256': data_sha, 'manifest_sha256': manifest_sha,
            'tokenizer_sha256': metadata['tokenizer_sha256'], 'source_archive_sha256': refs['source_archive_sha256'],
            'toolchain_manifest_sha256': refs['toolchain_manifest_sha256'], 'reference_report_sha256': ref_sha,
            'kernel_report_sha256': kernel_sha,
            'declared_generation_budget': 2048, 'protocol': PROTOCOL, 'results': results,
            'full_file_delivery_rate': sum(row['full_file_delivery'] for row in results) / 24,
            'compilation_success_rate': sum(row['compilation_passed'] for row in results) / 24,
            'published_module_pass_rate': sum(row['published_tests_passed'] for row in results) / 24,
            'goal_achieved': False, 'limitations': ['Reused source-group-held development modules; semantic prior exposure is not exhaustively audited.',
                'Successful publisher tests do not prove universal Rust, asymptotic optimality or all-repository engineering.',
                'Same-process Rust test event inventories are not adversarial score-forgery proof; exact temporary compiler/library identities must be restored if removed. Only pinned provided crates and original integration suites are supported.']}
        seal(report, runtime, files, data=data, digest=data_sha, device=device, tokenizer_sha256=metadata['tokenizer_sha256'])
        atomic_json(output, report); return report
    finally: suites.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=ROOT / 'data/exercism-code-v1/valid.jsonl')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:2')
    args = parser.parse_args(); result = evaluate(args.checkpoint, args.data, args.output, args.device)
    print(json.dumps({key: result[key] for key in ('full_file_delivery_rate', 'compilation_success_rate', 'published_module_pass_rate')}), flush=True)
