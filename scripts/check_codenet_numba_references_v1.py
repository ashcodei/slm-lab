"""Pinned private JIT checks over original sources selected by their imports."""
import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from slm_lab.codenet_sample_oracles_v1 import SUPPORTED, check_sample
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.standalone_numba_v1 import numba_runtime, prepare_numba, run_numba

ROOT = Path(__file__).resolve().parents[1]


def imports_numba(code):
    tree = ast.parse(code)
    return any((isinstance(node, ast.Import) and any(a.name.split('.')[0] == 'numba' for a in node.names))
        or (isinstance(node, ast.ImportFrom) and (node.module or '').split('.')[0] == 'numba') for node in ast.walk(tree))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pilot-only', action='store_true')
    parser.add_argument('--reports', default='codenet-numba-original-references-v1')
    args = parser.parse_args()
    folder = ROOT / 'reports' / args.reports
    if Path(args.reports).name != args.reports or folder.exists():
        raise ValueError('Fresh owned JIT reference report required')
    maximum_reports = 128 * 1024**2
    if shutil.disk_usage(ROOT).free < 26 * 1024**3 + maximum_reports:
        raise ValueError('Reserve report capacity above persistent planned writers')
    folder.mkdir()
    def status(message, **fields):
        atomic_json(folder / 'status.json', {'status': message, 'updated_at': datetime.now(timezone.utc).isoformat(),
            'dataset_admitted_to_training': False, 'new_model_training_started': False, 'goal_achieved': False, **fields})
    try:
        runtime = numba_runtime()
        full_plan_path = ROOT / 'reports/codenet-complete-reference-checks-v1/plan.json'
        full_selection_path = full_plan_path.with_name('selection.jsonl')
        full_plan = json.loads(full_plan_path.read_text())
        if file_hash(full_selection_path) != full_plan['selection_sha256']:
            raise ValueError('Original fixed full-cohort selection differs')
        python_rows = {r['original_archive_member']: r for r in map(json.loads, full_selection_path.open()) if r['language'] == 'Python'}
        if len(python_rows) != full_plan['selected_by_language']['Python']:
            raise ValueError('Original Python population differs')
        prior_path = ROOT / 'reports/codenet-numpy-reference-check-v1/results.json'
        prior = json.loads(prior_path.read_text())
        prior_members = {r['archive_member'] for r in prior['records']}
        if len(prior_members) != 2 or not prior_members <= set(python_rows):
            raise ValueError('Exact two original Numba-dependent references required')
        pack = ROOT / 'data/codenet-durable-pack-v1'
        manifest_path, rows_path = pack / 'manifest.json', pack / 'original-rows.jsonl.gz'
        manifest = json.loads(manifest_path.read_text())
        if file_hash(rows_path) != manifest['files_sha256']['original-rows.jsonl.gz']:
            raise ValueError('Durable original programme rows differ')
        desc_path = ROOT / 'reports/codenet-description-format-v1/records.jsonl'
        desc_summary_path = desc_path.with_name('summary.json')
        if file_hash(desc_path) != json.loads(desc_summary_path.read_text())['records_sha256']:
            raise ValueError('Original problem/sample export differs')
        descriptions = {r['problem_id']: r for r in map(json.loads, desc_path.open())}
        files = [Path(__file__), ROOT / 'src/slm_lab/standalone_numba_v1.py', ROOT / 'scripts/numba_program_sandbox_v1.py',
            ROOT / 'src/slm_lab/jit_transport_v1.py',
            ROOT / 'src/slm_lab/standalone_program_v1.py', ROOT / 'src/slm_lab/native_code.py',
            ROOT / 'scripts/native_sandbox.py', ROOT / 'src/slm_lab/codenet_sample_oracles_v1.py',
            ROOT / '.tools/numba-runtime-v1.json', Path(runtime['file_manifest']), Path(runtime['receipt']),
            full_plan_path, full_selection_path, prior_path, manifest_path, rows_path, desc_path, desc_summary_path]
        runtime_files = [Path(runtime['package_root']) / r['relative_path'] for r in map(json.loads, Path(runtime['file_manifest']).open())]
        bound = {p: file_hash(p) for p in files + runtime_files}
        states = {p: (p.stat().st_ino, p.stat().st_size, p.stat().st_mtime_ns) for p in bound}
        def guard(full=False):
            if (sum(p.stat().st_size for p in folder.iterdir() if p.is_file()) > maximum_reports
                or shutil.disk_usage(ROOT).free < 512 * 1024**2
                or shutil.disk_usage(tempfile.gettempdir()).free < 12 * 1024**3 + 128 * 1024**2):
                raise ValueError('Reserved JIT reference report/workspace budget exceeded')
            for path, digest in bound.items():
                s = path.stat()
                if (s.st_ino, s.st_size, s.st_mtime_ns) != states[path] or (full and file_hash(path) != digest):
                    raise ValueError('Bound JIT runtime/source changed')
        selected, classifications = set(), Counter()
        with gzip.open(rows_path, 'rt', encoding='utf-8') as rows:
            for line in rows:
                row = json.loads(line)
                member = row['original_archive_member']
                if member not in python_rows:
                    continue
                try:
                    required = imports_numba(row['response'])
                except SyntaxError:
                    classifications['static_import_scan_needs_original_runtime_review'] += 1
                    continue
                classifications['explicit_numba_import' if required else 'no_explicit_numba_import'] += 1
                if required and (not args.pilot_only or member in prior_members):
                    selected.add(member)
        if args.pilot_only and selected != prior_members:
            raise ValueError('Original two-reference JIT selection differs')
        selection_path = folder / 'selection.jsonl'
        with selection_path.open('x') as output:
            for member in sorted(selected):
                output.write(json.dumps(python_rows[member]) + '\n')
        atomic_json(folder / 'plan.json', {'recorded_at': datetime.now(timezone.utc).isoformat(),
            'source_files_sha256': {str(p): bound[p] for p in files},
            'pinned_runtime_versions': runtime['versions'], 'selection_sha256': file_hash(selection_path),
            'original_python_cohort': len(python_rows), 'static_import_classifications': dict(classifications),
            'selected_programmes': len(selected), 'pilot_only': args.pilot_only, 'maximum_cpu_workers': 1,
            'address_space_limit_bytes': 2 * 1024**3, 'cpu_seconds_per_process': 20,
            'wall_seconds_per_sample_including_cold_jit': 30, 'cache_file_size_limit_bytes': 2 * 1024**2,
            'parent_monitored_total_cache_budget_bytes': 64 * 1024**2,
            'maximum_cache_entries': 256, 'maximum_target_descriptors': 32,
            'cache_policy': 'Per-program private temporary cache only; no other filesystem writes, network/process/chmod syscalls denied before target interpreter exec.',
            'selection_rule': 'Original fixed source roles; direct import/from-import of numba found statically. No model or execution outcome used in full-mode selection. Dynamic imports remain outside this profile inventory.',
            'dataset_admitted_to_training': False, 'new_model_training_started': False, 'goal_achieved': False})
        counts, seen = Counter(), set()
        status('checking original Numba references under pinned confined profile', checked_references=0, expected_references=len(selected))
        with gzip.open(rows_path, 'rt', encoding='utf-8') as rows, (folder / 'records.jsonl').open('x') as output:
            for line in rows:
                row = json.loads(line)
                member = row['original_archive_member']
                if member not in selected:
                    continue
                original = python_rows[member]
                for key in ['source_problem_id', 'group_id', 'split', 'original_source_body_sha256']:
                    if row[key] != original[key]:
                        raise ValueError('Original source/role identity differs')
                code = row['response'].encode()
                if member in seen or hashlib.sha256(code).hexdigest() != original['original_source_body_sha256']:
                    raise ValueError('Original complete programme bytes/population differ')
                result = {**original, 'samples': []}
                with tempfile.TemporaryDirectory(prefix='slm-codenet-numba-') as work:
                    constraints = prepare_numba(code, work, runtime)
                    for number, pair in enumerate(descriptions[row['source_problem_id']]['samples'], 1):
                        observed = run_numba(constraints, pair['input'].encode())
                        oracle = None
                        if observed['status'] == 'completed' and row['source_problem_id'] in SUPPORTED:
                            oracle = check_sample(row['source_problem_id'], pair['input'], pair['output'], observed['stdout'].decode(errors='replace'))
                        agrees = oracle['passed'] if oracle is not None else observed['stdout'].split() == pair['output'].encode().split()
                        result['samples'].append({'number': number,
                            **{k: v for k, v in observed.items() if k not in {'stdout', 'stderr'}},
                            'stdout_sha256': hashlib.sha256(observed['stdout']).hexdigest(),
                            'stdout_prefix': observed['stdout'][:1000].decode(errors='replace'),
                            'stderr_tail': observed['stderr'][-2000:].decode(errors='replace'),
                            'source_specific_oracle': oracle, 'sample_agrees': bool(agrees)})
                    result['private_compilation_cache_files'] = sum(p.is_file() for p in (Path(work) / 'jit-cache').rglob('*'))
                if any(s['status'] != 'completed' for s in result['samples']):
                    result['qualification'] = 'runtime_or_original_environment_review_needed'
                elif all(s['sample_agrees'] for s in result['samples']):
                    result['qualification'] = 'all_recognized_samples_agree_under_available_source_rule'
                else:
                    result['qualification'] = 'sample_or_original_oracle_review_needed'
                counts[result['qualification']] += 1
                seen.add(member)
                output.write(json.dumps(result) + '\n')
                output.flush()
                if len(seen) % 10 == 0:
                    guard()
                    status('checking original Numba references under pinned confined profile',
                           checked_references=len(seen), expected_references=len(selected), actual_qualifications=dict(counts))
                    print('checked_original_numba_references', len(seen), 'of', len(selected), flush=True)
        if seen != selected:
            raise ValueError('Original JIT source population incomplete')
        guard(full=True)
        atomic_json(folder / 'summary.json', {'completed_at': datetime.now(timezone.utc).isoformat(),
            'references_checked': len(seen), 'qualification_counts': dict(counts),
            'records_sha256': file_hash(folder / 'records.jsonl'), 'selection_sha256': file_hash(selection_path),
            'plan_sha256': file_hash(folder / 'plan.json'), 'pilot_only': args.pilot_only,
            'dataset_admitted_to_training': False, 'new_model_training_started': False, 'goal_achieved': False,
            'scope': 'Original reference/source/runtime/sample compatibility only. Cold JIT/cache overhead included. No model, hidden-case correctness, efficiency or human preference assessment.'})
        status('original Numba profile checks complete; dataset admission pending', checked_references=len(seen), actual_qualifications=dict(counts))
    except Exception as error:
        status('original JIT reference profile stopped; preserve evidence and inspect', error=str(error))
        raise


if __name__ == '__main__':
    main()
