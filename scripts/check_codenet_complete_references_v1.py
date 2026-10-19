"""Bounded two-worker checks over every eligible durable Python/C++ reference."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from check_codenet_standalone_references_v1 import original_python3
from slm_lab.codenet_sample_oracles_v1 import SUPPORTED, check_sample
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.native_code import toolchain
from slm_lab.standalone_numpy_v1 import numpy_runtime_paths, prepare_numpy
from slm_lab.standalone_program_v1 import compile_cpp_program, run_bounded

ROOT = Path(__file__).resolve().parents[1]


def execute_reference(row, problem, tools):
    if shutil.disk_usage(tempfile.gettempdir()).free < 12 * 1024**3 + 300 * 1024**2:
        raise ValueError('Preserve temporary reserve and both compiler workspaces')
    code = row['response'].encode()
    if hashlib.sha256(code).hexdigest() != row['original_source_body_sha256']:
        raise ValueError('Durable original programme digest differs')
    result = {key: row[key] for key in ['original_archive_member', 'source_problem_id', 'group_id', 'split', 'language', 'original_source_body_sha256']}
    result.update(original_runtime=row['original_language'], samples=[])
    with tempfile.TemporaryDirectory(prefix='slm-codenet-all-reference-') as work:
        if row['language'] == 'Python':
            constraints = prepare_numpy(code, work)
        else:
            constraints, compilation = compile_cpp_program(code, work, tools)
            result['compilation'] = {k: v for k, v in compilation.items() if k not in {'stdout', 'stderr'}}
            result['compilation']['diagnostic_tail'] = compilation['stderr'][-2000:].decode(errors='replace')
        if constraints is None:
            result['qualification'] = 'compiler_or_original_environment_review_needed'
            return result
        for number, pair in enumerate(problem['samples'], 1):
            observed = run_bounded(constraints, pair['input'].encode(), python=row['language'] == 'Python')
            oracle = None
            if observed['status'] == 'completed' and row['source_problem_id'] in SUPPORTED:
                oracle = check_sample(row['source_problem_id'], pair['input'], pair['output'], observed['stdout'].decode(errors='replace'))
            agrees = (oracle['passed'] if oracle is not None else observed['stdout'].split() == pair['output'].encode().split())
            result['samples'].append({'number': number,
                **{k: v for k, v in observed.items() if k not in {'stdout', 'stderr'}},
                'stdout_sha256': hashlib.sha256(observed['stdout']).hexdigest(),
                'stdout_prefix': observed['stdout'][:1000].decode(errors='replace'),
                'stderr_tail': observed['stderr'][-1000:].decode(errors='replace'),
                'source_specific_oracle': oracle, 'sample_agrees': bool(agrees)})
    if any(sample['status'] != 'completed' for sample in result['samples']):
        result['qualification'] = 'runtime_or_original_environment_review_needed'
    elif all(sample['sample_agrees'] for sample in result['samples']):
        result['qualification'] = 'all_recognized_samples_agree_under_available_source_rule'
    else:
        result['qualification'] = 'sample_or_original_oracle_review_needed'
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reports', default='codenet-complete-reference-checks-v1')
    parser.add_argument('--limit-per-language', type=int, default=0)
    args = parser.parse_args()
    if Path(args.reports).name != args.reports or args.limit_per_language < 0:
        raise ValueError('Fresh report name and nonnegative explicit smoke limit required')
    folder = ROOT / 'reports' / args.reports
    if folder.exists():
        raise ValueError('Preserve earlier full-cohort execution evidence')
    # Reserve this job's entire report budget before it starts. Later model
    # checkpoint writes consume their separately reserved capacity; transient
    # drops below the original 26GiB must not invalidate a tiny report commit.
    maximum_report_bytes = 256 * 1024**2
    if shutil.disk_usage(ROOT).free < 26 * 1024**3 + maximum_report_bytes:
        raise ValueError('Require report budget above the persistent writer reserve')
    folder.mkdir()
    def status(message, **fields):
        atomic_json(folder / 'status.json', {'status': message, 'updated_at': datetime.now(timezone.utc).isoformat(),
            'dataset_admitted_to_training': False, 'new_model_training_started': False, 'goal_achieved': False, **fields})
    try:
        pack = ROOT / 'data/codenet-durable-pack-v1'
        manifest_path, rows_path = pack / 'manifest.json', pack / 'original-rows.jsonl.gz'
        audit_path = ROOT / 'reports/codenet-durable-pack-v1/reconstruction-check.json'
        manifest, audit = json.loads(manifest_path.read_text()), json.loads(audit_path.read_text())
        if (audit['source_files_sha256'][str(manifest_path)] != file_hash(manifest_path)
            or audit['original_complete_examples_reconstructed'] != manifest['source_examples']
            or file_hash(rows_path) != manifest['files_sha256']['original-rows.jsonl.gz']):
            raise ValueError('Completed durable source reconstruction required')
        desc_path = ROOT / 'reports/codenet-description-format-v1/records.jsonl'
        desc_summary_path = desc_path.with_name('summary.json')
        if file_hash(desc_path) != json.loads(desc_summary_path.read_text())['records_sha256']:
            raise ValueError('Original description/sample export differs')
        descriptions = {row['problem_id']: row for row in map(json.loads, desc_path.open())}
        tools = toolchain()
        files = [Path(__file__), ROOT / 'scripts/check_codenet_standalone_references_v1.py',
            ROOT / 'src/slm_lab/standalone_program_v1.py', ROOT / 'src/slm_lab/standalone_numpy_v1.py',
            ROOT / 'src/slm_lab/codenet_sample_oracles_v1.py', ROOT / 'src/slm_lab/native_code.py',
            ROOT / 'scripts/native_sandbox.py', ROOT / 'scripts/python_program_sandbox_v1.py',
            Path(tools['compiler_path']), ROOT / '.tools/native-toolchain.json',
            manifest_path, rows_path, audit_path, desc_path, desc_summary_path]
        runtime_files = sorted({p.resolve() for directory in numpy_runtime_paths() for p in directory.rglob('*') if p.is_file()})
        bound = {path: file_hash(path) for path in files + runtime_files}
        states = {p: (p.stat().st_ino, p.stat().st_size, p.stat().st_mtime_ns) for p in bound}
        def guard(full=False):
            if (sum(p.stat().st_size for p in folder.iterdir() if p.is_file()) > maximum_report_bytes
                or shutil.disk_usage(ROOT).free < 512 * 1024**2):
                raise ValueError('Reserved report budget exceeded or filesystem almost full')
            for path, digest in bound.items():
                s = path.stat()
                if (s.st_ino, s.st_size, s.st_mtime_ns) != states[path] or (full and file_hash(path) != digest):
                    raise ValueError('Bound complete-reference runtime/source changed')
        eligible, population, chosen_counts = set(), Counter(), Counter()
        selection_path = folder / 'selection.jsonl'
        with gzip.open(rows_path, 'rt', encoding='utf-8') as source, selection_path.open('x') as output:
            for line in source:
                row = json.loads(line)
                language, problem = row['language'], descriptions[row['source_problem_id']]
                if language not in {'C++', 'Python'} or (language == 'Python' and not original_python3(row['original_language'])):
                    population['original_runtime_not_configured'] += 1
                    continue
                if not problem['recognized_samples_usable'] or problem['sample_parse_issues'] or not problem['samples']:
                    population['original_sample_format_review_needed'] += 1
                    continue
                if any(len(pair['input'].encode()) > 4 * 1024**2 for pair in problem['samples']):
                    population['original_sample_exceeds_input_budget'] += 1
                    continue
                population['eligible_' + language] += 1
                if args.limit_per_language and chosen_counts[language] >= args.limit_per_language:
                    continue
                member = row['original_archive_member']
                if member in eligible:
                    raise ValueError('Repeated durable programme source')
                eligible.add(member)
                chosen_counts[language] += 1
                output.write(json.dumps({key: row[key] for key in ['original_archive_member', 'source_problem_id', 'group_id', 'split', 'language', 'original_language', 'original_source_body_sha256']}) + '\n')
        runtime_manifest = folder / 'numpy-files.jsonl'
        with runtime_manifest.open('x') as output:
            for path in runtime_files:
                output.write(json.dumps({'path': str(path), 'sha256': bound[path]}) + '\n')
        atomic_json(folder / 'plan.json', {'recorded_at': datetime.now(timezone.utc).isoformat(),
            'source_files_sha256': {str(p): bound[p] for p in files},
            'numpy_runtime_files_sha256': file_hash(runtime_manifest),
            'selection_sha256': file_hash(selection_path), 'original_candidate_population': dict(population),
            'selected_by_language': dict(chosen_counts), 'total_selected': len(eligible),
            'explicit_smoke_limit_per_language': args.limit_per_language, 'maximum_workers': 2,
            'maximum_persistent_report_bytes': maximum_report_bytes,
            'initial_free_capacity_gib_required': 26.25, 'minimum_temporary_free_gib': 12,
            'source_role_policy': 'Keep every original sealed role; execute original references from all roles without model updates. No execution outcome used to change source roles.',
            'qualification_policy': 'Compilation/runtime/sample-rule differences require review, never automatic wrong-code or preference labels. NumPy is exposed; Numba and GNU extensions remain qualified.',
            'dataset_admitted_to_training': False, 'new_model_training_started': False, 'goal_achieved': False})
        status('checking complete original programme cohort with two bounded CPU workers', checked_references=0, expected_references=len(eligible))
        checked, counts, seen, futures = 0, Counter(), set(), set()
        with ThreadPoolExecutor(max_workers=2) as pool, (folder / 'records.jsonl').open('x') as output:
            def collect(done):
                nonlocal checked
                for future in done:
                    record = future.result()
                    member = record['original_archive_member']
                    if member in seen or member not in eligible:
                        raise ValueError('Executed reference population differs')
                    seen.add(member)
                    checked += 1
                    counts[record['qualification']] += 1
                    output.write(json.dumps(record) + '\n')
                    output.flush()
                    if checked % 25 == 0:
                        guard()
                        status('checking complete original programme cohort with two bounded CPU workers',
                            checked_references=checked, expected_references=len(eligible), actual_qualifications=dict(counts))
                        print('checked_references', checked, 'of', len(eligible), flush=True)
            with gzip.open(rows_path, 'rt', encoding='utf-8') as source:
                for line in source:
                    row = json.loads(line)
                    if row['original_archive_member'] not in eligible:
                        continue
                    futures.add(pool.submit(execute_reference, row, descriptions[row['source_problem_id']], tools))
                    if len(futures) >= 4:
                        done, futures = wait(futures, return_when=FIRST_COMPLETED)
                        collect(done)
            while futures:
                done, futures = wait(futures, return_when=FIRST_COMPLETED)
                collect(done)
        if seen != eligible:
            raise ValueError('Original reference cohort incomplete')
        guard(full=True)
        atomic_json(folder / 'summary.json', {'completed_at': datetime.now(timezone.utc).isoformat(),
            'references_checked': checked, 'qualification_counts': dict(counts),
            'records_sha256': file_hash(folder / 'records.jsonl'), 'plan_sha256': file_hash(folder / 'plan.json'),
            'selection_sha256': file_hash(selection_path), 'dataset_admitted_to_training': False,
            'new_model_training_started': False, 'goal_achieved': False,
            'scope': 'Complete eligible original-reference cohort only; smoke cap if explicitly recorded. Original recognised samples and limited problem-specific checks, not hidden cases, model quality, efficiency or human preferences.'})
        status('complete original-reference cohort checked; dataset admission still requires inspection', checked_references=checked, actual_qualifications=dict(counts))
    except Exception as error:
        status('original-reference cohort stopped; preserve evidence and inspect', error=str(error))
        raise


if __name__ == '__main__':
    main()
