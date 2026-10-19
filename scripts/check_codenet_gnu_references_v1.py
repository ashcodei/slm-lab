"""Independent GNU profile over the same fixed original C++ reference cohort."""
import argparse
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
from slm_lab.standalone_gnu_v1 import gnu_toolchain, compile_gnu_program
from slm_lab.standalone_program_v1 import run_bounded

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pilot-only', action='store_true')
    parser.add_argument('--reports', default='codenet-gnu-original-references-v1')
    args = parser.parse_args()
    folder = ROOT / 'reports' / args.reports
    if Path(args.reports).name != args.reports or folder.exists():
        raise ValueError('Fresh owned reference report folder required')
    maximum_reports = 256 * 1024**2
    if shutil.disk_usage(ROOT).free < 26 * 1024**3 + maximum_reports:
        raise ValueError('Reserve this report budget above the persistent writers')
    folder.mkdir()
    def status(message, **fields):
        atomic_json(folder / 'status.json', {'status': message, 'updated_at': datetime.now(timezone.utc).isoformat(),
            'dataset_admitted_to_training': False, 'new_model_training_started': False, 'goal_achieved': False, **fields})
    try:
        tools = gnu_toolchain()
        full_plan_path = ROOT / 'reports/codenet-complete-reference-checks-v1/plan.json'
        full_selection_path = full_plan_path.with_name('selection.jsonl')
        full_plan = json.loads(full_plan_path.read_text())
        if file_hash(full_selection_path) != full_plan['selection_sha256']:
            raise ValueError('Predeclared full source cohort differs')
        all_cpp = {row['original_archive_member']: row for row in map(json.loads, full_selection_path.open()) if row['language'] == 'C++'}
        if len(all_cpp) != full_plan['selected_by_language']['C++']:
            raise ValueError('Complete original C++ population differs')
        prior_path = ROOT / 'reports/codenet-standalone-reference-pilot-v2/records.jsonl'
        prior_summary_path = prior_path.with_name('summary.json')
        if file_hash(prior_path) != json.loads(prior_summary_path.read_text())['records_sha256']:
            raise ValueError('Preserved original pilot differs')
        wanted = set(all_cpp)
        if args.pilot_only:
            wanted = {row['archive_member'] for row in map(json.loads, prior_path.open())
                      if row['language'] == 'C++' and row['qualification'] == 'compiler_or_source_environment_review_needed'}
            if len(wanted) != 2 or not wanted <= set(all_cpp):
                raise ValueError('Exact two original compiler qualifications required')
        pack = ROOT / 'data/codenet-durable-pack-v1'
        manifest_path, row_path = pack / 'manifest.json', pack / 'original-rows.jsonl.gz'
        manifest = json.loads(manifest_path.read_text())
        if file_hash(row_path) != manifest['files_sha256']['original-rows.jsonl.gz']:
            raise ValueError('Durable complete source rows differ')
        desc_path = ROOT / 'reports/codenet-description-format-v1/records.jsonl'
        desc_summary_path = desc_path.with_name('summary.json')
        if file_hash(desc_path) != json.loads(desc_summary_path.read_text())['records_sha256']:
            raise ValueError('Original description/sample export differs')
        descriptions = {row['problem_id']: row for row in map(json.loads, desc_path.open())}
        files = [Path(__file__), ROOT / 'src/slm_lab/standalone_gnu_v1.py',
            ROOT / 'src/slm_lab/standalone_program_v1.py', ROOT / 'src/slm_lab/codenet_sample_oracles_v1.py',
            ROOT / 'src/slm_lab/native_code.py', ROOT / 'scripts/native_sandbox.py',
            ROOT / '.tools/gnu-toolchain-v1.json', Path(tools['file_manifest']), Path(tools['package_receipt']),
            full_plan_path, full_selection_path, prior_path, prior_summary_path,
            manifest_path, row_path, desc_path, desc_summary_path]
        runtime_files = [Path(tools['toolchain_root']) / entry['relative_path']
                         for entry in map(json.loads, Path(tools['file_manifest']).open())]
        bound = {p: file_hash(p) for p in files + runtime_files}
        states = {p: (p.stat().st_ino, p.stat().st_size, p.stat().st_mtime_ns) for p in bound}
        def guard(full=False):
            if (sum(p.stat().st_size for p in folder.iterdir() if p.is_file()) > maximum_reports
                or shutil.disk_usage(ROOT).free < 512 * 1024**2
                or shutil.disk_usage(tempfile.gettempdir()).free < 12 * 1024**3 + 32 * 1024**2):
                raise ValueError('Reserved reference report or temporary workspace budget exceeded')
            for path, digest in bound.items():
                s = path.stat()
                if (s.st_ino, s.st_size, s.st_mtime_ns) != states[path] or (full and file_hash(path) != digest):
                    raise ValueError('Bound original GNU profile/source changed')
        selection_path = folder / 'selection.jsonl'
        with selection_path.open('x') as output:
            for member in sorted(wanted):
                output.write(json.dumps(all_cpp[member]) + '\n')
        atomic_json(folder / 'plan.json', {'recorded_at': datetime.now(timezone.utc).isoformat(),
            'source_files_sha256': {str(p): bound[p] for p in files},
            'compiler_version': tools['version'], 'selection_sha256': file_hash(selection_path),
            'complete_original_cpp_population': len(all_cpp), 'selected_programmes': len(wanted),
            'pilot_only_original_two_compiler_qualifications': args.pilot_only,
            'maximum_cpu_workers': 1, 'maximum_persistent_report_bytes': maximum_reports,
            'source_roles_unchanged': True, 'source_programme_bytes_unchanged': True,
            'dataset_admitted_to_training': False, 'new_model_training_started': False, 'goal_achieved': False,
            'scope': 'Additional GNU profile. All predeclared C++ sources in full mode, independent of previous execution outcomes. Original recognized samples and available source-specific checkers only; no model, hidden-case, efficiency or preference assessment.'})
        counts, seen = Counter(), set()
        status('checking original references with isolated verified GNU profile', checked_references=0, expected_references=len(wanted))
        with gzip.open(row_path, 'rt', encoding='utf-8') as rows, (folder / 'records.jsonl').open('x') as output:
            for line in rows:
                row = json.loads(line)
                member = row['original_archive_member']
                if member not in wanted:
                    continue
                if member in seen:
                    raise ValueError('Repeated original GNU source')
                selected = all_cpp[member]
                for key in ['source_problem_id', 'group_id', 'split', 'original_source_body_sha256']:
                    if row[key] != selected[key]:
                        raise ValueError('Original role/programme identity differs')
                code = row['response'].encode()
                if hashlib.sha256(code).hexdigest() != selected['original_source_body_sha256']:
                    raise ValueError('Complete original programme bytes differ')
                result = {**selected, 'samples': []}
                with tempfile.TemporaryDirectory(prefix='slm-codenet-gnu-') as work:
                    constraints, compilation = compile_gnu_program(code, work, tools)
                    result['compilation'] = {k: v for k, v in compilation.items() if k not in {'stdout', 'stderr'}}
                    result['compilation']['diagnostic_tail'] = compilation['stderr'][-2000:].decode(errors='replace')
                    if constraints is None:
                        result['qualification'] = 'compiler_or_original_environment_review_needed'
                    else:
                        for number, pair in enumerate(descriptions[row['source_problem_id']]['samples'], 1):
                            observed = run_bounded(constraints, pair['input'].encode())
                            oracle = None
                            if observed['status'] == 'completed' and row['source_problem_id'] in SUPPORTED:
                                oracle = check_sample(row['source_problem_id'], pair['input'], pair['output'], observed['stdout'].decode(errors='replace'))
                            agrees = oracle['passed'] if oracle is not None else observed['stdout'].split() == pair['output'].encode().split()
                            result['samples'].append({'number': number,
                                **{k: v for k, v in observed.items() if k not in {'stdout', 'stderr'}},
                                'stdout_sha256': hashlib.sha256(observed['stdout']).hexdigest(),
                                'stdout_prefix': observed['stdout'][:1000].decode(errors='replace'),
                                'stderr_tail': observed['stderr'][-1000:].decode(errors='replace'),
                                'source_specific_oracle': oracle, 'sample_agrees': bool(agrees)})
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
                if len(seen) % 25 == 0:
                    guard()
                    status('checking original references with isolated verified GNU profile',
                           checked_references=len(seen), expected_references=len(wanted), actual_qualifications=dict(counts))
                    print('checked_original_gnu_references', len(seen), 'of', len(wanted), flush=True)
        if seen != wanted:
            raise ValueError('Original GNU reference population incomplete')
        guard(full=True)
        atomic_json(folder / 'summary.json', {'completed_at': datetime.now(timezone.utc).isoformat(),
            'references_checked': len(seen), 'qualification_counts': dict(counts),
            'records_sha256': file_hash(folder / 'records.jsonl'), 'plan_sha256': file_hash(folder / 'plan.json'),
            'selection_sha256': file_hash(selection_path), 'pilot_only': args.pilot_only,
            'dataset_admitted_to_training': False, 'new_model_training_started': False, 'goal_achieved': False})
        status('original GNU reference profile complete; dataset admission pending', checked_references=len(seen), actual_qualifications=dict(counts))
    except Exception as error:
        status('original GNU reference profile stopped; inspect preserved evidence', error=str(error))
        raise


if __name__ == '__main__':
    main()
