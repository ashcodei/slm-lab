"""Predeclared, constrained original-reference pilot; never a model score."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.native_code import toolchain
from slm_lab.standalone_program_v1 import compile_cpp_program, prepare_python, run_bounded

ROOT = Path(__file__).resolve().parents[1]
QUOTAS = {'train': 16, 'validation': 8, 'test': 8}
SALT = 'slm-lab:codenet-standalone-reference-pilot:v1'


def original_python3(runtime):
    return runtime == 'Python3' or runtime.startswith(('Python (3.', 'PyPy3 ('))


def main(folder_name='codenet-standalone-reference-pilot-v1', extra_bound_paths=(), previous_selection_path=None):
    folder = ROOT / 'reports' / folder_name
    if folder.exists():
        raise ValueError('Preserve earlier reference executions and selection')
    folder.mkdir()
    def status(message, **fields):
        atomic_json(folder / 'status.json', {'status': message,
            'updated_at': datetime.now(timezone.utc).isoformat(),
            'dataset_admitted_to_training': False, 'new_model_training_started': False,
            'goal_achieved': False, **fields})
    try:
        body_path = ROOT / 'reports/codenet-candidate-bodies-v1/bodies.json'
        body = json.loads(body_path.read_text())
        cache = Path(body['candidate_cache'])
        source_path, index_path = cache / 'source-bodies.bin', cache / 'source-index.jsonl'
        metadata_path = ROOT / 'data/codenet-candidate-selection-v1/selected-metadata.jsonl'
        role_manifest_path = ROOT / 'data/codenet-internal-splits-v1/manifest.json'
        role_path = role_manifest_path.with_name('members.jsonl')
        role_manifest = json.loads(role_manifest_path.read_text())
        description_summary_path = ROOT / 'reports/codenet-description-format-v1/summary.json'
        description_path = description_summary_path.with_name('records.jsonl')
        description_summary = json.loads(description_summary_path.read_text())
        expected = {source_path: body['source_body_store_sha256'], index_path: body['source_index_sha256'],
                    metadata_path: body['selected_metadata_sha256'], role_path: role_manifest['members_sha256'],
                    description_path: description_summary['records_sha256']}
        for path, digest in expected.items():
            if file_hash(path) != digest:
                raise ValueError('Prepared original reference source differs')
        tools = toolchain()
        other = [Path(__file__), ROOT / 'src/slm_lab/standalone_program_v1.py',
                 ROOT / 'scripts/python_program_sandbox_v1.py', ROOT / 'src/slm_lab/native_code.py',
                 ROOT / 'scripts/native_sandbox.py', Path(sys.executable).resolve(),
                 ROOT / '.tools/native-toolchain.json', Path(tools['compiler_path']),
                 body_path, role_manifest_path, description_summary_path]
        other.extend(Path(p) for p in extra_bound_paths)
        if previous_selection_path is not None:
            other.append(Path(previous_selection_path))
        bound = {**expected, **{path: file_hash(path) for path in other}}
        states = {p: (p.stat().st_ino, p.stat().st_size, p.stat().st_mtime_ns) for p in bound}
        def guard(full=False):
            for path, digest in bound.items():
                s = path.stat()
                if (s.st_ino, s.st_size, s.st_mtime_ns) != states[path] or (full and file_hash(path) != digest):
                    raise ValueError('Bound reference source/runtime changed')
        metadata = {row['archive_member']: row['metadata'] for row in map(json.loads, metadata_path.open())}
        indices = {row['archive_member']: row for row in map(json.loads, index_path.open())}
        descriptions = {row['problem_id']: row for row in map(json.loads, description_path.open())}
        candidates = []
        population = Counter()
        for row in map(json.loads, role_path.open()):
            if row['context_qualification'] != 'complete_original_pair_fits' or row['duplicate_complete_example']:
                continue
            language = row['language']
            m = metadata[row['archive_member']]
            if language not in {'C++', 'Python'} or (language == 'Python' and not original_python3(m['original_language'])):
                continue
            desc = descriptions[row['problem_id']]
            if not desc['recognized_samples_usable'] or desc['sample_parse_issues'] or not desc['samples']:
                continue
            if any(len(pair['input'].encode()) > 4 * 1024**2 for pair in desc['samples']):
                continue
            population[(language, row['split'])] += 1
            candidates.append(row)
        selected, groups, counts = [], set(), Counter()
        for row in sorted(candidates, key=lambda r: hashlib.sha256((SALT + '\0' + r['archive_member']).encode()).digest()):
            key = row['language'], row['split']
            group_key = row['language'], row['group_id']
            if counts[key] >= QUOTAS[row['split']] or group_key in groups:
                continue
            groups.add(group_key)
            counts[key] += 1
            selected.append(row)
        if any(counts[(language, role)] != quota for language in ['Python', 'C++'] for role, quota in QUOTAS.items()):
            raise ValueError('Insufficient original reference coverage for the predeclared quotas')
        selection_path = folder / 'selection.jsonl'
        with selection_path.open('x') as output:
            for row in selected:
                desc, index = descriptions[row['problem_id']], indices[row['archive_member']]
                output.write(json.dumps({**row, 'original_runtime': metadata[row['archive_member']]['original_language'],
                    'original_source_body_sha256': index['body_sha256'],
                    'recognized_samples': [{'input_sha256': hashlib.sha256(pair['input'].encode()).hexdigest(),
                        'output_sha256': hashlib.sha256(pair['output'].encode()).hexdigest()} for pair in desc['samples']]}) + '\n')
        if previous_selection_path is not None and file_hash(selection_path) != file_hash(Path(previous_selection_path)):
            raise ValueError('Preserved predeclared original reference selection differs')
        plan = {'recorded_at': datetime.now(timezone.utc).isoformat(),
            'input_files_sha256': {str(p): digest for p, digest in bound.items()},
            'selection_sha256': file_hash(selection_path), 'salt': SALT, 'per_language_role_quotas': QUOTAS,
            'eligible_candidate_population': {language: {role: population[(language, role)] for role in QUOTAS} for language in ['Python', 'C++']},
            'selection_rule': 'Fixed SHA member rank; at most one original program per problem component/language; no execution outcome or model output used.',
            'runtime': {'Python': sys.version, 'C++': tools['version'],
                'Python_policy': 'Isolated/no-site interpreter, read-only standard library without site-packages, Landlock+seccomp before interpreter exec. Recorded trusted bootstrap provides exit/quit via sys.exit, leaves original program bytes unchanged.',
                'C++_policy': 'Isolated ISO C++17/libc++/musl/static, recorded standard-header bits shim; does not recreate source-era GNU judge.'},
            'seconds_per_sample': 3, 'combined_output_bytes_per_sample': 1024**2,
            'judge_scope': 'Original recognized samples only; whitespace-normalized equality is a diagnostic. Float/property/multiple-output or extraction differences require review, never an automatic wrong-program label.',
            'historical_judge_status_is_not_human_preference': True,
            'dataset_admitted_to_training': False, 'new_model_training_started': False, 'goal_achieved': False}
        atomic_json(folder / 'plan.json', plan)
        statistics = Counter()
        results_path = folder / 'records.jsonl'
        with source_path.open('rb') as sources, results_path.open('x') as output:
            for number, row in enumerate(selected, 1):
                guard()
                if shutil.disk_usage(tempfile.gettempdir()).free < 12 * 1024**3 + 200 * 1024**2:
                    raise ValueError('Preserve 12GiB temporary reserve and bounded compiler scratch space')
                index = indices[row['archive_member']]
                sources.seek(index['body_offset'])
                code = sources.read(index['body_bytes'])
                if len(code) != index['body_bytes'] or hashlib.sha256(code).hexdigest() != index['body_sha256']:
                    raise ValueError('Selected original program digest differs')
                record = {key: row[key] for key in ['archive_member', 'problem_id', 'group_id', 'language', 'split', 'publisher_size_outlier']}
                record.update(original_runtime=metadata[row['archive_member']]['original_language'],
                              original_source_body_sha256=index['body_sha256'], samples=[])
                with tempfile.TemporaryDirectory(prefix='slm-codenet-reference-') as work:
                    if row['language'] == 'Python':
                        constraints = prepare_python(code, work)
                    else:
                        constraints, compilation = compile_cpp_program(code, work, tools)
                        record['compilation'] = {key: value for key, value in compilation.items() if key not in {'stdout', 'stderr'}}
                        record['compilation']['diagnostic_tail'] = compilation['stderr'][-4000:].decode(errors='replace')
                    if constraints is None:
                        record['qualification'] = 'compiler_or_source_environment_review_needed'
                    else:
                        for sample_number, pair in enumerate(descriptions[row['problem_id']]['samples'], 1):
                            result = run_bounded(constraints, pair['input'].encode(), python=row['language'] == 'Python')
                            expected = pair['output'].encode()
                            record['samples'].append({
                                'sample': sample_number, **{k: v for k, v in result.items() if k not in {'stdout', 'stderr'}},
                                'stdout_sha256': hashlib.sha256(result['stdout']).hexdigest(),
                                'stderr_sha256': hashlib.sha256(result['stderr']).hexdigest(),
                                'stdout_prefix': result['stdout'][:2000].decode(errors='replace'),
                                'stderr_tail': result['stderr'][-2000:].decode(errors='replace'),
                                'exact_output_equal': result['stdout'] == expected,
                                'whitespace_token_output_equal': result['stdout'].split() == expected.split()})
                        if any(s['status'] != 'completed' for s in record['samples']):
                            record['qualification'] = 'runtime_or_source_environment_review_needed'
                        elif all(s['whitespace_token_output_equal'] for s in record['samples']):
                            record['qualification'] = 'original_recognized_samples_agree_after_whitespace_normalization'
                        else:
                            record['qualification'] = 'sample_output_or_oracle_review_needed'
                statistics[(row['language'], record['qualification'])] += 1
                output.write(json.dumps(record) + '\n')
                output.flush()
                status('executing fixed original references; no model training', checked_references=number,
                       expected_references=len(selected))
                print(number, row['language'], row['problem_id'], record['qualification'], flush=True)
        guard(full=True)
        if file_hash(selection_path) != plan['selection_sha256']:
            raise ValueError('Predeclared source selection changed')
        atomic_json(folder / 'summary.json', {
            'completed_at': datetime.now(timezone.utc).isoformat(), 'checked_references': len(selected),
            'by_language': {language: {q: count for (lang, q), count in statistics.items() if lang == language}
                            for language in ['Python', 'C++']},
            'records_sha256': file_hash(results_path), 'selection_sha256': plan['selection_sha256'],
            'plan_sha256': file_hash(folder / 'plan.json'),
            'dataset_admitted_to_training': False, 'new_model_training_started': False, 'goal_achieved': False,
            'scope': 'Original-reference/runtime/sample-extraction compatibility pilot only. No model was scored; no correctness, hidden-case efficiency, human preference or independent contamination-free evaluation claim.'})
        status('fixed original-reference execution pilot complete; dataset admission pending', checked_references=len(selected))
    except Exception as error:
        status('original-reference pilot stopped; requires inspection', error=str(error))
        raise


if __name__ == '__main__':
    main()
