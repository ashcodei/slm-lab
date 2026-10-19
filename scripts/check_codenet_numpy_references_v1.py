"""Recheck only the original pilot's NumPy-dependent human references."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.standalone_numpy_v1 import numpy_runtime_paths, prepare_numpy
from slm_lab.standalone_program_v1 import run_bounded

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT / 'reports/codenet-numpy-reference-check-v1'
    if folder.exists():
        raise ValueError('Preserve earlier dependency checks')
    folder.mkdir()
    prior = ROOT / 'reports/codenet-standalone-reference-pilot-v2/records.jsonl'
    summary = json.loads(prior.with_name('summary.json').read_text())
    if file_hash(prior) != summary['records_sha256']:
        raise ValueError('Original pilot bytes differ')
    selected = [row for row in map(json.loads, prior.open()) if row['qualification'] == 'runtime_or_source_environment_review_needed']
    if len(selected) != 2 or any(row['language'] != 'Python' or not all("No module named 'numpy'" in sample['stderr_tail'] for sample in row['samples']) for row in selected):
        raise ValueError('Expected exact two original NumPy runtime qualifications')
    body_path = ROOT / 'reports/codenet-candidate-bodies-v1/bodies.json'
    body = json.loads(body_path.read_text())
    cache = Path(body['candidate_cache'])
    source_path, index_path = cache / 'source-bodies.bin', cache / 'source-index.jsonl'
    if file_hash(source_path) != body['source_body_store_sha256'] or file_hash(index_path) != body['source_index_sha256']:
        raise ValueError('Original programme byte store differs')
    desc_path = ROOT / 'reports/codenet-description-format-v1/records.jsonl'
    desc_summary = json.loads(desc_path.with_name('summary.json').read_text())
    if file_hash(desc_path) != desc_summary['records_sha256']:
        raise ValueError('Original sample source differs')
    descriptions = {row['problem_id']: row for row in map(json.loads, desc_path.open())}
    indices = {row['archive_member']: row for row in map(json.loads, index_path.open())}
    paths = [Path(__file__), ROOT / 'src/slm_lab/standalone_numpy_v1.py',
        ROOT / 'src/slm_lab/standalone_program_v1.py', ROOT / 'scripts/python_program_sandbox_v1.py',
        ROOT / 'scripts/native_sandbox.py', ROOT / 'reports/standalone-numpy-v1-focused-tests.log',
        prior, prior.with_name('summary.json'), desc_path, body_path, source_path, index_path]
    runtime_files = sorted({p.resolve() for root in numpy_runtime_paths() for p in root.rglob('*') if p.is_file()})
    hashes = {path: file_hash(path) for path in paths + runtime_files}
    states = {path: (path.stat().st_ino, path.stat().st_size, path.stat().st_mtime_ns) for path in hashes}
    runtime_manifest = folder / 'numpy-files.jsonl'
    with runtime_manifest.open('x') as output:
        for path in runtime_files:
            output.write(json.dumps({'path': str(path), 'sha256': hashes[path]}) + '\n')
    atomic_json(folder / 'plan.json', {'recorded_at': datetime.now(timezone.utc).isoformat(),
        'source_files_sha256': {str(p): hashes[p] for p in paths},
        'numpy_version': np.__version__, 'numpy_runtime_files': len(runtime_files),
        'numpy_runtime_manifest_sha256': file_hash(runtime_manifest),
        'original_reference_selection': [row['archive_member'] for row in selected],
        'qualification': 'Current NumPy package and its bundled libraries only, isolated/no-site interpreter, single BLAS thread, prior pre-exec confinement unchanged. Does not recreate the source-era NumPy version.',
        'focused_dependency_checks_passed': 2, 'dataset_admitted_to_training': False,
        'new_model_training_started': False, 'goal_achieved': False})
    records = []
    with source_path.open('rb') as source:
        for selected_row in selected:
            index = indices[selected_row['archive_member']]
            source.seek(index['body_offset'])
            code = source.read(index['body_bytes'])
            if hashlib.sha256(code).hexdigest() != index['body_sha256'] or index['body_sha256'] != selected_row['original_source_body_sha256']:
                raise ValueError('Original selected programme differs')
            record = {key: selected_row[key] for key in ['archive_member', 'problem_id', 'group_id', 'language', 'split', 'original_source_body_sha256']}
            record['samples'] = []
            with tempfile.TemporaryDirectory(prefix='slm-codenet-numpy-') as work:
                constraints = prepare_numpy(code, work)
                for number, pair in enumerate(descriptions[record['problem_id']]['samples'], 1):
                    result = run_bounded(constraints, pair['input'].encode(), python=True)
                    record['samples'].append({'sample': number,
                        **{k: v for k, v in result.items() if k not in {'stdout', 'stderr'}},
                        'stdout_sha256': hashlib.sha256(result['stdout']).hexdigest(),
                        'stdout_prefix': result['stdout'][:2000].decode(errors='replace'),
                        'stderr_tail': result['stderr'][-2000:].decode(errors='replace'),
                        'whitespace_token_output_equal': result['stdout'].split() == pair['output'].encode().split()})
            record['all_recognized_samples_agree'] = all(sample['status'] == 'completed' and sample['whitespace_token_output_equal'] for sample in record['samples'])
            records.append(record)
    for path, digest in hashes.items():
        s = path.stat()
        if (s.st_ino, s.st_size, s.st_mtime_ns) != states[path] or file_hash(path) != digest:
            raise ValueError('Dependency/source files changed during checking')
    atomic_json(folder / 'results.json', {'completed_at': datetime.now(timezone.utc).isoformat(),
        'original_references_checked': len(records), 'original_references_with_sample_agreement': sum(r['all_recognized_samples_agree'] for r in records),
        'recognized_samples_checked': sum(len(r['samples']) for r in records), 'records': records,
        'plan_sha256': file_hash(folder / 'plan.json'), 'source_and_numpy_file_hashes_reverified': True,
        'dataset_admitted_to_training': False, 'new_model_training_started': False, 'goal_achieved': False,
        'scope': 'Additional dependency compatibility for the same two original public references only. No model, hidden-case correctness, efficiency or human preference evaluation.'})
    print('original_references_checked', len(records), 'sample_agreement', sum(r['all_recognized_samples_agree'] for r in records))


if __name__ == '__main__':
    main()
