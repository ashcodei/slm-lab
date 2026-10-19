"""Assess original training targets with unchanged complete publisher suites."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json, jsonl
from slm_lab.evaluation_provenance import capture
from slm_lab.native_code import NativeBackendError

ROOT = Path(__file__).resolve().parents[1]
BACKENDS = {'JavaScript': 'js', 'Go': 'go', 'Rust': 'rust'}
EXPECTED = {'JavaScript': 123, 'Go': 124, 'Rust': 82}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--language', choices=BACKENDS, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=ROOT / 'data/exercism-code-v1/train.jsonl')
    parser.add_argument('--exercise')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve completed training-source native evidence')
    args.output.parent.mkdir(exist_ok=True)
    data = args.data.resolve(); manifest = data.parent / 'manifest.json'
    digest = file_hash(data); manifest_digest = file_hash(manifest)
    if json.loads(manifest.read_text())['files_sha256'][data.name] != digest:
        raise ValueError('Original training targets changed')
    backend = BACKENDS[args.language]
    module = __import__('slm_lab.exercise_' + backend + '_runner', fromlist=['PublishedSuites'])
    cache = ROOT / 'data/exercism-source-cache-v1' / {'js': 'javascript', 'go': 'go', 'rust': 'rust'}[backend]
    tools = ROOT / '.tools' / ({'js': 'javascript', 'go': 'go', 'rust': 'rust'}[backend] + '-toolchain-v1.json')
    files = [Path(__file__), Path(module.__file__), ROOT / 'scripts' / ({'js': 'javascript', 'go': 'go', 'rust': 'rust'}[backend] + '_sandbox.py'),
             ROOT / 'scripts/native_sandbox.py', ROOT / 'scripts/javascript_sandbox.py', tools,
             ROOT / 'src/slm_lab/native_code.py', ROOT / 'src/slm_lab/evaluation_provenance.py']
    runtime = capture(files); tools_digest = file_hash(tools); source_digest = file_hash(cache / 'source.zip')
    rows = [row for row in jsonl(data) if row['language'] == args.language]
    if args.exercise:
        rows = [row for row in rows if row['exercise'] == args.exercise]
    source_manifest = json.loads(manifest.read_text())
    expected = source_manifest.get('by_language', {}).get(args.language, EXPECTED[args.language])
    if len(rows) != (1 if args.exercise else expected):
        raise ValueError('Complete original training population differs')
    originals = {row['id']: row for row in jsonl(ROOT / 'data/exercism-code-v1/train.jsonl')}
    suites = module.PublishedSuites(cache); results = []
    try:
        for row in rows:
            item = {'id': row['id'], 'group_id': row['group_id'], 'exercise': row['exercise'],
                    'original_row_sha256': hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()}
            try:
                original = originals.get(row.get('source_original_id', row['id']))
                if original is None:
                    raise ValueError('Derived source row lacks its unchanged original')
                reference = suites.reference(original); suite = suites.suite(original)
                if row.get('reference_relocation'):
                    transformed = original['response']
                    for substitution in row['reference_relocation']['literal_import_substitutions']:
                        if transformed.count(substitution['before']) != 1:
                            raise ValueError('Declared static import relocation is not unique')
                        transformed = transformed.replace(substitution['before'], substitution['after'])
                    if transformed != row['response']:
                        raise ValueError('Source target differs beyond its declared import relocation')
                    reference = module.extract_files(transformed, reference)
                    item['source_target_transformation'] = row['reference_relocation']
                for provided in row.get('provided_support_files', []):
                    path = (data.parent / provided['stored_path']).resolve()
                    if (not path.is_relative_to(data.parent) or file_hash(path) != provided['sha256']
                            or provided['file_name'] in reference):
                        raise ValueError('Pinned publisher supporting-file evidence differs')
                    reference[provided['file_name']] = path.read_bytes().decode('utf-8')
                item.update(original_reference_files_sha256={name: hashlib.sha256(value.encode()).hexdigest() for name, value in reference.items()},
                            original_test_sha256=suite['test_sha256'], result=suites.run(reference, suite))
            except (ValueError, NativeBackendError) as error:
                item['assessment_error'] = {'type': type(error).__name__, 'message': str(error)}
            results.append(item)
            passed = bool(item.get('result', {}).get('published_tests_passed'))
            print(json.dumps({'language': args.language, 'exercise': row['exercise'], 'published_tests_passed': passed,
                              'assessment_error': item.get('assessment_error')}), flush=True)
            atomic_json(args.output.with_suffix('.progress.json'), {'updated_at': datetime.now(timezone.utc).isoformat(),
                'language': args.language, 'processed_examples': len(results), 'expected_examples': len(rows),
                'passed': sum(bool(item.get('result', {}).get('published_tests_passed')) for item in results), 'goal_achieved': False})
        if (capture(files) != runtime or file_hash(data) != digest or file_hash(manifest) != manifest_digest
                or file_hash(tools) != tools_digest or file_hash(cache / 'source.zip') != source_digest):
            raise ValueError('Original training source/runtime changed during native assessment')
        passed = sum(bool(item.get('result', {}).get('published_tests_passed')) for item in results)
        atomic_json(args.output, {'completed_at': datetime.now(timezone.utc).isoformat(), 'language': args.language,
            'data_sha256': digest, 'manifest_sha256': manifest_digest, 'source_archive_sha256': source_digest,
            'toolchain_manifest_sha256': tools_digest, 'evaluation_provenance': runtime,
            'examples': len(rows), 'passed': passed, 'all_original_targets_pass_complete_publisher_suites': passed == len(rows),
            'assessment_errors': sum('assessment_error' in item for item in results), 'results': results,
            'scope': 'Every admitted original training target for this language, original complete publisher tests and pinned native backend. Reference tests are not model scores.',
            'limitations': ['An unsuccessful assessment can reflect unsupported dependencies, confined runtime or budget limits; inspect diagnostics before treating a source target as incorrect.',
                           'Publisher suite success does not prove universal correctness, asymptotic efficiency or held-task generalization.',
                           'No training targets or held reference/evaluation reports are changed by this assessment.'], 'goal_achieved': False})
        print('Training-source native assessment completed', args.language, passed, '/', len(rows), flush=True)
    finally:
        suites.close()


if __name__ == '__main__':
    main()
