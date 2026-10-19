"""Run all original held JavaScript references against unchanged publisher Jest suites."""
from datetime import datetime, timezone
import json
from pathlib import Path

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json, jsonl
from slm_lab.evaluation_provenance import capture
from slm_lab.exercise_js_runner import PublishedSuites

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / 'reports/exercism-javascript-publisher-references-v1.json'
    if output.exists(): raise ValueError('Preserve existing original JavaScript reference evidence')
    data = ROOT / 'data/exercism-code-v1/valid.jsonl'; digest = file_hash(data)
    manifest_path = data.parent / 'manifest.json'; manifest_sha = file_hash(manifest_path)
    if json.loads(manifest_path.read_text())['files_sha256'][data.name] != digest: raise ValueError('Finalized original exercise validation data changed')
    files = [Path(__file__), ROOT / 'src/slm_lab/exercise_js_runner.py', ROOT / 'scripts/javascript_sandbox.py',
             ROOT / 'scripts/native_sandbox.py', ROOT / 'src/slm_lab/native_code.py', ROOT / 'src/slm_lab/corpus.py',
             ROOT / 'src/slm_lab/diverse_images.py', ROOT / 'src/slm_lab/evaluation_provenance.py']
    runtime = capture(files); suites = PublishedSuites(ROOT / 'data/exercism-source-cache-v1/javascript')
    source_sha = file_hash(ROOT / 'data/exercism-source-cache-v1/javascript/source.zip')
    toolchain_sha = file_hash(ROOT / '.tools/javascript-toolchain-v1.json')
    results = []
    try:
        rows = [row for row in jsonl(data) if row['language'] == 'JavaScript']
        if len(rows) != 37: raise ValueError('All37 original held-out JavaScript modules required')
        for row in rows:
            reference = suites.reference(row); suite = suites.suite(row)
            actual = suites.run(reference, suite)
            results.append({'id': row['id'], 'group_id': row['group_id'], 'original_test_sha256': suite['test_sha256'],
                            'original_reference_file_sha256': {name: file_hash_bytes(code) for name, code in reference.items()},
                            'result': actual})
            print('Original JavaScript reference', row['exercise'], actual['published_tests_passed'], actual.get('inventory', {}).get('assertions'), flush=True)
        if capture(files) != runtime or file_hash(data) != digest or file_hash(manifest_path) != manifest_sha:
            raise ValueError('Original JavaScript reference input/runner/runtime changed')
        if file_hash(ROOT / 'data/exercism-source-cache-v1/javascript/source.zip') != source_sha or file_hash(ROOT / '.tools/javascript-toolchain-v1.json') != toolchain_sha:
            raise ValueError('Original JavaScript source archive/toolchain receipt changed')
        report = {'completed_at': datetime.now(timezone.utc).isoformat(), 'data_sha256': digest, 'manifest_sha256': manifest_sha,
                  'source_archive_sha256': source_sha, 'toolchain_manifest_sha256': toolchain_sha, 'evaluation_provenance': runtime,
                  'examples': len(results), 'passed': sum(row['result']['published_tests_passed'] for row in results),
                  'complete_all37_reference_suites_passed': all(row['result']['published_tests_passed'] for row in results),
                  'results': results, 'goal_achieved': False,
                  'scope': 'Original publisher reference modules and original test bytes; skipped publisher cases enabled through the registered Jest test tree. Reference correctness is not a model score.',
                  'limitations': ['Same-process Jest inventories do not certify adversarial score-forgery resistance.',
                                  'ESM via Node VM modules remains experimental; methods/API scope is defined by original suites, not all JavaScript.']}
        atomic_json(output, report)
        print(json.dumps({key: report[key] for key in ['examples', 'passed', 'complete_all37_reference_suites_passed']}), flush=True)
    finally: suites.close()


def file_hash_bytes(value):
    import hashlib
    return hashlib.sha256(value.encode()).hexdigest()


if __name__ == '__main__': main()
