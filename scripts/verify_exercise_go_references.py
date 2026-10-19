"""Verify all original held Go implementations against complete publisher suites."""
import hashlib
import json
from pathlib import Path

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json, jsonl
from slm_lab.evaluation_provenance import capture
from slm_lab.exercise_go_runner import PublishedSuites

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / 'reports/exercism-go-publisher-references-v1.json'
    if output.exists(): raise ValueError('Preserve completed original Go reference evidence')
    data = ROOT / 'data/exercism-code-v1/valid.jsonl'; manifest = data.parent / 'manifest.json'
    data_sha = file_hash(data); manifest_sha = file_hash(manifest)
    if json.loads(manifest.read_text())['files_sha256']['valid.jsonl'] != data_sha: raise ValueError('Original held Go rows changed')
    rows = [row for row in jsonl(data) if row['language'] == 'Go']
    if len(rows) != 42: raise ValueError('All42 original held Go modules required')
    files = [Path(__file__), ROOT / 'src/slm_lab/exercise_go_runner.py', ROOT / 'scripts/go_sandbox.py',
             ROOT / 'scripts/native_sandbox.py', ROOT / '.tools/go-toolchain-v1.json']
    runtime = capture(files); suites = PublishedSuites(ROOT / 'data/exercism-source-cache-v1/go'); results = []
    try:
        for row in rows:
            source = suites.reference(row); suite = suites.suite(row); measured = suites.run(source, suite)
            results.append({'id': row['id'], 'group_id': row['group_id'], 'original_test_sha256': suite['test_sha256'],
                'original_reference_file_sha256': {name: hashlib.sha256(raw.encode()).hexdigest() for name, raw in source.items()},
                'result': measured})
            print('Original Go publisher suite', row['exercise'], measured['compilation_passed'], measured['published_tests_passed'], flush=True)
        if capture(files) != runtime or file_hash(data) != data_sha or file_hash(manifest) != manifest_sha:
            raise ValueError('Original Go reference evidence/runtime changed')
        passed = sum(row['result']['published_tests_passed'] for row in results)
        atomic_json(output, {'data_sha256': data_sha, 'manifest_sha256': manifest_sha,
            'source_archive_sha256': suites.source['archive_sha256'], 'toolchain_manifest_sha256': file_hash(ROOT / '.tools/go-toolchain-v1.json'),
            'evaluation_provenance': runtime, 'examples': 42, 'passed': passed,
            'complete_all42_reference_suites_passed': passed == 42, 'results': results, 'goal_achieved': False,
            'scope': 'Exact complete original Go test source and module files; all registered ordinary tests and subtests, no benchmarks.',
            'limitations': ['Same-process Go test event inventories are not adversarial score-forgery proof.',
                'A bounded full module suite does not prove asymptotic optimality, universal Go support or model capability.']})
        print('Complete Go reference module result', passed, '/42', flush=True)
    finally: suites.close()


if __name__ == '__main__': main()
