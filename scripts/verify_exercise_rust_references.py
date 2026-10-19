"""Verify every original held Rust reference against complete unmodified native tests."""
import hashlib
import argparse
import json
from pathlib import Path

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json, jsonl
from slm_lab.evaluation_provenance import capture
from slm_lab.exercise_rust_runner import PublishedSuites

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/exercism-rust-publisher-references-v1.json')
    output = parser.parse_args().output
    if output.exists(): raise ValueError('Preserve completed original Rust reference evidence')
    data = ROOT / 'data/exercism-code-v1/valid.jsonl'; manifest = data.parent / 'manifest.json'
    data_sha = file_hash(data); manifest_sha = file_hash(manifest)
    if json.loads(manifest.read_text())['files_sha256']['valid.jsonl'] != data_sha: raise ValueError('Original held Rust rows changed')
    rows = [row for row in jsonl(data) if row['language'] == 'Rust']
    if len(rows) != 24: raise ValueError('All24 original held Rust modules required')
    files = [Path(__file__), ROOT / 'src/slm_lab/exercise_rust_runner.py', ROOT / 'scripts/rust_sandbox.py',
        ROOT / 'scripts/javascript_sandbox.py', ROOT / 'scripts/native_sandbox.py', ROOT / '.tools/rust-toolchain-v1.json',
        ROOT / 'scripts/prepare_rust_test_libraries.py']
    runtime = capture(files); suites = PublishedSuites(ROOT / 'data/exercism-source-cache-v1/rust'); results = []
    try:
        for row in rows:
            source = suites.reference(row); suite = suites.suite(row); measured = suites.run(source, suite)
            results.append({'id': row['id'], 'group_id': row['group_id'], 'original_test_sha256': suite['test_sha256'],
                'original_reference_file_sha256': {name: hashlib.sha256(raw.encode()).hexdigest() for name, raw in source.items()},
                'result': measured})
            print('Original Rust publisher suite', row['exercise'], measured['compilation_passed'], measured['published_tests_passed'], flush=True)
            if not measured['published_tests_passed']: print('Rust reference diagnostic', repr(measured)[-3500:], flush=True)
        if capture(files) != runtime or file_hash(data) != data_sha or file_hash(manifest) != manifest_sha:
            raise ValueError('Original Rust reference evidence/runtime changed')
        passed = sum(row['result']['published_tests_passed'] for row in results)
        atomic_json(output, {'data_sha256': data_sha, 'manifest_sha256': manifest_sha,
            'source_archive_sha256': suites.source['archive_sha256'], 'toolchain_manifest_sha256': file_hash(ROOT / '.tools/rust-toolchain-v1.json'),
            'evaluation_provenance': runtime, 'examples': 24, 'passed': passed,
            'complete_all24_reference_suites_passed': passed == 24, 'results': results, 'goal_achieved': False,
            'scope': 'Every original integration test source, unchanged package metadata and solution references; Rust2024/musl native execution including ignored and should-panic cases.',
            'limitations': ['Explicit pinned regex-lite and pretty_assertions provision supplements publisher references without changing their source or package manifests.',
                'Same-process native test inventories are not adversarial score-forgery proof.',
                'This is not Rust doctest coverage, arbitrary Cargo dependency support, an asymptotic guarantee or model capability evidence.']})
        print('Complete Rust reference module result', passed, '/24', flush=True)
    finally: suites.close()


if __name__ == '__main__': main()
