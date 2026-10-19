"""Audit original C++ training targets under the declared published-test runner."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json, jsonl
from slm_lab.evaluation_provenance import capture, seal
from slm_lab.exercise_cpp_evaluation import solution
from slm_lab.exercise_cpp_runner import PublishedSuites, validate_sources

ROOT = Path(__file__).resolve().parents[1]


def main():
    # Limit this auxiliary compiler audit to two CPUs while model training runs.
    affinity = set(sorted(os.sched_getaffinity(0))[:2]); os.sched_setaffinity(0, affinity)
    data = ROOT / 'data/exercism-code-v1/train.jsonl'
    manifest_path = data.parent / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()); digest = file_hash(data)
    if digest != manifest['files_sha256']['train.jsonl']: raise ValueError('Finalized source-module training data changed')
    rows = [row for row in jsonl(data) if row['language'] == 'C++']
    if len(rows) != manifest['splits']['train']['languages']['C++']: raise ValueError('C++ source count differs from manifest')
    files = [ROOT / name for name in capture()['files_sha256']] + [Path(__file__), manifest_path,
             ROOT / 'src/slm_lab/exercise_cpp_evaluation.py', ROOT / 'src/slm_lab/exercise_cpp_runner.py',
             ROOT / 'data/exercism-source-cache-v1/cpp/source.json']
    provenance = capture(files)
    suites = PublishedSuites(ROOT / 'data/exercism-source-cache-v1/cpp')
    output = ROOT / 'reports/exercism-cpp-training-reference-audit-v1.json'
    results = []
    try:
        for row in rows:
            reference = solution(row, suites)
            try: validate_sources(reference)
            except ValueError as error:
                item = {'status': 'unsupported_by_declared_solution_policy', 'reason': str(error),
                        'headers': [line for code in reference.values() for line in code.splitlines()
                                    if line.strip().startswith('#include')], 'reference_executed': False}
            else:
                try: suite = suites.suite(row)
                except KeyError as error:
                    item = {'status': 'original_bundled_test_harness_unavailable', 'reason': str(error),
                            'reference_executed': False}
                else:
                    result = suites.run(reference, suite)
                    if result['published_tests_passed']: category = 'published_tests_passed'
                    elif not result['compilation_succeeded']: category = 'compile_failed_under_declared_toolchain'
                    elif result.get('runtime_error'): category = 'runtime_or_report_failed_under_declared_limits'
                    else: category = 'published_tests_failed_under_declared_runtime'
                    item = {'status': category, 'reference_executed': result['compilation_succeeded'],
                            'test_sha256': suite['test_sha256'], 'catch_sha256': suite['catch_sha256'],
                            'main_sha256': suite['main_sha256'], 'result': result}
            item.update(id=row['id'], group_id=row['group_id'], exercise=row['exercise'],
                        solution_files=sorted(reference), source_url=row['source_url'],
                        response_sha256=hashlib.sha256(row['response'].encode()).hexdigest())
            results.append(item)
            atomic_json(output.with_suffix('.progress.json'), {'completed': len(results), 'examples': len(rows),
                        'counts': dict(Counter(r['status'] for r in results)), 'results': results, 'goal_achieved': False})
            print('C++ training reference', row['exercise'], item['status'], flush=True)
        report = {'updated_at': datetime.now(timezone.utc).isoformat(), 'data_sha256': digest,
                  'manifest_sha256': file_hash(manifest_path), 'source': suites.source, 'toolchain': suites.tools,
                  'compiler_options': suites.common[1:], 'audit_cpu_affinity': sorted(affinity), 'examples': len(rows),
                  'counts': dict(Counter(r['status'] for r in results)), 'results': results, 'goal_achieved': False,
                  'training_corpus_modified': False, 'trained_model_score': False,
                  'scope': 'Original training-reference execution quality under the fixed C++ runner; not held-out model capability.',
                  'limitations': ['Unsupported library/preprocessor/thread/dependency forms are not evidence of wrong publisher code.',
                                  'Compile/runtime failures can reflect toolchain/dependency/resource incompatibility; inspect before changing data.',
                                  'Published tests do not prove universal correctness or efficient algorithms.',
                                  'Source targets, original tests and active model training data remain unchanged.']}
        seal(report, provenance, files=files, data=data, digest=digest); atomic_json(output, report)
    finally: suites.close()


if __name__ == '__main__': main()
