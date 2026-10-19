"""Run all35 untouched Java reference modules against complete original tests."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json, jsonl
from slm_lab.evaluation_provenance import capture
from slm_lab.exercise_java_runner import PublishedSuites, HARNESS, LAUNCHER

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports/exercism-java-publisher-references-v3'
DATA = ROOT / 'data/exercism-support-context-v3/valid.jsonl'


def main():
    if REPORTS.exists():
        raise ValueError('Preserve earlier Java reference evidence; no automatic retry')
    if shutil.disk_usage(ROOT).free < 26 * 1024**3 + 64 * 1024**2:
        raise ValueError('Fund64MiB reports/work above the unchanged26GiB reserve')
    manifest = DATA.parent / 'manifest.json'
    if file_hash(DATA) != json.loads(manifest.read_text())['files_sha256'][DATA.name]:
        raise ValueError('Exact finalized original held module data required')
    rows = [r for r in jsonl(DATA) if r['language'] == 'Java']
    if len(rows) != 35 or len({r['id'] for r in rows}) != 35:
        raise ValueError('Every35original Java module required')
    REPORTS.mkdir()
    paths = [Path(__file__), ROOT / 'src/slm_lab/exercise_java_runner.py', HARNESS, LAUNCHER,
        ROOT / 'scripts/native_sandbox.py', ROOT / 'src/slm_lab/native_code.py',
        ROOT / 'src/slm_lab/codenet_checkpoint_v1.py', ROOT / 'src/slm_lab/evaluation_provenance.py',
        ROOT / '.tools/java-toolchain-v1.json', ROOT / '.tools/java-test-framework-v1.json',
        ROOT / 'data/exercism-source-cache-v1/java/source.json', ROOT / 'data/exercism-source-cache-v1/java/source.zip', DATA, manifest]
    runtime = capture(paths)
    bindings = {str(p): file_hash(p) for p in paths}
    guard = FileBindings(bindings)
    suites = PublishedSuites(ROOT / 'data/exercism-source-cache-v1/java')
    results = []
    try:
        for row in rows:
            guard.check()
            original, suite = suites.reference(row), suites.suite(row)
            measured = suites.run(original, suite)
            result = dict(id=row['id'], exercise=row['exercise'], group_id=row['group_id'],
                response_sha256=__import__('hashlib').sha256(row['response'].encode()).hexdigest(),
                original_test_sha256=suite['test_sha256'], original_editor_sha256=suite['editor_sha256'],
                original_build_sha256=suite['build_sha256'], mode=suite['mode'],
                expected_solution_files=sorted(original), result=measured)
            results.append(result)
            atomic_json(REPORTS / ('reference-' + row['exercise'] + '.json'), result)
            atomic_json(REPORTS / 'status.json', dict(status='executing complete original Java references in confinement',
                updated_at=datetime.now(timezone.utc).isoformat(), completed=len(results), planned=35,
                passed=sum(r['result']['published_tests_passed'] for r in results), goal_achieved=False))
            print('java_reference',len(results),row['exercise'],measured['compilation_passed'],
                measured['published_tests_passed'],len(measured.get('inventory',{}).get('cases',[])),flush=True)
        guard.check(full=True)
        suites.guard.check(full=True)
        if capture(paths) != runtime:
            raise ValueError('Original Java source/runtime changed during reference verification')
        passed = sum(r['result']['published_tests_passed'] for r in results)
        report = dict(completed_at=datetime.now(timezone.utc).isoformat(), data_sha256=file_hash(DATA),
            manifest_sha256=file_hash(manifest), source_archive_sha256=suites.source['archive_sha256'],
            Java_toolchain_manifest_sha256=file_hash(ROOT / '.tools/java-toolchain-v1.json'),
            test_framework_manifest_sha256=file_hash(ROOT / '.tools/java-test-framework-v1.json'),
            evaluation_provenance=runtime, source_files_sha256=bindings, results=results, passed=passed,
            complete_all35_reference_suites_passed=passed==35, every_original_held_module_attempted=True,
            source_or_test_bytes_modified=False, teaching_disabled_tests_enabled_by_runner=True,
            original_provided_editor_files_supplied=True, model_generation_or_training_performed=False,
            same_process_event_inventory_not_adversarial_forgery_proof=True, goal_achieved=False)
        atomic_json(REPORTS / 'summary.json', report)
        if passed != 35:
            raise ValueError('Some original Java references failed; inspect every retained result')
        atomic_json(REPORTS / 'status.json', dict(status='all35 original Java reference suites passed; model assessment remains',
            updated_at=datetime.now(timezone.utc).isoformat(), passed=35, goal_achieved=False))
    except Exception as error:
        atomic_json(REPORTS / 'failure.json', dict(error=str(error), recorded_at=datetime.now(timezone.utc).isoformat(),
            automatic_retry=False, retain_all_evidence=True, goal_achieved=False))
        raise
    finally:
        suites.close()


if __name__ == '__main__':
    main()
