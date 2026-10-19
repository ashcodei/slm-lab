"""Run the original training exercise that bundles a newer Catch test harness."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json, jsonl
from slm_lab.evaluation_provenance import capture, seal
from slm_lab.exercise_cpp_evaluation import solution
from slm_lab.exercise_cpp_runner import PublishedSuites, parse_junit, validate_sources
from slm_lab.native_code import NativeBackendError, launch, policy, stop

ROOT = Path(__file__).resolve().parents[1]


def main():
    affinity = set(sorted(os.sched_getaffinity(0))[:2]); os.sched_setaffinity(0, affinity)
    data = ROOT / 'data/exercism-code-v1/train.jsonl'; digest = file_hash(data)
    manifest = json.loads((data.parent / 'manifest.json').read_text())
    if digest != manifest['files_sha256']['train.jsonl']: raise ValueError('Original training data changed')
    row = next(r for r in jsonl(data) if r['language'] == 'C++' and r['exercise'] == 'complex-numbers')
    files = [ROOT / name for name in capture()['files_sha256']] + [Path(__file__),
             ROOT / 'src/slm_lab/exercise_cpp_runner.py', ROOT / 'src/slm_lab/exercise_cpp_evaluation.py',
             data.parent / 'manifest.json', ROOT / 'data/exercism-source-cache-v1/cpp/source.json']
    provenance = capture(files); suites = PublishedSuites(ROOT / 'data/exercism-source-cache-v1/cpp')
    try:
        work = suites.work; reference = solution(row, suites); validate_sources(reference)
        base = row['source_files'][0]['path'].split('/.meta/')[0] + '/'
        (work / 'test').mkdir(); hashes = {}
        for name in ['test/catch_amalgamated.hpp', 'test/catch_amalgamated.cpp', 'test/tests-main.cpp']:
            raw = suites.read(base + name); (work / name).write_bytes(raw); hashes[name] = hashlib.sha256(raw).hexdigest()
        tests = []
        for artifact in row['published_test_artifacts']:
            if not artifact['path'].endswith('_test.cpp'): continue
            raw = suites.read(artifact['path'])
            if hashlib.sha256(raw).hexdigest() != artifact['sha256']: raise ValueError('Original publisher test changed')
            name = Path(artifact['path']).name; (work / name).write_bytes(raw); tests.append(name); hashes[name] = artifact['sha256']
        if len(tests) != 1: raise ValueError('Unexpected original modern-harness exercise tests')
        for name, code in reference.items(): (work / name).write_text(code)
        sources = [name for name in reference if name.endswith('.cpp')] + tests + ['test/catch_amalgamated.cpp', 'test/tests-main.cpp']
        objects = []
        for index, source in enumerate(sources):
            target = f'unit-{index}.o'; error = suites.compile(['-c', source, '-o', target], work)
            if error: raise NativeBackendError('Original modern-harness compilation failed: ' + error)
            objects.append(target)
        error = suites.compile(objects + ['-o', 'program'], work)
        if error: raise NativeBackendError('Original modern-harness link failed: ' + error)
        stdout, stderr = work / 'junit.xml', work / 'runtime-errors.txt'
        with stdout.open('wb') as output, stderr.open('wb') as errors:
            configuration = policy('runtime', [work / 'program', '--reporter', 'junit', '--rng-seed', '42'], work)
            # Catch3/libc++ opens the entropy device during startup. This is a
            # single read-only device grant; workspace, process and network
            # restrictions remain enforced before executing the original code.
            configuration['paths'].append(('/dev/urandom', 1 << 2))
            process = launch(configuration,
                             stdin=subprocess.DEVNULL, stdout=output, stderr=errors)
            try: returncode = process.wait(timeout=15)
            except subprocess.TimeoutExpired: stop(process); raise RuntimeError('Original modern-harness execution deadline exceeded')
        diagnostic = stderr.read_bytes()[-1200:].decode(errors='replace')
        if 'SANDBOX_ERROR:' in diagnostic: raise NativeBackendError(diagnostic)
        raw = stdout.read_bytes()
        print('Original modern-harness runtime', returncode, 'stdout bytes', len(raw), 'diagnostic', diagnostic, flush=True)
        try: inventory = parse_junit(raw); report_error = None
        except (ValueError, ET.ParseError, KeyError) as error:
            inventory = None; report_error = str(error)
        report = {'updated_at': datetime.now(timezone.utc).isoformat(), 'id': row['id'], 'exercise': row['exercise'],
                  'data_sha256': digest, 'source': suites.source, 'toolchain': suites.tools,
                  'original_test_harness_files_sha256': hashes, 'compiler_options': suites.common[1:],
                  'cpu_affinity': sorted(affinity), 'returncode': returncode, 'runtime_diagnostic': diagnostic,
                  'additional_read_only_device': '/dev/urandom',
                  'inventory': inventory, 'report_error': report_error,
                  'published_tests_passed': returncode == 0 and inventory is not None and inventory['all_passed'],
                  'report_sha256': hashlib.sha256(raw).hexdigest(), 'goal_achieved': False,
                  'training_corpus_modified': False, 'trained_model_score': False,
                  'scope': 'Original newer bundled harness and original training reference only; fixed held-out model evaluator unchanged.'}
        seal(report, provenance, files=files, data=data, digest=digest)
        atomic_json(ROOT / 'reports/exercism-cpp-modern-reference-audit-v1.json', report)
        if not report['published_tests_passed']: raise RuntimeError('Original newer-harness reference failed its publisher tests')
        print('Original newer Catch harness and training reference passed', inventory['assertions'], 'assertions', flush=True)
    finally: suites.close()


if __name__ == '__main__': main()
