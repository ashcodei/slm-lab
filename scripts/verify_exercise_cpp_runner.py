"""Run actual compiled-code checks for the published C++ module evaluator."""
import copy
import json
from pathlib import Path
import tempfile

from slm_lab.diverse_images import atomic_json, jsonl
from slm_lab.evaluation_provenance import capture, seal
from slm_lab.exercise_cpp_runner import PublishedSuites, extract_files

ROOT = Path(__file__).resolve().parents[1]


def main():
    files = [Path(__file__), ROOT / 'src/slm_lab/exercise_cpp_runner.py',
             ROOT / 'src/slm_lab/native_code.py', ROOT / 'scripts/native_sandbox.py']
    provenance = capture(files)
    row = next(r for r in jsonl(ROOT / 'data/exercism-code-v1/valid.jsonl')
               if r['language'] == 'C++' and r['exercise'] == 'grains')
    suites = PublishedSuites(ROOT / 'data/exercism-source-cache-v1/cpp')
    try:
        suite = suites.suite(row); gold = extract_files(row['response'])
        reference = suites.run(gold, suite)
        assert reference['published_tests_passed'], reference
        serialized_inventory = json.loads(json.dumps(reference['inventory']))
        round_trip = suites.run(gold, suite, serialized_inventory)
        assert round_trip['published_tests_passed'], round_trip
        original = 'return 1ULL << (which - 1);'
        assert original in gold['grains.cpp']
        wrong = copy.deepcopy(gold)
        wrong['grains.cpp'] = wrong['grains.cpp'].replace(original, 'return 0;')
        bad = suites.run(wrong, suite, serialized_inventory)
        assert bad['compilation_succeeded'] and not bad['published_tests_passed'], bad
        early = copy.deepcopy(gold)
        early['grains.cpp'] = '#include <cstdlib>\n' + early['grains.cpp'].replace(original, 'std::exit(0);')
        exited = suites.run(early, suite, serialized_inventory)
        assert exited['returncode'] == 0 and not exited['published_tests_passed'], exited
        with tempfile.TemporaryDirectory(prefix='slm-denied-ambient-') as temporary:
            sentinel = Path(temporary) / 'sentinel'; sentinel.write_text('kernel isolation test')
            protected = copy.deepcopy(gold)
            protected['grains.cpp'] = 'extern "C" void* fopen(const char*, const char*);\n' + protected['grains.cpp'].replace(
                original, 'if (fopen(' + json.dumps(str(sentinel)) + ', "r")) return 0; ' + original)
            blocked = suites.run(protected, suite, serialized_inventory)
            assert sentinel.read_text() == 'kernel isolation test' and blocked['published_tests_passed'], blocked
        report = {'reference': reference, 'serialized_reference_inventory': round_trip, 'wrong_algorithm': bad,
                  'exit_zero_without_complete_report': exited, 'ambient_file_read_denied': blocked,
                  'source': suites.source, 'compiler': suites.tools, 'goal_achieved': False,
                  'scope': 'Actual compiled-code failure, saved-inventory and kernel-isolation checks; not model capability.'}
        seal(report, provenance, files=files)
        atomic_json(ROOT / 'reports/exercism-cpp-kernel-check-v1.json', report)
        print('All actual C++ runner checks passed', flush=True)
    finally: suites.close()


if __name__ == '__main__': main()
