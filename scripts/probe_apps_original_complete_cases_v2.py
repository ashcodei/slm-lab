"""Check source-selected complete APPS train references in a declared runtime.

Uses every original case for each supported selected source. It creates no
training labels and never reads test/validation programs or model generations.
"""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from slm_lab.apps_published_cases_v2 import PRELUDE, contract, execute, implementation_sources
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'data/public-apps-original-inventory-v1'
VIEW = ROOT/'data/public-apps-human-solution-view-v1'
OUTPUT = ROOT/'reports/APPS-original-complete-cases-v2'


def check(row, inventory):
    problem = row['source_problem_id']
    folder = SOURCE/'original-train'/problem
    base = dict(id=row['id'], problem=problem, publisher_role='train', admitted_role='train',
        original_solution_index=row['original_solution_index'], difficulty=inventory[problem]['metadata']['difficulty'])
    path = folder/'input_output.json'
    if not path.exists():
        return dict(**base, status='original_source_missing_tests')
    raw = path.read_bytes()
    base['original_tests_sha256'] = hashlib.sha256(raw).hexdigest()
    if base['original_tests_sha256'] != inventory[problem]['files']['input_output.json']['sha256']:
        raise ValueError('Actual complete original publisher test bytes changed')
    try:
        cases = contract(raw)
    except ValueError as error:
        return dict(**base, status='source_contract_review', qualification=str(error))
    programs_path = folder/'solutions.json'
    programs_raw = programs_path.read_bytes()
    if hashlib.sha256(programs_raw).hexdigest() != inventory[problem]['files']['solutions.json']['sha256']:
        raise ValueError('Actual complete original publisher human-program bytes changed')
    program = json.loads(programs_raw)[row['original_solution_index']]
    result = execute(program, cases, seconds=20)
    return {**base, **result}


def main():
    if OUTPUT.exists():
        raise ValueError('Preserve earlier complete-case observations; no implicit retry')
    inventory = json.loads((SOURCE/'train-problems.json').read_text())
    groups = defaultdict(list)
    for row in map(json.loads, (VIEW/'train-records.jsonl').read_text().splitlines()):
        if row['split'] != 'train' or row['source_problem_id'] not in inventory:
            raise ValueError('Original admitted publisher-train programs only')
        groups[row['source_problem_id']].append(row)
    selected = [min(rows, key=lambda r:(r['target_tokens'], r['id'])) for _,rows in sorted(groups.items())]
    files = implementation_sources()
    for p in [Path(__file__).resolve(), VIEW/'admission.json', VIEW/'train-records.jsonl', SOURCE/'train-problems.json',
            ROOT/'tests/test_apps_published_cases_v2.py', ROOT/'reports/apps-published-cases-v2-tests.log']:
        files[str(p)] = file_hash(p)
    if '3 passed' not in (ROOT/'reports/apps-published-cases-v2-tests.log').read_text():
        raise ValueError('Actual standard-IO/class/full-case/confinement checks required')
    for row in selected:
        problem = row['source_problem_id']
        for name in ['solutions.json', 'input_output.json']:
            if name in inventory[problem]['files']:
                files[str(SOURCE/'original-train'/problem/name)] = inventory[problem]['files'][name]['sha256']
    guard = FileBindings(files)
    guard.check(full=True)
    OUTPUT.mkdir()
    plan = dict(created_at=datetime.now(timezone.utc).isoformat(), source_files_sha256=files,
        source_selected_original_train_problems=len(selected), selected_program_ids=[r['id'] for r in selected],
        selection='minimum complete response target tokens, then original source ID, per admitted train problem',
        maximum_concurrent_CPU_workers=4, maximum_seconds_per_complete_program=20,
        maximum_cases_per_program=1000, maximum_case_payload_bytes=4*1024**2,
        all_cases_required_without_clipping=True, expected_values_visible_to_program=False,
        runtime='Own stdlib-only full Python source execution with existing Landlock/seccomp confinement',
        injected_runtime_prelude=PRELUDE, original_program_bytes_modified=False,
        comparison=dict(stdin='Exact complete text after outside whitespace stripping; ordered, no numeric/bag approximation',
            function='Literal expected value OR first element of a nonempty expected list; tuples serialize as lists'),
        qualifications=['Not the publisher pyext/NumPy runtime or an official APPS score.',
            'Fresh stdin namespace for each case; function module/Solution object retained across its cases.',
            'Numeric dictionary key reconstruction and unsupported/full-size source schemas require review.',
            'Complete source programs may fail because of the bounded local runtime, Python version or source tests.',
            'Finite published-case passing is not proof of algorithmic correctness or efficiency.'],
        original_test_or_validation_programs_used=False, model_generations=0, model_updates=0,
        training_or_preference_labels_admitted=False, goal_achieved=False)
    atomic_json(OUTPUT/'plan.json', plan)
    counts = Counter()
    results = []
    with ThreadPoolExecutor(max_workers=4) as pool, (OUTPUT/'observations.jsonl').open('x') as stream:
        futures = [pool.submit(check, row, inventory) for row in selected]
        for future in as_completed(futures):
            row = future.result()
            results.append(row)
            counts[row['status']] += 1
            stream.write(json.dumps(row, separators=(',', ':'), allow_nan=False)+'\n')
            stream.flush()
            if len(results) % 50 == 0:
                guard.check()
                progress = dict(observed_at=datetime.now(timezone.utc).isoformat(), checked=len(results),
                    planned=len(selected), outcomes=dict(counts), goal_achieved=False)
                atomic_json(OUTPUT/'progress.json', progress)
                print(json.dumps(progress), flush=True)
    guard.check(full=True)
    if sorted(r['id'] for r in results) != sorted(r['id'] for r in selected):
        raise ValueError('Every source-selected original result required exactly once')
    passing = [r for r in results if r['status'] == 'all_finite_published_cases_pass']
    summary = dict(completed_at=datetime.now(timezone.utc).isoformat(), checked=len(results),
        outcomes=dict(counts), passing_original_programs=len(passing),
        complete_original_cases_passing=sum(r['total_cases'] for r in passing),
        passing_by_interface=dict(Counter(r['mode'] for r in passing)),
        passing_by_difficulty=dict(Counter(r['difficulty'] for r in passing)),
        plan_sha256=file_hash(OUTPUT/'plan.json'), observations_sha256=file_hash(OUTPUT/'observations.jsonl'),
        all_selected_source_ids_reconciled=True, full_original_case_sets_required=True,
        model_generations=0, model_updates=0, training_or_preference_labels_admitted=False,
        official_APPS_score=False, trained_model_gain=False, goal_achieved=False)
    atomic_json(OUTPUT/'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
