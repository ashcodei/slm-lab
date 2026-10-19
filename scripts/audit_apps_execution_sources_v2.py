"""Independently reconcile actual original source cases and held prompts."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from tokenizers import Tokenizer

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT = Path(__file__).resolve().parents[1]


def check_train():
    folder = ROOT/'reports/APPS-original-complete-cases-v2'
    plan = json.loads((folder/'plan.json').read_text())
    report = json.loads((folder/'summary.json').read_text())
    if (file_hash(folder/'plan.json') != report['plan_sha256']
            or file_hash(folder/'observations.jsonl') != report['observations_sha256']):
        raise ValueError('Exact actual completed source observations required')
    for name, expected in plan['source_files_sha256'].items():
        if file_hash(name) != expected:
            raise ValueError('Original source/runtime changed: '+name)
    view = ROOT/'data/public-apps-human-solution-view-v1/train-records.jsonl'
    groups = defaultdict(list)
    for row in map(json.loads, view.read_text().splitlines()):
        if row['split'] != 'train':
            raise ValueError('Actual source roles differ')
        groups[row['source_problem_id']].append(row)
    chosen = {min(rows,key=lambda r:(r['target_tokens'],r['id']))['id']:min(rows,key=lambda r:(r['target_tokens'],r['id']))
        for rows in groups.values()}
    rows = list(map(json.loads,(folder/'observations.jsonl').read_text().splitlines()))
    if len(rows) != len(chosen) or {r['id'] for r in rows} != chosen.keys():
        raise ValueError('All actual source-selected originals must reconcile exactly once')
    cases_passing = 0
    for row in rows:
        original = chosen[row['id']]
        if (row['publisher_role'] != 'train' or row['admitted_role'] != 'train'
                or row['problem'] != original['source_problem_id']
                or row['original_solution_index'] != original['original_solution_index']):
            raise ValueError('Every original program/role identity must match')
        source = ROOT/'data/public-apps-original-inventory-v1/original-train'/row['problem']
        if 'original_tests_sha256' in row and file_hash(source/'input_output.json') != row['original_tests_sha256']:
            raise ValueError('Actual original test bytes differ')
        if 'original_program_sha256' in row:
            program = json.loads((source/'solutions.json').read_bytes())[row['original_solution_index']]
            if hashlib.sha256(program.encode()).hexdigest() != row['original_program_sha256']:
                raise ValueError('Actual complete human program bytes differ')
        if row.get('all_cases_observed'):
            tests = json.loads((source/'input_output.json').read_bytes())
            values = row['raw_results']['results']
            if len(values) != len(tests['inputs']) or len(values) != len(tests['outputs']) or len(values) != row['total_cases']:
                raise ValueError('Every complete original case must be observed')
            flags = []
            for actual, expected in zip(values, tests['outputs']):
                actual = actual['value']
                if tests.get('fn_name') is None:
                    if isinstance(expected,list):
                        expected = '\n'.join(expected)
                    flags.append(isinstance(actual,str) and actual.strip() == expected.strip())
                else:
                    flags.append(actual == expected or (isinstance(expected,list) and bool(expected) and actual == expected[0]))
            if (flags != row['case_passes'] or sum(flags) != row['passed_cases']
                    or all(flags) != row['all_finite_published_cases_pass']):
                raise ValueError('Raw values must independently match the actual source expected answers')
        if row['status'] == 'all_finite_published_cases_pass':
            if not row.get('all_cases_observed') or not row['all_finite_published_cases_pass']:
                raise ValueError('No complete-case success without every actual raw result')
            cases_passing += row['total_cases']
    counts = Counter(r['status'] for r in rows)
    passing = [r for r in rows if r['status'] == 'all_finite_published_cases_pass']
    if (dict(counts) != report['outcomes'] or len(passing) != report['passing_original_programs']
            or cases_passing != report['complete_original_cases_passing']):
        raise ValueError('Actual source evidence must reconcile with every reported count')
    result = dict(observed_at=datetime.now(timezone.utc).isoformat(), original_programs_reconciled=len(rows),
        passing_original_programs=len(passing), passing_complete_original_cases=cases_passing,
        outcomes=dict(counts), all_selected_original_IDs_program_hashes_and_test_hashes_reconciled=True,
        every_retained_raw_case_result_independently_compared_to_original_expected=True,
        complete_original_case_sets_required=True, actual_model_generations=0,
        training_or_preference_labels_admitted=False, trained_model_gain=False, goal_achieved=False,
        files_sha256={str(p):file_hash(p) for p in [Path(__file__).resolve(),folder/'plan.json',folder/'summary.json',folder/'observations.jsonl']})
    atomic_json(folder/'independent-readback.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='files_sha256'}),flush=True)


def check_test():
    folder = ROOT/'data/public-apps-publisher-test-evaluation-v1'
    manifest_path = folder/'manifest.json'
    report = json.loads(manifest_path.read_text())
    source = json.loads((ROOT/'data/public-apps-original-inventory-v1/test-problems.json').read_text())
    for name, expected in report['source_files_sha256'].items():
        if file_hash(name) != expected:
            raise ValueError('Actual test preparation source changed')
    actual_bytes = 0
    for name, expected in report['files_sha256'].items():
        path = folder/name
        if file_hash(path) != expected:
            raise ValueError('Actual complete held source file differs')
        if name.startswith('original-test/'):
            _, problem, filename = Path(name).parts
            original = source[problem]['files'][filename]
            if path.stat().st_size != original['bytes'] or expected != original['sha256']:
                raise ValueError('Actual test files must match their separately inventoried original archive members')
            actual_bytes += path.stat().st_size
    if list((folder/'original-test').glob('*/solutions.json')):
        raise ValueError('No original test reference program may enter the evaluation prompt store')
    rows = list(map(json.loads,(folder/'evaluation-records.jsonl').read_text().splitlines()))
    if len(rows) != 5000 or {r['problem'] for r in rows} != source.keys():
        raise ValueError('All5000actual original test problem identities required')
    tokenizer = Tokenizer.from_file(str(ROOT/'checkpoints/dialogue-retention-v1/systems-variety-code/tokenizer.json'))
    supported = 0
    modes = Counter()
    difficulties = Counter()
    all_cases = 0
    for row in rows:
        original = source[row['problem']]
        if (row['publisher_split'] != 'test' or row['purpose'] != 'evaluation_only'
                or not row['optimization_forbidden'] or row['source_program_reference_extracted']):
            raise ValueError('Original test sources must remain strictly outside optimization')
        base = folder/'original-test'/row['problem']
        tests_path = base/'input_output.json'
        tests = json.loads(tests_path.read_bytes()) if tests_path.stat().st_size <= 4*1024**2 else None
        starter = (base/'starter_code.py').read_bytes().decode() if (base/'starter_code.py').exists() else None
        fn = tests.get('fn_name') if tests else None
        interface = ('Use the original required function or class interface.' if starter or fn
            else 'Read standard input and write standard output as specified.')
        prompt = ('Write a Python solution to the following problem. '+interface+
            ' Return only the complete code.\n\n'+(base/'question.txt').read_bytes().decode())
        if starter:
            prompt += '\n\nOriginal starter code:\n'+starter
        if prompt != row['complete_prompt']:
            raise ValueError('Every full original question/starter prompt must reconstruct from actual source bytes')
        prefix = [tokenizer.token_to_id('<bos>')]+tokenizer.encode('User: '+prompt+'\nAssistant:').ids
        digest = hashlib.sha256(json.dumps(prefix,separators=(',',':')).encode()).hexdigest()
        if len(prefix) != row['complete_prompt_tokens'] or digest != row['complete_prompt_input_ids_sha256']:
            raise ValueError('Every actual whole own-tokenizer held prompt must reconstruct')
        if row['supported_by_declared_local_protocol']:
            if (row['review_reasons'] or len(prefix)+1024 > 4096 or tests is None
                    or len(tests['inputs']) != len(tests['outputs']) or len(tests['inputs']) != row['complete_published_case_count']):
                raise ValueError('Complete supported cases/whole prompt and exact context reserve required')
            supported += 1
            modes[row['mode']] += 1
            difficulties[row['difficulty']] += 1
            all_cases += len(tests['inputs'])
    if (supported != report['supported_complete_test_problems'] or all_cases != report['supported_complete_published_cases']
            or actual_bytes != report['original_test_bytes_retained'] or dict(modes) != report['supported_by_interface']
            or dict(difficulties) != report['supported_by_difficulty']):
        raise ValueError('Actual whole source/prompt/case counts must reconcile')
    result = dict(observed_at=datetime.now(timezone.utc).isoformat(), original_test_problems=5000,
        actual_original_source_bytes_rehashed=actual_bytes, original_test_source_files=report['original_test_files_retained'],
        all_original_test_files_reconciled_to_separate_archive_inventory=True,
        every_complete_prompt_and_own_BPE_span_independently_reconstructed=True,
        supported_complete_test_problems=supported, supported_complete_published_cases=all_cases,
        by_interface=dict(modes), by_difficulty=dict(difficulties),
        original_test_solution_files_extracted=0, original_prompts_or_cases_clipped=0,
        source_manifest_sha256=file_hash(manifest_path), model_generations=0, model_updates=0,
        optimization_forbidden=True, earlier_public_corpus_independence_certified=False,
        standard_full_APPS_score_claimed=False, goal_achieved=False,
        audit_source_sha256=file_hash(__file__))
    atomic_json(folder/'independent-evaluation-readback.json',result)
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    check_train()
    check_test()
