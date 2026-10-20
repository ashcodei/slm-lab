"""Run all screened candidates against complete original-source test sets.

One confined CPU child at a time. Every fixed train/reserved candidate gets a
result; unsupported whole sources abstain. No training or reward admission.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
SCREEN = Path('/tmp/slm-public-reasoned-code-screen-v2')
CACHE = Path('/tmp/slm-public-reasoned-code-train-questions-v1')
OUT = Path('/tmp/slm-public-reasoned-code-case-qualification-v1')
GIB = 1024**3


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def live(owner):
    path = Path('/proc') / str(owner['pid']) / 'stat'
    try:
        fields = path.read_text().rsplit(')', 1)[1].split()
    except FileNotFoundError:
        return False
    return fields[19] == str(owner['birth']) and fields[0] != 'Z'


def main():
    if (OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) != 19
            or any(os.environ.get(k) != '1' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'))):
        raise ValueError('Fresh CUDA-hidden nice19 single-child full-case qualification required')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    from slm_lab.public_reasoned_code_cases_v1 import contract, execute, implementation_sources, qualify_reference, score_candidate
    tests = ROOT / 'reports/public-reasoned-code-cases-v1-tests.log'
    assert '3 passed' in tests.read_text()
    predecessor = json.loads((SCREEN / 'owner-plan.json').read_text())
    inputs = implementation_sources()
    for path in [Path(__file__).resolve(), SCREEN / 'owner-plan.json', tests,
            ROOT / 'tests/test_public_reasoned_code_cases_v1.py',
            ROOT / 'src/slm_lab/public_reasoned_code_cases_v1.py', CACHE / 'complete.json',
            CACHE / 'downloads-complete.json', ROOT / 'src/slm_lab/code_contests_curriculum_candidates_v1.py']:
        inputs[str(path)] = sha(path)
    if shutil.disk_usage(OUT.parent).free < 168 * GIB:
        raise ValueError('128GiB output allowance above40GiB original scratch floor required')
    OUT.mkdir()
    written = 0

    def emit(relative, value):
        nonlocal written
        raw = (json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n').encode()
        if written + len(raw) > 128 * GIB or shutil.disk_usage(OUT).free < 40 * GIB:
            raise ValueError('Whole-output allowance or original scratch floor exhausted; preserve partial results')
        path = OUT / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(raw)
        written += len(raw)
        return path

    owner = dict(pid=os.getpid(), birth=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19])
    emit('owner-plan.json', dict(owner=owner, at=datetime.now(timezone.utc).isoformat(), source_files_sha256=inputs,
        exact_predecessor=predecessor['owner'], phase='wait for complete corrected full-shard screening',
        source_selection='every screened metadata-positive candidate in both original fixed train/reserved roles',
        original_reference='first publisher Python3 solution for CodeContests; first publisher solution for APPS/TACO',
        no_fallback_to_another_reference_after_qualification=True, all_original_cases_required=True,
        maximum_cases=1000, maximum_complete_case_payload_bytes=4 * 1024**2,
        maximum_seconds_per_whole_program=20, maximum_concurrent_confined_children=1,
        scratch_output_allowance_bytes=128 * GIB, scratch_floor_bytes=40 * GIB,
        CPU_threads=1, nice=19, scheduler='SCHED_IDLE',
        expected_values_never_sent_to_confined_children=True, original_source_bytes_not_modified=True,
        data_admitted_to_training=False, model_updates=0, original_workers_unchanged=True))
    counts = Counter()
    try:
        while live(predecessor['owner']):
            print(json.dumps(dict(phase='waiting_exact_corrected_screen', predecessor=predecessor['owner'])), flush=True)
            time.sleep(30)
        if (SCREEN / 'failure.json').exists() or not (SCREEN / 'complete.json').exists():
            raise ValueError('Exact screening predecessor lacks successful complete evidence; never restart it')
        screening = json.loads((SCREEN / 'complete.json').read_text())
        assert screening['complete'] and screening['original_rows'] == 20000
        assert screening['all_original_reported_pass_one_metadata_rows'] == 11806
        assert sha(SCREEN / 'every-original-row-screen.jsonl') == screening['ledger_sha256']
        inputs.update(screening['source_files_sha256'])
        for path in [SCREEN / 'complete.json', SCREEN / 'every-original-row-screen.jsonl']:
            inputs[str(path)] = sha(path)
        cache = json.loads((CACHE / 'complete.json').read_text())
        question_file = Path(cache['questions_file'])
        assert sha(question_file) == cache['questions_file_sha256']
        inputs[str(question_file)] = cache['questions_file_sha256']
        stat = dict(line.split()[:2] for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())
        if int(Path('/sys/fs/cgroup/memory.max').read_text()) - sum(int(stat[k]) for k in ('anon', 'shmem', 'kernel')) < 56 * GIB:
            raise ValueError('8GiB own plus48GiB other-job nonfile memory reserve required')
        import pyarrow as pa
        import pyarrow.parquet as pq
        from slm_lab.code_contests_curriculum_candidates_v1 import question_hash
        pa.set_cpu_count(1)
        pa.set_io_thread_count(1)
        selected = {}
        tasks = defaultdict(list)
        for line in (SCREEN / 'every-original-row-screen.jsonl').open():
            row = json.loads(line)
            if not row['fresh_execution_pending']:
                continue
            assert row['status'] == 'screened_metadata_positive_pending_fresh_execution'
            assert row['split'] == 'train' and row['source_role'] in ('train', 'reserved')
            assert row['fixed_pre_execution_role']['optimization_forbidden'] is (row['source_role'] == 'reserved')
            selected[row['original_shard_position']] = row
            tasks[(row['dataset'], row['source_index'])].append(row)
        questions = {}
        for line in question_file.open():
            row = json.loads(line)
            if (row['dataset'], row['index']) in tasks:
                questions[(row['dataset'], row['index'])] = row
        assert len(questions) == len(tasks)
        codes = {}
        position = 0
        candidate_path = Path('/tmp/slm-public-reasoned-code-candidate-v1/original-python-00000.parquet')
        for batch in pq.ParquetFile(candidate_path).iter_batches(columns=['id', 'solution'], batch_size=32, use_threads=False):
            for row in batch.to_pylist():
                if position in selected:
                    assert row['id'] == selected[position]['id']
                    assert hashlib.sha256(row['solution'].encode()).hexdigest() == selected[position]['publisher_solution_sha256']
                    codes[position] = row['solution']
                position += 1
        assert position == 20000 and set(codes) == set(selected)
        records_path = ROOT / 'data/code-contests-native-train-records-v1/all-original-problem-records.jsonl'
        profiles_path = ROOT / 'reports/code-contests-complete-curriculum-candidate-v1/all-original-problem-source-dispositions.json'
        schema_path = ROOT / 'data/code-contests-protobuf-schema-v1/manifest.json'
        for path in [records_path, profiles_path, schema_path]:
            inputs[str(path)] = sha(path)
        profiles = {r['id']: r for r in json.loads(profiles_path.read_text())}
        original = defaultdict(list)
        for line in records_path.open():
            row = json.loads(line)
            original[(row['original_name'], profiles[row['id']]['complete_original_question_normalized_sha256'])].append(row)
        schema = json.loads(schema_path.read_text())
        for name, digest in schema['files_sha256'].items():
            inputs[str(schema_path.parent / name)] = digest
        sys.path.insert(0, schema['isolated_reader_site_packages'])
        sys.path.insert(0, str(schema_path.parent))
        import contest_problem_pb2 as proto
        assert all(sha(path) == value for path, value in inputs.items())
        emit('execution-plan.json', dict(at=datetime.now(timezone.utc).isoformat(), source_files_sha256=inputs,
            original_candidates=len(selected), original_problem_sources=len(tasks),
            candidate_positions=sorted(selected), roles_fixed_by_closed_screen=True,
            source_processing_order='dataset lexicographic, publisher path order, original file row; candidate position order',
            runtime='Existing stdlib-only Landlock/seccomp confined whole-program harness; fresh stdin namespace per case',
            function_output_convention='reference-fixed exact JSON value or uniform singleton wrapper; never per-candidate alternatives',
            stdin_comparison='CodeContests exact ordered byte tokens; APPS/TACO outside whitespace only',
            no_op_control_must_complete_and_fail_at_least_one_case_before_candidate_labels=True,
            complete_reasoning_text_not_independently_verified_by_code_test_agreement=True,
            finite_case_passing_not_algorithmic_efficiency_or_semantic_proof=True, data_admitted_to_training=False))
        completed = set()
        output_hashes = {}

        def qualify(key, raw, source_details):
            rows = sorted(tasks[key], key=lambda r: r['original_shard_position'])
            slug = key[0] + '-' + str(key[1])
            base = dict(dataset=key[0], original_source_index=key[1], source_details=source_details,
                original_source_role='train', stage_roles_fixed_before_execution=True)
            try:
                if key[0] == 'code_contests':
                    if any(raw.HasField(field) and getattr(raw, field) for field in ('input_file', 'output_file')):
                        raise ValueError('Original file-based interface outside declared stdin protocol')
                    refs = [solution.solution for solution in raw.solutions if int(solution.language) == 3]
                    all_tests = [case for kind in ('public_tests', 'private_tests', 'generated_tests') for case in getattr(raw, kind)]
                    payload = json.dumps(dict(inputs=[c.input for c in all_tests], outputs=[c.output for c in all_tests]),
                        ensure_ascii=False, separators=(',', ':')).encode()
                    source_details['complete_original_test_counts'] = {kind: len(getattr(raw, kind))
                        for kind in ('public_tests', 'private_tests', 'generated_tests')}
                else:
                    if not isinstance(raw.get('solutions'), str) or not isinstance(raw.get('input_output'), str):
                        raise ValueError('Whole original reference list and test payload must be strings')
                    refs = json.loads(raw['solutions'])
                    payload = raw['input_output'].encode()
                if not isinstance(refs, list) or not refs or not isinstance(refs[0], str):
                    raise ValueError('First declared original reference unavailable; no fallback')
                reference_code = refs[0]
                if not reference_code or len(reference_code.encode()) > 512 * 1024:
                    raise ValueError('Whole first original reference exceeds declared source bound')
                cases = contract(payload)
                contract_path = emit('source-contracts/' + slug + '.json', dict(**base, cases=cases,
                    complete_original_case_payload_sha256=hashlib.sha256(payload).hexdigest(), reference_code=reference_code))
                observation = execute(reference_code, cases, seconds=20)
                counts['reference_program_executions'] += 1
                fixed = qualify_reference(cases, observation, stdin_comparison=(
                    'stdin_exact_ordered_tokens' if key[0] == 'code_contests' else 'stdin_outside_whitespace_only'))
                control = None
                control_score = None
                if fixed['reference_qualified']:
                    noop = 'pass\n' if cases['mode'] == 'stdin' else 'def ' + cases['function'] + '(*args, **kwargs):\n    return None\n'
                    control = execute(noop, cases, seconds=20)
                    counts['no_op_program_executions'] += 1
                    control_score = score_candidate(cases, control, fixed)
                reference_path = emit('reference-executions/' + slug + '.json', dict(**base,
                    source_contract_sha256=sha(contract_path), original_reference_observation=observation,
                    fixed_output_convention=fixed, no_op_observation=control, no_op_score=control_score))
                usable = (fixed['reference_qualified'] and control_score is not None
                    and control_score['all_cases_observed'] and not control_score['all_finite_cases_pass'])
                for row in rows:
                    record = dict(original_shard_position=row['original_shard_position'], id=row['id'],
                        source_role=row['source_role'], fixed_pre_execution_role=row['fixed_pre_execution_role'],
                        fitting_views=row['fitting_views'], source_contract_path=str(contract_path),
                        source_contract_sha256=sha(contract_path), original_reference_result_path=str(reference_path),
                        original_reference_result_sha256=sha(reference_path), data_admitted_to_training=False,
                        status='original_reference_or_no_op_population_qualification', candidate_execution_attempted=False)
                    if usable:
                        actual = execute(codes[row['original_shard_position']], cases, seconds=20)
                        counts['candidate_program_executions'] += 1
                        score = score_candidate(cases, actual, fixed)
                        record.update(status=score['status'], candidate_execution_attempted=True,
                            candidate_observation=actual, candidate_score=score)
                    retain(row, record)
            except (ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
                # Resource/source-contract limitations abstain; they never
                # silently crop a test set or label the original code wrong.
                for row in rows:
                    if row['original_shard_position'] in completed:
                        continue
                    retain(row, dict(**base, original_shard_position=row['original_shard_position'], id=row['id'],
                        source_role=row['source_role'], fixed_pre_execution_role=row['fixed_pre_execution_role'],
                        fitting_views=row['fitting_views'], status='whole_original_source_contract_qualification',
                        diagnostic=str(error), candidate_execution_attempted=False, data_admitted_to_training=False))

        def retain(row, result):
            position = row['original_shard_position']
            assert position not in completed
            path = emit('candidate-executions/' + f'{position:05d}.json', result)
            output_hashes[str(path)] = sha(path)
            completed.add(position)
            counts[result['status']] += 1
            if result.get('candidate_score', {}).get('all_finite_cases_pass'):
                counts['finite_case_positives_' + row['source_role']] += 1
            if len(completed) % 25 == 0:
                print(json.dumps(dict(phase='complete_original_case_qualification', completed=len(completed),
                    total=len(selected), counts=dict(counts))), flush=True)

        downloads = json.loads((CACHE / 'downloads-complete.json').read_text())
        receipts = {(r['dataset'], r['publisher_path']): r for r in downloads}
        for dataset in sorted({key[0] for key in tasks}):
            keys = sorted([key for key in tasks if key[0] == dataset], key=lambda k: k[1])
            if dataset == 'code_contests':
                for key in keys:
                    question = questions[key]
                    found = original[(question['source_metadata']['name'], question_hash(question['question']))]
                    assert len(found) == 1
                    row = found[0]
                    with Path(row['extracted_record_file']).open('rb') as stream:
                        stream.seek(row['extracted_record_offset'])
                        payload = stream.read(row['extracted_record_bytes'])
                    assert hashlib.sha256(payload).hexdigest() == row['actual_original_record_sha256']
                    problem = proto.ContestProblem()
                    assert problem.ParseFromString(payload) == len(payload)
                    qualify(key, problem, dict(original_record_id=row['id'], original_record_sha256=row['actual_original_record_sha256']))
                continue
            by_file = defaultdict(dict)
            for key in keys:
                question = questions[key]
                by_file[question['original_file']][question['original_file_row']] = key
            for filename, local_keys in sorted(by_file.items()):
                receipt = receipts[(dataset, filename)]
                assert sha(receipt['local_path']) == receipt['sha256']
                inputs[receipt['local_path']] = receipt['sha256']
                if dataset == 'apps':
                    def read_rows():
                        with Path(receipt['local_path']).open('rb') as source:
                            for line in source:
                                yield json.loads(line)
                else:
                    def read_rows():
                        for batch in pq.ParquetFile(receipt['local_path']).iter_batches(
                                columns=['question', 'solutions', 'input_output'], batch_size=1, use_threads=False):
                            yield from batch.to_pylist()
                for local_index, raw in enumerate(read_rows()):
                    if local_index not in local_keys:
                        continue
                    key = local_keys[local_index]
                    assert raw['question'] == questions[key]['question']
                    qualify(key, raw, dict(original_file=filename, original_file_row=local_index,
                        original_file_sha256=receipt['sha256']))
        assert completed == set(selected)
        assert all(sha(path) == value for path, value in inputs.items())
        emit('candidate-result-hashes.json', output_hashes)
        emit('complete.json', dict(complete=True, owner=owner, at=datetime.now(timezone.utc).isoformat(),
            source_files_sha256=inputs, original_candidates=len(selected), original_problem_sources=len(tasks),
            all_fixed_candidate_positions_accounted_for=True, counts=dict(counts),
            candidate_result_hashes_sha256=sha(OUT / 'candidate-result-hashes.json'),
            original_full_source_populations_not_clipped=True, original_stage_roles_unchanged=True,
            independent_returned_value_and_source_readback_required_before_admission=True,
            data_admitted_to_training=False, RL_reward_admitted=False, model_updates=0,
            model_quality_gain_established=False, original_workers_unchanged=True, goal_achieved=False))
        print(json.dumps(dict(complete=True, candidates=len(selected), counts=dict(counts))), flush=True)
    except BaseException as error:
        emit('failure.json', dict(owner=owner, error=repr(error), traceback=traceback.format_exc(),
            partial_outputs_retained=True, data_admitted_to_training=False, original_workers_unchanged=True))
        raise


if __name__ == '__main__':
    main()
