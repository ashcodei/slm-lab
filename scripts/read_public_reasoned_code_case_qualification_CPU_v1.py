"""Rebuild original source cases and independently check all saved values.

Waits for the exact complete execution owner. No code is executed by this
reader; it emits verified finite-case positives, never training admission.
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
SOURCE = Path('/tmp/slm-public-reasoned-code-case-qualification-v1')
SCREEN = Path('/tmp/slm-public-reasoned-code-screen-v2')
CACHE = Path('/tmp/slm-public-reasoned-code-train-questions-v1')
OUT = Path('/tmp/slm-public-reasoned-code-output-readback-v1')


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


def original_cases(payload):
    # Independent of the producer's contract()/comparison implementations.
    if not isinstance(payload, bytes) or len(payload) > 4 * 1024**2:
        raise ValueError('Whole original case payload outside declared bounds')

    def invalid(value):
        raise ValueError('Nonfinite original JSON number')

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate original JSON key')
            result[key] = value
        return result

    value = json.loads(payload, parse_constant=invalid, object_pairs_hook=pairs)
    json.dumps(value, allow_nan=False)
    inputs, outputs, function = value.get('inputs'), value.get('outputs'), value.get('fn_name')
    if (not isinstance(inputs, list) or not isinstance(outputs, list) or len(inputs) != len(outputs)
            or not 1 <= len(inputs) <= 1000):
        raise ValueError('Every original case required without subset selection')
    if function is not None:
        if (not isinstance(function, str) or not function.isidentifier() or function.startswith('__')
                or not all(isinstance(item, list) for item in inputs)):
            raise ValueError('Declared complete function argument schema differs')
        pending = [inputs, outputs]
        while pending:
            item = pending.pop()
            if isinstance(item, dict):
                raise ValueError('Original dictionary conversion outside declared local protocol')
            if isinstance(item, list):
                pending.extend(item)
        mode = 'function'
    else:
        def text(item):
            if isinstance(item, str):
                return item
            if isinstance(item, list) and all(isinstance(part, str) for part in item):
                return '\n'.join(item)
            raise ValueError('Complete original stdin/stdout text schema differs')
        inputs, outputs = [text(item) for item in inputs], [text(item) for item in outputs]
        mode = 'stdin'
    return dict(mode=mode, function=function, inputs=inputs, outputs=outputs,
        original_tests_sha256=hashlib.sha256(payload).hexdigest(), total_cases=len(inputs))


def main():
    if (OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) != 19
            or any(os.environ.get(k) != '1' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'))):
        raise ValueError('Fresh CUDA-hidden nice19 one-thread complete original-output readback required')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    tests = ROOT / 'reports/public-reasoned-code-output-readback-v1-tests.log'
    assert '3 passed' in tests.read_text()
    predecessor = json.loads((SOURCE / 'owner-plan.json').read_text())
    direct = [Path(__file__).resolve(), SOURCE / 'owner-plan.json', tests,
        ROOT / 'tests/test_public_reasoned_code_output_readback_v1.py',
        ROOT / 'src/slm_lab/public_reasoned_code_output_readback_v1.py']
    files = {str(path): sha(path) for path in direct}
    OUT.mkdir()

    def emit(name, value):
        with (OUT / name).open('x') as stream:
            stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

    owner = dict(pid=os.getpid(), birth=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19])
    emit('owner-plan.json', dict(owner=owner, at=datetime.now(timezone.utc).isoformat(), source_files_sha256=files,
        exact_predecessor=predecessor['owner'], complete_original_candidates_required=6116,
        original_source_cases_and_first_reference_rebuilt_independently=True,
        producer_comparators_not_imported=True, saved_outputs_not_new_program_executions=True,
        qualification_records_without_candidate_execution_not_incorrect_code_labels=True,
        CPU_threads=1, nice=19, scheduler='SCHED_IDLE',
        data_admitted_to_training=False, model_updates=0, original_workers_unchanged=True))
    try:
        while live(predecessor['owner']):
            print(json.dumps(dict(phase='waiting_exact_complete_source_execution', predecessor=predecessor['owner'])), flush=True)
            time.sleep(30)
        if (SOURCE / 'failure.json').exists() or not (SOURCE / 'complete.json').exists():
            raise ValueError('Exact source owner lacks complete successful execution; never restart it')
        done = json.loads((SOURCE / 'complete.json').read_text())
        assert done['complete'] and done['original_candidates'] == 6116 and done['all_fixed_candidate_positions_accounted_for']
        plan = json.loads((SOURCE / 'execution-plan.json').read_text())
        hashes_path = SOURCE / 'candidate-result-hashes.json'
        assert sha(hashes_path) == done['candidate_result_hashes_sha256']
        result_hashes = json.loads(hashes_path.read_text())
        screening = json.loads((SCREEN / 'complete.json').read_text())
        assert sha(SCREEN / 'every-original-row-screen.jsonl') == screening['ledger_sha256']
        selected = {}
        tasks = defaultdict(list)
        for line in (SCREEN / 'every-original-row-screen.jsonl').open():
            row = json.loads(line)
            if row['fresh_execution_pending']:
                selected[row['original_shard_position']] = row
                tasks[(row['dataset'], row['source_index'])].append(row)
        assert set(selected) == set(plan['candidate_positions']) and len(selected) == 6116 and len(tasks) == 3186
        expected_result_paths = {str(SOURCE / 'candidate-executions' / f'{position:05d}.json') for position in selected}
        assert set(result_hashes) == expected_result_paths
        files.update(done['source_files_sha256'])
        files.update(result_hashes)
        for path in [SOURCE / 'complete.json', SOURCE / 'execution-plan.json', hashes_path,
                SCREEN / 'complete.json', SCREEN / 'every-original-row-screen.jsonl']:
            files[str(path)] = sha(path)
        assert all(sha(path) == digest for path, digest in files.items())
        stat = dict(line.split()[:2] for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())
        if (shutil.disk_usage(OUT).free < 42 * 1024**3
                or int(Path('/sys/fs/cgroup/memory.max').read_text())
                    - sum(int(stat[k]) for k in ('anon', 'shmem', 'kernel')) < 56 * 1024**3):
            raise ValueError('8GiB own plus48GiB other-job memory and40GiB scratch reserve required')
        import pyarrow as pa
        import pyarrow.parquet as pq
        from slm_lab.code_contests_curriculum_candidates_v1 import question_hash
        from slm_lab.public_reasoned_code_output_readback_v1 import verify
        pa.set_cpu_count(1)
        pa.set_io_thread_count(1)
        cache = json.loads((CACHE / 'complete.json').read_text())
        questions = {}
        for line in Path(cache['questions_file']).open():
            row = json.loads(line)
            if (row['dataset'], row['index']) in tasks:
                questions[(row['dataset'], row['index'])] = row
        assert len(questions) == len(tasks)
        codes = {}
        position = 0
        for batch in pq.ParquetFile('/tmp/slm-public-reasoned-code-candidate-v1/original-python-00000.parquet').iter_batches(
                columns=['id', 'solution'], batch_size=32, use_threads=False):
            for row in batch.to_pylist():
                if position in selected:
                    assert row['id'] == selected[position]['id']
                    assert hashlib.sha256(row['solution'].encode()).hexdigest() == selected[position]['publisher_solution_sha256']
                    codes[position] = row['solution']
                position += 1
        assert position == 20000 and set(codes) == set(selected)
        profiles = {r['id']: r for r in json.loads((ROOT / 'reports/code-contests-complete-curriculum-candidate-v1/all-original-problem-source-dispositions.json').read_text())}
        originals = defaultdict(list)
        for line in (ROOT / 'data/code-contests-native-train-records-v1/all-original-problem-records.jsonl').open():
            row = json.loads(line)
            originals[(row['original_name'], profiles[row['id']]['complete_original_question_normalized_sha256'])].append(row)
        schema_path = ROOT / 'data/code-contests-protobuf-schema-v1/manifest.json'
        schema = json.loads(schema_path.read_text())
        sys.path.insert(0, schema['isolated_reader_site_packages'])
        sys.path.insert(0, str(schema_path.parent))
        import contest_problem_pb2 as proto
        counts = Counter()
        checked = set()
        positives = []
        role_groups = defaultdict(set)
        ledger = (OUT / 'every-fixed-candidate-readback.jsonl').open('x')

        def consume(key, original_source):
            cases = None
            reference_code = None
            for source in sorted(tasks[key], key=lambda r: r['original_shard_position']):
                position = source['original_shard_position']
                path = SOURCE / 'candidate-executions' / f'{position:05d}.json'
                assert sha(path) == result_hashes[str(path)]
                actual = json.loads(path.read_text())
                if (actual['original_shard_position'] != position or actual['id'] != source['id']
                        or actual['source_role'] != source['source_role']
                        or actual['fixed_pre_execution_role'] != source['fixed_pre_execution_role']
                        or actual['fitting_views'] != source['fitting_views']
                        or actual['data_admitted_to_training'] is not False):
                    raise ValueError('Every fixed original source identity, role and complete view must remain exact')
                result = dict(original_shard_position=position, id=source['id'], source_role=source['source_role'],
                    original_result_sha256=result_hashes[str(path)], finite_case_positive_verified=False,
                    original_status=actual['status'], data_admitted_to_training=False)
                if actual['status'] == 'whole_original_source_contract_qualification':
                    assert actual['candidate_execution_attempted'] is False
                    result['qualification_not_an_incorrect_code_label'] = True
                    result['unexecuted_source_qualification_reason_not_independently_replayed'] = True
                else:
                    if cases is None:
                        if key[0] == 'code_contests':
                            original_tests = [test for kind in ('public_tests', 'private_tests', 'generated_tests')
                                for test in getattr(original_source, kind)]
                            payload = json.dumps(dict(inputs=[t.input for t in original_tests], outputs=[t.output for t in original_tests]),
                                ensure_ascii=False, separators=(',', ':')).encode()
                            reference_code = next(solution.solution for solution in original_source.solutions if int(solution.language) == 3)
                            assert not any(original_source.HasField(field) and getattr(original_source, field)
                                for field in ('input_file', 'output_file'))
                        else:
                            payload = original_source['input_output'].encode()
                            reference_code = json.loads(original_source['solutions'])[0]
                        cases = original_cases(payload)
                    contract_path = SOURCE / 'source-contracts' / f'{key[0]}-{key[1]}.json'
                    reference_path = SOURCE / 'reference-executions' / f'{key[0]}-{key[1]}.json'
                    assert actual['source_contract_path'] == str(contract_path)
                    assert actual['original_reference_result_path'] == str(reference_path)
                    assert sha(contract_path) == actual['source_contract_sha256']
                    assert sha(reference_path) == actual['original_reference_result_sha256']
                    contract = json.loads(contract_path.read_text())
                    reference = json.loads(reference_path.read_text())
                    assert contract['dataset'] == key[0] and contract['original_source_index'] == key[1]
                    assert contract['cases'] == cases and contract['reference_code'] == reference_code
                    assert contract['original_source_role'] == 'train'
                    files[str(contract_path)] = actual['source_contract_sha256']
                    files[str(reference_path)] = actual['original_reference_result_sha256']
                    verified = verify(cases, reference_code, codes[position], reference, actual,
                        stdin_comparison=('stdin_exact_ordered_tokens' if key[0] == 'code_contests' else 'stdin_outside_whitespace_only'))
                    result.update(verified)
                    counts['returned_value_presentations_rechecked_including_reused_reference_controls'] += verified['checked_return_values']
                group = source['fixed_pre_execution_role']['declared_duplicate_component_sha256']
                role_groups[source['source_role']].add(group)
                if result['finite_case_positive_verified']:
                    positives.append(dict(original_shard_position=position, id=source['id'], source_role=source['source_role'],
                        fixed_pre_execution_role=source['fixed_pre_execution_role'], fitting_views=source['fitting_views'],
                        verified_original_result_sha256=result_hashes[str(path)]))
                    counts['verified_finite_case_positives_' + source['source_role']] += 1
                assert position not in checked
                checked.add(position)
                counts['fixed_candidates_checked'] += 1
                counts['original_status:' + actual['status']] += 1
                ledger.write(json.dumps(result, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n')
                if len(checked) % 100 == 0:
                    ledger.flush()
                    print(json.dumps(dict(phase='independent_original_case_and_output_readback', checked=len(checked), total=6116)), flush=True)

        receipts = {(r['dataset'], r['publisher_path']): r for r in json.loads((CACHE / 'downloads-complete.json').read_text())}
        for dataset in sorted({key[0] for key in tasks}):
            keys = sorted([key for key in tasks if key[0] == dataset], key=lambda k: k[1])
            if dataset == 'code_contests':
                for key in keys:
                    question = questions[key]
                    found = originals[(question['source_metadata']['name'], question_hash(question['question']))]
                    assert len(found) == 1
                    row = found[0]
                    with Path(row['extracted_record_file']).open('rb') as stream:
                        stream.seek(row['extracted_record_offset'])
                        payload = stream.read(row['extracted_record_bytes'])
                    assert hashlib.sha256(payload).hexdigest() == row['actual_original_record_sha256']
                    problem = proto.ContestProblem()
                    assert problem.ParseFromString(payload) == len(payload)
                    consume(key, problem)
                continue
            by_file = defaultdict(dict)
            for key in keys:
                by_file[questions[key]['original_file']][questions[key]['original_file_row']] = key
            for filename, local_keys in sorted(by_file.items()):
                receipt = receipts[(dataset, filename)]
                assert sha(receipt['local_path']) == receipt['sha256']
                if dataset == 'apps':
                    def rows():
                        with Path(receipt['local_path']).open('rb') as stream:
                            for line in stream:
                                yield json.loads(line)
                else:
                    def rows():
                        for batch in pq.ParquetFile(receipt['local_path']).iter_batches(
                                columns=['question', 'solutions', 'input_output'], batch_size=1, use_threads=False):
                            yield from batch.to_pylist()
                for local_index, raw in enumerate(rows()):
                    if local_index in local_keys:
                        key = local_keys[local_index]
                        assert raw['question'] == questions[key]['question']
                        consume(key, raw)
        ledger.close()
        assert checked == set(selected) and not role_groups['train'].intersection(role_groups['reserved'])
        assert all(sha(path) == digest for path, digest in files.items())
        positives.sort(key=lambda row: row['original_shard_position'])
        emit('verified-finite-case-positives.json', positives)
        emit('complete.json', dict(complete=True, owner=owner, at=datetime.now(timezone.utc).isoformat(),
            source_files_sha256=files, all6116_fixed_candidate_positions_reconciled=True, counts=dict(counts),
            positive_index_sha256=sha(OUT / 'verified-finite-case-positives.json'),
            ledger_sha256=sha(OUT / 'every-fixed-candidate-readback.jsonl'),
            original_reference_and_complete_case_bytes_rebuilt_for_all_scored_candidate_records=True,
            actual_values_recompared_without_importing_producer_comparators=True,
            original_stage_roles_and_views_preserved=True, declared_duplicate_component_cross_role_intersections=0,
            unexecuted_source_qualifications_preserved_not_certified_incorrect_programs=True,
            no_program_or_neural_execution_performed=True, confinement_not_independently_reexecuted=True,
            finite_case_agreement_not_semantic_correctness_or_efficiency_proof=True,
            complete_native_token_pack_and_independent_token_readback_still_required=True,
            data_admitted_to_training=False, RL_reward_admitted=False, model_updates=0,
            model_quality_gain_established=False, original_workers_unchanged=True, goal_achieved=False))
        print(json.dumps(dict(complete=True, counts=dict(counts))), flush=True)
    except BaseException as error:
        emit('failure.json', dict(owner=owner, error=repr(error), traceback=traceback.format_exc(),
            partial_outputs_retained=True, data_admitted_to_training=False, original_workers_unchanged=True))
        raise


if __name__ == '__main__':
    main()
