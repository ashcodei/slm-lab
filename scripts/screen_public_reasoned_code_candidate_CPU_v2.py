"""Screen the entire joined public shard before fresh execution or training.

Preserves all original rows, quarantines named held overlaps by duplicate
component, and fixes stage-only roles before execution outcomes are known.
Publisher pass metadata never becomes an independent correctness guarantee.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
JOIN = Path('/tmp/slm-public-reasoned-code-question-join-v2')
HELD = Path('/tmp/slm-public-reasoned-code-held-registry-v1')
QUESTIONS = Path('/tmp/slm-public-reasoned-code-train-questions-v1')
CANDIDATE = Path('/tmp/slm-public-reasoned-code-candidate-v1')
OUT = Path('/tmp/slm-public-reasoned-code-screen-v2')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(name, value):
    with (OUT / name).open('x') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


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
        raise ValueError('Fresh CUDA-hidden nice19 one-thread complete screening required')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    predecessor = json.loads((JOIN / 'owner-plan.json').read_text())
    held_receipt = json.loads((HELD / 'complete.json').read_text())
    assert held_receipt['complete'] and held_receipt['total_held_question_stems'] == 38791
    assert sha(HELD / 'held-question-registry.json') == held_receipt['held_registry_sha256']
    metadata_tests = ROOT / 'reports/public-reasoned-code-metadata-v1-tests.log'
    assert '2 passed' in metadata_tests.read_text()
    direct = [Path(__file__).resolve(), JOIN / 'owner-plan.json', HELD / 'complete.json',
        HELD / 'held-question-registry.json', QUESTIONS / 'complete.json',
        ROOT / 'src/slm_lab/held_text_overlap_v1.py',
        ROOT / 'src/slm_lab/public_code_candidate_roles_v1.py',
        ROOT / 'src/slm_lab/public_synthetic_code_screen_policy_v1.py',
        ROOT / 'src/slm_lab/code_contests_curriculum_candidates_v1.py',
        ROOT / 'src/slm_lab/code_tasks.py',
        ROOT / 'data/verified-code-source-cache-v1/sanitized-mbpp.json',
        ROOT / 'data/verified-code-expanded-v1/reserved-mbpp.jsonl',
        ROOT / 'src/slm_lab/public_reasoned_code_metadata_v1.py',
        ROOT / 'tests/test_public_reasoned_code_metadata_v1.py', metadata_tests,
        ROOT / 'scripts/screen_public_reasoned_code_candidate_CPU_v1.py',
        Path('/tmp/slm-public-reasoned-code-screen-v1/complete.json'),
        Path('/tmp/slm-public-reasoned-code-screen-v1.log')]
    inputs = {str(path): sha(path) for path in direct}
    OUT.mkdir()
    owner = dict(pid=os.getpid(), birth=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19])
    write('owner-plan.json', dict(owner=owner, at=datetime.now(timezone.utc).isoformat(), source_files_sha256=inputs,
        exact_predecessor=predecessor['owner'], complete_original_rows=20000,
        phase='wait for exact predecessor, then screen every original row',
        source_roles_fixed_before_fresh_execution=True, source_role_scope='new stage only; historical exposure not certified',
        only_reported_pass_rate_one_parseable_whole_context_fitting_candidates_considered=True,
        publisher_string_numeric_metadata_parsed_with_finite_range_validation=True,
        original_join_two_strict_numeric_metadata_counters_not_used=True,
        superseded_zero_candidate_v1_source_and_results_retained=True,
        entire_duplicate_component_quarantined_if_any_selected_view_has_protected_overlap=True,
        original_data_and_held_assessments_unchanged=True, CPU_threads=1, nice=19, scheduler='SCHED_IDLE',
        data_admitted_to_training=False, fresh_program_executions=0, model_updates=0))
    try:
        while live(predecessor['owner']):
            print(json.dumps(dict(phase='waiting_exact_question_join', predecessor=predecessor['owner'])), flush=True)
            time.sleep(30)
        if (JOIN / 'failure.json').exists() or not (JOIN / 'complete.json').exists():
            raise ValueError('Exact predecessor ended without complete successful source evidence; never restart it')
        joined = json.loads((JOIN / 'complete.json').read_text())
        assert joined['complete'] and joined['all_complete_upstream_train_questions_joined']
        assert joined['counts']['original_rows'] == 20000 and joined['counts']['whole_train_question_joined'] == 17424
        assert sha(JOIN / 'every-candidate-disposition.jsonl') == joined['ledger_sha256']
        cache = json.loads((QUESTIONS / 'complete.json').read_text())
        question_file = Path(cache['questions_file'])
        assert sha(question_file) == cache['questions_file_sha256']
        inputs.update(joined['source_files_sha256'])
        inputs.update(held_receipt['source_files_sha256'])
        inputs.update({str(path): sha(path) for path in [JOIN / 'complete.json', JOIN / 'every-candidate-disposition.jsonl', question_file]})
        assert all(sha(path) == value for path, value in inputs.items())
        stat = dict(line.split()[:2] for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())
        nonfile = sum(int(stat[k]) for k in ('anon', 'shmem', 'kernel'))
        if (shutil.disk_usage(OUT).free < 42 * 1024**3
                or int(Path('/sys/fs/cgroup/memory.max').read_text()) - nonfile < 56 * 1024**3):
            raise ValueError('8GiB own plus48GiB other-job memory and40GiB scratch reserve required')
        import pyarrow as pa
        import pyarrow.parquet as pq
        from slm_lab.code_tasks import TASKS
        from slm_lab.code_contests_curriculum_candidates_v1 import normalized_program_hash
        from slm_lab.held_text_overlap_v1 import HeldTextOverlap, normalize
        from slm_lab.public_code_candidate_roles_v1 import assign_roles, verify_role_separation
        from slm_lab.public_synthetic_code_screen_policy_v1 import body_fingerprint
        from slm_lab.public_reasoned_code_metadata_v1 import normalized_pass_rate, reported_all_pass
        pa.set_cpu_count(1)
        pa.set_io_thread_count(1)
        registry = json.loads((HELD / 'held-question-registry.json').read_text())
        overlap = HeldTextOverlap(registry)
        protected_programs = defaultdict(list)
        protected_ASTs = defaultdict(list)

        def protect(code, identity):
            protected_programs[normalized_program_hash(code)].append(identity)
            try:
                fingerprint = body_fingerprint(code)
            except (SyntaxError, ValueError):
                return
            protected_ASTs[fingerprint].append(identity)

        for row in json.loads((ROOT / 'data/verified-code-source-cache-v1/sanitized-mbpp.json').read_text()):
            if 11 <= row['task_id'] <= 510:
                protect(row['code'], 'sanitized-MBPP:' + str(row['task_id']))
        for line in (ROOT / 'data/verified-code-expanded-v1/reserved-mbpp.jsonl').open():
            row = json.loads(line)
            protect(row['code'], 'reserved-MBPP:' + str(row['task_id']))
        for name, args, spec, scaling, python_code, js_code in TASKS:
            protect(python_code, 'local-code:Python:' + name)
            protect(js_code, 'local-code:JavaScript:' + name)
        questions = {}
        for line in question_file.open():
            row = json.loads(line)
            questions[(row['dataset'], row['split'], row['index'])] = row['question']
        write('execution-plan.json', dict(at=datetime.now(timezone.utc).isoformat(), source_files_sha256=inputs,
            held_overlap_configuration=overlap.configuration, pyarrow_version=pa.__version__,
            pyahocorasick_version=importlib.metadata.version('pyahocorasick'),
            protected_normalized_program_fingerprints=len(protected_programs),
            protected_docstring_free_AST_fingerprints=len(protected_ASTs),
            lexical_screen='complete original question; literal screen also covers each entire fitting prompt/target',
            no_held_gold_value_or_model_score_used_for_gradient_or_selection=True,
            metadata_positives_not_independent_execution=True, data_admitted_to_training=False))
        records = []
        candidates = []
        question_overlap = {}
        counts = Counter()
        normalized_reported_ones = 0
        columns = ['id', 'question_id', 'dataset', 'split', 'index', 'solution', 'r1_generation', 'pass_rate']
        with (JOIN / 'every-candidate-disposition.jsonl').open() as join_ledger:
            for batch in pq.ParquetFile(CANDIDATE / 'original-python-00000.parquet').iter_batches(
                    columns=columns, batch_size=16, use_threads=False):
                for row in batch.to_pylist():
                    position = len(records)
                    source = json.loads(join_ledger.readline())
                    assert source['position'] == position and source['id'] == row['id'] and source['question_id'] == row['question_id']
                    assert (source['dataset'], source['split'], source['index']) == (row['dataset'], row['split'], int(row['index']))
                    result = dict(original_shard_position=position, id=row['id'], question_id=row['question_id'],
                        dataset=row['dataset'], split=row['split'], source_index=int(row['index']),
                        publisher_pass_rate_raw=row['pass_rate'],
                        publisher_pass_rate_normalized=normalized_pass_rate(row['pass_rate']),
                        status='upstream_non_train_quarantined', protected_overlap=False,
                        source_role='excluded', training_admitted=False, fresh_execution_pending=False)
                    records.append(result)
                    normalized_reported_ones += reported_all_pass(row['pass_rate'])
                    if row['split'] != 'train':
                        continue
                    result['status'] = 'publisher_not_reported_all_pass_abstention'
                    if not reported_all_pass(row['pass_rate']):
                        continue
                    code = row['solution']
                    result['status'] = 'empty_or_unparseable_publisher_solution_abstention'
                    if not isinstance(code, str) or not code.strip():
                        continue
                    try:
                        fingerprint = body_fingerprint(code)
                    except (SyntaxError, ValueError):
                        continue
                    fitting = [name for name, view in source['views'].items() if view['complete_context_fits_4096']]
                    # The executed code field must actually occur in the whole
                    # reasoning target before that target can share its label.
                    if 'reasoning' in fitting and code.strip() not in row['r1_generation']:
                        fitting.remove('reasoning')
                        result['reasoning_code_identity_abstention'] = True
                    result['status'] = 'no_complete_fitting_target_view_abstention'
                    if not fitting:
                        continue
                    question = questions[(row['dataset'], row['split'], int(row['index']))]
                    result['status'] = 'source_NUL_abstention'
                    if any('\0' in text for text in [question, code] + [source['views'][name]['prompt'] for name in fitting]
                            + [row['r1_generation']] * ('reasoning' in fitting)):
                        continue
                    question_key = hashlib.sha256(normalize(question).encode()).hexdigest()
                    if question_key not in question_overlap:
                        question_overlap[question_key] = dict(literal=overlap.literal(question), lexical=overlap.lexical_copies(question))
                    qmatches = question_overlap[question_key]
                    view_matches = {}
                    for name in fitting:
                        target = code if name == 'code_only' else row['r1_generation']
                        view = source['views'][name]
                        assert hashlib.sha256(target.encode()).hexdigest() == view['entire_target_utf8_sha256']
                        view_matches[name] = dict(prompt_literal=overlap.literal(view['prompt']), target_literal=overlap.literal(target))
                    program_matches = protected_programs.get(normalized_program_hash(code), [])
                    AST_matches = protected_ASTs.get(fingerprint, [])
                    prior_role_flag = (source['known_held_source_component'] or source['APPS_internal_validation']
                        or bool(source['known_excluded_CodeContests_normalized_question_matches']))
                    flagged = bool(prior_role_flag or qmatches['literal'] or qmatches['lexical'] or program_matches or AST_matches
                        or any(value['prompt_literal'] or value['target_literal'] for value in view_matches.values()))
                    result.update(status='metadata_positive_whole_view_pending_component_screen',
                        input_only_group_sha256=question_key, program_AST_without_docstrings_sha256=fingerprint,
                        publisher_solution_sha256=hashlib.sha256(code.encode()).hexdigest(), fitting_views=fitting,
                        protected_overlap=flagged, known_held_source_role_flag=bool(prior_role_flag),
                        complete_question_overlap=qmatches, complete_fitting_view_literal_overlap=view_matches,
                        protected_normalized_program_matches=program_matches, protected_AST_matches=AST_matches)
                    candidates.append(result)
                print(json.dumps(dict(phase='whole_shard_overlap_screen', rows=len(records), total=20000)), flush=True)
            assert join_ledger.readline() == ''
        assert len(records) == 20000 and normalized_reported_ones == 11806
        roles = assign_roles(candidates)
        separation = verify_role_separation(candidates, roles)
        bad_components = {roles[row['original_shard_position']]['declared_duplicate_component_sha256']
            for row in candidates if row['protected_overlap']}
        for row in candidates:
            role = roles[row['original_shard_position']]
            row['fixed_pre_execution_role'] = role
            if role['declared_duplicate_component_sha256'] in bad_components:
                row['status'] = 'protected_overlap_duplicate_component_quarantined'
            else:
                row.update(status='screened_metadata_positive_pending_fresh_execution', source_role=role['source_role'],
                    fresh_execution_pending=True, optimization_forbidden=role['optimization_forbidden'])
        with (OUT / 'every-original-row-screen.jsonl').open('x') as ledger:
            for row in records:
                counts[row['status']] += 1
                if row['fresh_execution_pending']:
                    counts['pending_execution_' + row['source_role']] += 1
                    for name in row['fitting_views']:
                        counts['pending_execution_' + row['source_role'] + '_' + name + '_views'] += 1
                ledger.write(json.dumps(row, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n')
        assert all(sha(path) == value for path, value in inputs.items())
        write('complete.json', dict(complete=True, owner=owner, at=datetime.now(timezone.utc).isoformat(),
            source_files_sha256=inputs, original_rows=20000, counts=dict(counts), role_separation=separation,
            all_original_reported_pass_one_metadata_rows=normalized_reported_ones,
            string_numeric_metadata_normalized_before_positive_selection=True,
            quarantined_duplicate_components=len(bad_components),
            ledger_sha256=sha(OUT / 'every-original-row-screen.jsonl'),
            all_original_rows_preserved=True, full_held_questions_and_selected_view_inputs_screened=True,
            only_declared_literal_lexical_exact_program_AST_and_original_role_signals_certified=True,
            stage_reserved_roles_do_not_certify_historical_or_semantic_independence=True,
            publisher_metadata_not_independent_execution=True, fresh_program_executions=0,
            independent_execution_qualification_and_readback_required_before_SFT_admission=True,
            RL_reward_and_human_preference_admission_forbidden=True,
            data_admitted_to_training=False, original_model_workers_unchanged=True,
            model_updates=0, model_quality_gain_established=False, goal_achieved=False))
        print(json.dumps(dict(complete=True, counts=dict(counts), role_separation=separation)), flush=True)
    except BaseException as error:
        write('failure.json', dict(owner=owner, error=repr(error), traceback=traceback.format_exc(),
            partial_outputs_retained=True, data_admitted_to_training=False, original_workers_unchanged=True))
        raise


if __name__ == '__main__':
    main()
