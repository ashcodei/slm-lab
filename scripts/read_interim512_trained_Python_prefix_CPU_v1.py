"""Compare the first five delivered trained Python completions with baseline.

The cohort is the untouched first five tasks in the original fixed order.
No generation, repair, optimization or change to the full96producer occurs.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import traceback

from read_interim512_initial_Python_deliveries_CPU_v1 import sha, write

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('/tmp/slm-broad-code-step512-code-evaluation-CPU-v1')
SNAPSHOT = Path('/tmp/slm-broad-code-step512-snapshot-CPU-v1')
BASELINE = ROOT / 'reports/interim512-initial-Python-deliveries-CPU-v1'
OUT = ROOT / 'reports/interim512-trained-Python-prefix-CPU-v1'
COUNT = 5


def main():
    if (OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) != 19
            or any(os.environ.get(k) != '1' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'))):
        raise ValueError('Fresh CUDA-hidden low-priority one-thread readback required')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    from tokenizers import Tokenizer
    from slm_lab.code_tasks import TASKS, request
    from slm_lab.decoded_native_code_reexecution_v1 import reconstruct
    from slm_lab.local_source_closure_v2 import python_sources
    from slm_lab.ordinary_python_function_cases_v1 import execute, implementation_sources

    producer_path = SOURCE / 'owner-plan.json'
    producer = json.loads(producer_path.read_text())
    baseline_path = BASELINE / 'complete.json'
    baseline = json.loads(baseline_path.read_text())
    assert baseline['complete'] and baseline['summary']['programs'] == 12
    assert producer['interim_updates'] == 512 and producer['total_programs'] == 96
    assert producer['precision'] == 'FP32' and producer['generation_budget'] == 512 and len(TASKS) == 12
    files = dict(baseline['source_files_sha256'])
    paths = [producer_path, baseline_path, SOURCE / 'interim-source-state-readback.json', SNAPSHOT / 'tokenizer.json']
    paths += python_sources(ROOT, [Path(__file__)])
    paths += [BASELINE / (task[0] + '.json') for task in TASKS[:COUNT]]
    for path in set(paths):
        digest = sha(path)
        assert str(path) not in files or files[str(path)] == digest
        files[str(path)] = digest
    files.update(implementation_sources())
    for path, digest in files.items():
        assert sha(path) == digest
        assert path not in producer['source_files_sha256'] or producer['source_files_sha256'][path] == digest
    tokenizer = Tokenizer.from_file(str(SNAPSHOT / 'tokenizer.json'))
    vocab_size = tokenizer.get_vocab_size()
    journal = SOURCE / 'actual-native-actions.jsonl'
    with journal.open('rb') as stream:
        original = os.fstat(stream.fileno())
        lines = [stream.readline() for _ in range(48 + COUNT)]
    if any(not row.endswith(b'\n') for row in lines):
        raise ValueError('All first five whole trained deliveries must already exist')
    prefix = b''.join(lines)
    actions = [json.loads(row) for row in lines]
    paired = []

    def decode(task, action, index, endpoint):
        name, args, specification = task[:3]
        assert (action['endpoint'], action['frame'], action['language'], action['case'], action['task']) == (
            endpoint, 'completion', 'Python', index, name)
        prompt = f'def {name}({args}):\n    """{specification}"""\n'
        assert action['complete_prompt_ids'] == [tokenizer.token_to_id('<bos>')] + tokenizer.encode(prompt).ids
        ids = action['actual_response_ids']
        assert 0 < len(ids) <= 512 and all(type(v) is int and 0 <= v < vocab_size for v in ids)
        assert 2 not in ids[:-1] and action['stopped_on_EOS'] == (ids[-1] == 2)
        assert action['ended_on_budget'] == (len(ids) == 512 and ids[-1] != 2)
        raw = tokenizer.decode(ids)
        code = re.split(r'\n(?=File:)', prompt + raw.split('```')[0])[0]
        case = dict(language='Python', task=name, specification=specification, raw_response=raw, code=code)
        actual_task, actual_code = reconstruct(case, 'completion')
        assert actual_task == task and actual_code == code
        return case

    for index, task in enumerate(TASKS[:COUNT], 1):
        before, after = actions[index - 1], actions[48 + index - 1]
        before_case = decode(task, before, index, 'initial')
        after_case = decode(task, after, index, 'step512')
        assert before['complete_prompt_ids'] == after['complete_prompt_ids']
        baseline_result = json.loads((BASELINE / (task[0] + '.json')).read_text())
        assert files[str(BASELINE / (task[0] + '.json'))] == baseline['result_files'][str(BASELINE / (task[0] + '.json'))]
        assert baseline_result['actual_actions'] == before and baseline_result['delivered'] == before_case
        paired.append((task, before, after, baseline_result, after_case))
    OUT.mkdir()
    owner = dict(pid=os.getpid(), ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19])
    captured = OUT / 'captured-journal-prefix.jsonl'
    with captured.open('xb') as stream:
        stream.write(prefix)
    files[str(captured)] = sha(captured)
    write(OUT / 'plan.json', dict(owner=owner, at=datetime.now(timezone.utc).isoformat(), source_files_sha256=files,
        original_assessment_owner=producer['owner'], original_journal=str(journal), captured_prefix_bytes=len(prefix),
        cohort='first five Python completion tasks in the original prescribed order, fixed before scoring',
        original_full_assessment_programs=96, this_reader_trained_programs=COUNT,
        complete_baseline_executions_reused_with_exact_source_code_and_action_hashes=True,
        fresh_ordinary_Python_reference_controls_required=True, generated_programs_never_repaired=True,
        no_new_generation_training_or_held_answer_optimization=True, full_matched_assessment_not_replaced=True, goal_achieved=False))
    try:
        results = []
        for task, before, after, baseline_result, case in paired:
            payload = request(task, 'Python', case['code'])
            reference = execute(task[4], task[0], payload['cases'])
            assert reference['functional_passed'] and reference['all_cases_observed']
            candidate = execute(case['code'], task[0], payload['cases'])
            assert candidate['total_cases'] == baseline_result['ordinary_Python']['total_cases']
            result = dict(task=task[0], delivered_trained=case, actual_trained_actions=after,
                ordinary_Python=candidate, fresh_reference=reference,
                trained_code_sha256=hashlib.sha256(case['code'].encode()).hexdigest(),
                initial_code_sha256=baseline_result['code_sha256'],
                earlier_initial_ordinary_Python=baseline_result['ordinary_Python'],
                earlier_initial_result_sha256=files[str(BASELINE / (task[0] + '.json'))],
                native_response_changed=before['actual_response_ids'] != after['actual_response_ids'])
            write(OUT / (task[0] + '.json'), result)
            results.append(result)
            print(json.dumps(dict(task=task[0], initial_syntax=baseline_result['ordinary_Python']['syntax_valid'],
                trained_syntax=candidate['syntax_valid'], initial_functional=baseline_result['ordinary_Python']['functional_passed'],
                trained_functional=candidate['functional_passed'], status=candidate['status'])), flush=True)
        with journal.open('rb') as stream:
            current = os.fstat(stream.fileno())
            assert (current.st_dev, current.st_ino) == (original.st_dev, original.st_ino)
            assert stream.read(len(prefix)) == prefix
        assert all(sha(path) == digest for path, digest in files.items())
        summary = dict(programs_per_endpoint=COUNT,
            initial_syntax=sum(r['earlier_initial_ordinary_Python']['syntax_valid'] for r in results),
            trained_syntax=sum(r['ordinary_Python']['syntax_valid'] for r in results),
            initial_functional=sum(r['earlier_initial_ordinary_Python']['functional_passed'] for r in results),
            trained_functional=sum(r['ordinary_Python']['functional_passed'] for r in results),
            changed_native_responses=sum(r['native_response_changed'] for r in results),
            fresh_reference_controls=COUNT, finite_cases_per_endpoint=sum(r['ordinary_Python']['total_cases'] for r in results),
            trained_EOS_stops=sum(r['actual_trained_actions']['stopped_on_EOS'] for r in results),
            trained_budget_ends=sum(r['actual_trained_actions']['ended_on_budget'] for r in results))
        write(OUT / 'complete.json', dict(complete=True, owner=owner, source_files_sha256=files, summary=summary,
            result_files={str(OUT / (r['task'] + '.json')): sha(OUT / (r['task'] + '.json')) for r in results},
            first_five_only_not_complete_trained_endpoint_or_full96comparison=True,
            baseline_executions_not_repeated_here=True, ordinary_Python_only_not_native_efficiency_measurement=True,
            no_model_weights_loaded_or_neural_computation=True, original_source_and_current_training_not_changed=True,
            original_growing_journal_not_bound_as_immutable=True,
            repeated_local_development_cohort_not_independent_final_benchmark=True,
            no_automatic_model_adoption=True, all_ten_final_requirements_incomplete=True, goal_achieved=False))
        print(json.dumps(dict(complete=True, summary=summary)), flush=True)
    except BaseException as error:
        write(OUT / 'failure.json', dict(owner=owner, error=repr(error), traceback=traceback.format_exc(),
            original_workers_and_inputs_not_changed=True, partial_outputs_retained=True, goal_achieved=False))
        raise


if __name__ == '__main__':
    main()
