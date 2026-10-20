"""Freshly reexecute both closed 24-program FP32 completion frames.

Retains the original 512-action contract, captured prompts and unchanged code.
The original 96-program comparison continues; no training/model load occurs.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import traceback

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('/tmp/slm-broad-code-step512-code-evaluation-CPU-v1')
OUT = ROOT / 'reports/matched-FP32-completion-frame-readback-v1'
SNAPSHOT = Path('/tmp/slm-broad-code-step512-snapshot-CPU-v1')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    if (OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) != 19
            or any(os.environ.get(k) != '1' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'))):
        raise ValueError('Fresh CUDA-hidden nice19 single-thread closed-frame readback required')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    from tokenizers import Tokenizer
    from slm_lab.code_tasks import TASKS, request
    from slm_lab.code_quality import execute as native_execute
    from slm_lab.decoded_native_code_reexecution_v1 import reconstruct
    from slm_lab.ordinary_python_function_cases_v1 import execute as ordinary_execute, implementation_sources
    from slm_lab.local_source_closure_v2 import python_sources
    owner_plan = json.loads((SOURCE / 'owner-plan.json').read_text())
    assert owner_plan['interim_updates'] == 512 and owner_plan['total_programs'] == 96
    assert owner_plan['precision'] == 'FP32' and owner_plan['generation_budget'] == 512
    reports = {endpoint: json.loads((SOURCE / (endpoint+'-completion.json')).read_text()) for endpoint in ('initial','step512')}
    ordinary = {endpoint: json.loads((SOURCE / (endpoint+'-completion-ordinary-Python.json')).read_text()) for endpoint in reports}
    paths = [SOURCE / 'owner-plan.json', SOURCE / 'interim-source-state-readback.json', SNAPSHOT / 'tokenizer.json']
    paths += [SOURCE / (endpoint+'-completion'+suffix+'.json') for endpoint in reports for suffix in ('','-ordinary-Python')]
    paths += python_sources(ROOT, [Path(__file__).resolve()])
    files = {str(path): sha(path) for path in paths}
    files.update(implementation_sources())
    for path, digest in files.items():
        if path in owner_plan['source_files_sha256']:
            assert digest == owner_plan['source_files_sha256'][path]
    tokenizer = Tokenizer.from_file(str(SNAPSHOT / 'tokenizer.json'))
    vocab = tokenizer.get_vocab_size()
    journal = SOURCE / 'actual-native-actions.jsonl'
    with journal.open('rb') as stream:
        original_stat = os.fstat(stream.fileno())
        lines = [stream.readline() for _ in range(72)]
    assert all(line.endswith(b'\n') for line in lines)
    prefix = b''.join(lines)
    all_actions = [json.loads(line) for line in lines]
    actions = dict(initial=all_actions[:24], step512=all_actions[48:72])
    OUT.mkdir()

    def emit(name, value):
        with (OUT / name).open('x') as stream:
            stream.write(json.dumps(value, indent=2, allow_nan=False)+'\n')

    (OUT / 'captured-first72-actions.jsonl').write_bytes(prefix)
    files[str(OUT / 'captured-first72-actions.jsonl')] = sha(OUT / 'captured-first72-actions.jsonl')
    owner = dict(pid=os.getpid(), birth=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19])
    emit('owner-plan.json', dict(owner=owner, at=datetime.now(timezone.utc).isoformat(), source_files_sha256=files,
        original_evaluation_owner=owner_plan['owner'], captured_complete_journal_records=72,
        actual_matched_native_programs=48, actual_ordinary_Python_programs=24,
        complete_frame='completion', original_generation_budget=512, original_generation_precision='CPU FP32',
        full96program_comparison_not_replaced=True, original_workers_unchanged=True,
        no_new_generation_or_neural_execution=True, goal_achieved=False))
    try:
        summaries = {}
        results = []
        references = {}
        for task in TASKS:
            cases = request(task, 'Python', task[4])['cases']
            ref = ordinary_execute(task[4], task[0], cases)
            assert ref['functional_passed'] and ref['all_cases_observed']
            references[task[0]] = ref
        for endpoint, report in reports.items():
            assert report['task_count'] == len(report['results']) == 24
            assert report['generation_budget'] == 512 and report['decoding'] == 'greedy, cached FP32'
            assert report['prompt_mode'] == 'completion' and report['inference_precision'] == 'fp32'
            assert report['inference_device_type'] == 'cpu' and report['tokenizer_sha256'] == sha(SNAPSHOT / 'tokenizer.json')
            assert report['suite_sha256'] == reports['initial']['suite_sha256']
            ordinary_rows = {row['task']: row for row in ordinary[endpoint]['results']}
            current = []
            for i, (case, action) in enumerate(zip(report['results'], actions[endpoint])):
                task = TASKS[i % 12]
                language = ('Python','JavaScript')[i//12]
                assert (action['endpoint'], action['frame'], action['case'], action['language'], action['task']) == (
                    endpoint, 'completion', i+1, language, task[0])
                prompt = (f'def {task[0]}({task[1]}):\n    """{task[2]}"""\n' if language=='Python'
                    else f'function {task[0]}({task[1]}) {{\n  // {task[2]}\n')
                assert action['complete_prompt_ids'] == [1]+tokenizer.encode(prompt).ids
                ids = action['actual_response_ids']
                assert 0 < len(ids) <= 512 and all(type(value) is int and 0 <= value < vocab for value in ids)
                assert 2 not in ids[:-1] and action['stopped_on_EOS'] is (ids[-1]==2)
                assert action['ended_on_budget'] is (len(ids)==512 and ids[-1]!=2)
                assert tokenizer.decode(ids) == case['raw_response']
                actual_task, code = reconstruct(case, 'completion')
                assert actual_task == task and case['language'] == language
                actual = native_execute(request(task, language, code))
                assert actual['syntax_valid'] == case['candidate']['syntax_valid']
                assert actual['functional_passed'] == case['candidate']['functional_passed']
                py = None
                if language == 'Python':
                    py = ordinary_execute(code, task[0], request(task, language, code)['cases'])
                    saved = ordinary_rows[task[0]]
                    assert saved['actual_unchanged_code_sha256'] == hashlib.sha256(code.encode()).hexdigest()
                    assert all(py[key] == saved['execution'][key] for key in ('syntax_valid','functional_passed','original_cases_sha256','total_cases'))
                record = dict(endpoint=endpoint, language=language, task=task[0], actual_actions=action,
                    actual_unchanged_code=code, actual_code_sha256=hashlib.sha256(code.encode()).hexdigest(),
                    fresh_native=actual, fresh_ordinary_Python=py,
                    producer_efficiency_pass=case['efficiency_stress_passed'],
                    empirical_efficiency_not_independently_retimed=True)
                emit(endpoint+'-'+language+'-'+task[0]+'.json', record)
                current.append(record)
                results.append(record)
                print(json.dumps(dict(endpoint=endpoint,language=language,task=task[0],
                    native_syntax=actual['syntax_valid'],native_functional=actual['functional_passed'])),flush=True)
            summary = dict(programs=24, native_syntax=sum(row['fresh_native']['syntax_valid'] for row in current),
                native_functional=sum(row['fresh_native']['functional_passed'] for row in current),
                ordinary_Python_syntax=sum(row['fresh_ordinary_Python']['syntax_valid'] for row in current if row['fresh_ordinary_Python'] is not None),
                ordinary_Python_functional=sum(row['fresh_ordinary_Python']['functional_passed'] for row in current if row['fresh_ordinary_Python'] is not None),
                producer_efficiency=sum(row['producer_efficiency_pass'] for row in current),
                EOS=sum(action['stopped_on_EOS'] for action in actions[endpoint]),
                budget_ends=sum(action['ended_on_budget'] for action in actions[endpoint]))
            assert report['syntax_pass_rate'] == summary['native_syntax']/24
            assert report['functional_pass_at_1'] == summary['native_functional']/24
            assert report['efficiency_stress_pass_rate'] == summary['producer_efficiency']/24
            summaries[endpoint] = summary
        with journal.open('rb') as stream:
            stat = os.fstat(stream.fileno())
            assert (stat.st_dev, stat.st_ino) == (original_stat.st_dev, original_stat.st_ino)
            assert stream.read(len(prefix)) == prefix
        assert all(sha(path)==digest for path,digest in files.items())
        emit('reference-controls.json', references)
        result_files = {str(path):sha(path) for path in OUT.glob('*.json') if path.name != 'owner-plan.json'}
        emit('complete.json', dict(complete=True, owner=owner, at=datetime.now(timezone.utc).isoformat(),
            source_files_sha256=files, closed_result_files_sha256=result_files, summary=summaries,
            actual_native_program_reexecutions=48, actual_ordinary_Python_reexecutions=24,
            fresh_ordinary_Python_reference_controls=12,
            fresh_reference_finite_cases=sum(row['total_cases'] for row in references.values()),
            every_original_prompt_action_and_unchanged_delivery_rebuilt=True,
            every_native_and_ordinary_syntax_functional_verdict_agrees=True,
            complete_completion_frame_only_not_full96program_comparison=True,
            no_new_generation_model_weight_load_or_neural_execution=True,
            repeated_local_development_suite_not_independent_final_benchmark=True,
            native_reference_controls_and_efficiency_not_independently_reexecuted=True,
            original_workers_and_full_assessments_unchanged=True, candidate_promoted=False, goal_achieved=False))
        print(json.dumps(dict(complete=True,summary=summaries)),flush=True)
    except BaseException as error:
        emit('failure.json',dict(owner=owner,error=repr(error),traceback=traceback.format_exc(),
            partial_outputs_retained=True,original_workers_unchanged=True,goal_achieved=False))
        raise


if __name__ == '__main__':
    main()
