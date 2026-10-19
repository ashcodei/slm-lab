"""Execute all twelve delivered baseline Python completions without a model.

This reads only an already delivered prefix of the live assessment journal.
It neither changes that assessment nor replaces its full matched comparison.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import traceback

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('/tmp/slm-broad-code-step512-code-evaluation-CPU-v1')
SNAPSHOT = Path('/tmp/slm-broad-code-step512-snapshot-CPU-v1')
OUT = ROOT / 'reports/interim512-initial-Python-deliveries-CPU-v1'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def main():
    if (OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) != 19
            or any(os.environ.get(k) != '1' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'))):
        raise ValueError('Fresh CUDA-hidden nice19 one-thread non-neural reader required')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    from tokenizers import Tokenizer
    from slm_lab.code_tasks import TASKS, request
    from slm_lab.decoded_native_code_reexecution_v1 import reconstruct
    from slm_lab.local_source_closure_v2 import python_sources
    from slm_lab.ordinary_python_function_cases_v1 import execute, implementation_sources

    owner_path = SOURCE / 'owner-plan.json'
    producer = json.loads(owner_path.read_text())
    first_path = ROOT / 'reports/first-matched-CPU-delivery-and-placement-v1-review/complete.json'
    first = json.loads(first_path.read_text())
    tokenizer_path = SNAPSHOT / 'tokenizer.json'
    files = {str(p): sha(p) for p in [owner_path, first_path, SOURCE / 'interim-source-state-readback.json',
        tokenizer_path, *python_sources(ROOT, [Path(__file__)])]}
    files.update(implementation_sources())
    if (producer['interim_updates'] != 512 or producer['total_programs'] != 96
            or producer['precision'] != 'FP32' or producer['generation_budget'] != 512
            or len(TASKS) != 12 or files[str(tokenizer_path)] != producer['source_files_sha256'][str(tokenizer_path)]):
        raise ValueError('Exact original matched512/3721 assessment and tokenizer required')
    for path, digest in files.items():
        if path in producer['source_files_sha256'] and digest != producer['source_files_sha256'][path]:
            raise ValueError('Original runtime source changed before independent execution')
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    journal = SOURCE / 'actual-native-actions.jsonl'
    with journal.open('rb') as stream:
        original = os.fstat(stream.fileno())
        lines = [stream.readline() for _ in TASKS]
    if any(not line.endswith(b'\n') for line in lines):
        raise ValueError('All twelve whole original deliveries must already exist')
    prefix = b''.join(lines)
    actions = [json.loads(line) for line in lines]
    deliveries = []
    for index, (task, action) in enumerate(zip(TASKS, actions), 1):
        name, args, specification = task[:3]
        if (action['endpoint'], action['frame'], action['language'], action['case'], action['task']) != (
                'initial', 'completion', 'Python', index, name):
            raise ValueError('Every original baseline task must retain its complete prescribed order')
        prompt = f'def {name}({args}):\n    """{specification}"""\n'
        if action['complete_prompt_ids'] != [tokenizer.token_to_id('<bos>')] + tokenizer.encode(prompt).ids:
            raise ValueError('Every full original prompt must reconstruct exactly')
        ids = action['actual_response_ids']
        if (not 0 < len(ids) <= 512 or any(type(i) is not int or not 0 <= i < tokenizer.get_vocab_size() for i in ids)
                or 2 in ids[:-1] or action['stopped_on_EOS'] != (ids[-1] == 2)
                or action['ended_on_budget'] != (len(ids) == 512 and ids[-1] != 2)):
            raise ValueError('Every delivered action and original EOS/budget boundary must remain')
        raw = tokenizer.decode(ids)
        code = re.split(r'\n(?=File:)', prompt + raw.split('```')[0])[0]
        case = dict(language='Python', task=name, specification=specification, raw_response=raw, code=code)
        actual_task, actual_code = reconstruct(case, 'completion')
        assert actual_task == task and actual_code == code
        deliveries.append(case)
    OUT.mkdir()
    owner = dict(pid=os.getpid(), ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19])
    captured = OUT / 'captured-first-twelve-actions.jsonl'
    with captured.open('xb') as stream:
        stream.write(prefix)
        stream.flush()
        os.fsync(stream.fileno())
    files[str(captured)] = sha(captured)
    write(OUT / 'plan.json', dict(owner=owner, at=datetime.now(timezone.utc).isoformat(), source_files_sha256=files,
        original_assessment_owner=producer['owner'], original_journal=str(journal), captured_prefix_bytes=len(prefix),
        original_assessment_programs=96, this_reader_programs=12, endpoint='initial', frame='completion',
        complete_unchanged_deliveries_only=True, no_new_generation_or_training=True,
        held_references_not_used_for_optimization=True, full_matched_assessment_not_replaced=True, goal_achieved=False))
    try:
        results = []
        for task, action, case in zip(TASKS, actions, deliveries):
            payload = request(task, 'Python', case['code'])
            reference = execute(task[4], task[0], payload['cases'])
            if not reference['functional_passed'] or not reference['all_cases_observed']:
                raise ValueError('Every complete fresh ordinary-Python reference control must pass')
            candidate = execute(case['code'], task[0], payload['cases'])
            result = dict(task=task[0], delivered=case, actual_actions=action, ordinary_Python=candidate,
                fresh_reference=reference, code_sha256=hashlib.sha256(case['code'].encode()).hexdigest())
            if task[0] == TASKS[0][0] and (result['code_sha256'] != first['actual_code_sha256']
                    or action['actual_response_ids'] != first['actual_original_actions']
                    or any(candidate[k] != first['fresh_ordinary_Python'][k] for k in ('syntax_valid', 'functional_passed'))):
                raise ValueError('First unchanged delivery differs from the previously verified execution')
            write(OUT / (task[0] + '.json'), result)
            results.append(result)
            print(json.dumps(dict(task=task[0], syntax=candidate['syntax_valid'], functional=candidate['functional_passed'],
                status=candidate['status'], reference_passed=True)), flush=True)
        with journal.open('rb') as stream:
            current = os.fstat(stream.fileno())
            if ((current.st_dev, current.st_ino) != (original.st_dev, original.st_ino)
                    or stream.read(len(prefix)) != prefix):
                raise ValueError('The original delivered journal prefix changed during readback')
        if any(sha(path) != digest for path, digest in files.items()):
            raise ValueError('An actual bound source or captured delivery changed during execution')
        summary = dict(programs=12, syntax=sum(r['ordinary_Python']['syntax_valid'] for r in results),
            functional=sum(r['ordinary_Python']['functional_passed'] for r in results),
            all_finite_cases_observed=sum(r['ordinary_Python']['all_cases_observed'] for r in results),
            reference_controls_passed=12, EOS=sum(a['stopped_on_EOS'] for a in actions),
            budget_ends=sum(a['ended_on_budget'] for a in actions),
            original_finite_cases=sum(r['ordinary_Python']['total_cases'] for r in results))
        write(OUT / 'complete.json', dict(complete=True, owner=owner, at=datetime.now(timezone.utc).isoformat(),
            source_files_sha256=files, summary=summary, result_files={str(OUT / (r['task'] + '.json')):
                sha(OUT / (r['task'] + '.json')) for r in results},
            initial_Python_completion_only_not_a_trained_endpoint_or_full96program_comparison=True,
            efficiency_not_assessed=True, no_model_weights_loaded_or_neural_computation=True,
            growing_producer_journal_not_bound_as_immutable_or_changed=True,
            native_stopping_and_decoding_contract_not_changed=True,
            repeated_local_development_suite_not_an_independent_final_benchmark=True,
            no_quality_gain_or_checkpoint_promotion=True, goal_achieved=False))
        print(json.dumps(dict(complete=True, summary=summary)), flush=True)
    except BaseException as error:
        write(OUT / 'failure.json', dict(owner=owner, error=repr(error), traceback=traceback.format_exc(),
            original_workers_and_inputs_not_changed=True, partial_outputs_retained=True, goal_achieved=False))
        raise


if __name__ == '__main__':
    main()
