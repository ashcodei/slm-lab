"""Assess every actual Python delivery under ordinary confined module behavior.

Keeps all original finite cases, program bytes and generated512-token budgets.
Reports this stdlib/normal-main-guard protocol alongside the old restricted
subset. It performs no neural generation or training, and no efficiency claim.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import time
import traceback

from slm_lab.code_quality import instruction_code, execute as restricted_execute
from slm_lab.code_tasks import TASKS, request
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.local_source_closure_v2 import python_sources
from slm_lab.ordinary_python_function_cases_v1 import execute, implementation_sources

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'reports/own-execution-contrast-full-CPU-v1'
OUT = ROOT / 'reports/own-execution-contrast-ordinary-Python-v1'
ENDPOINTS = ('initial', 'anchor_CE', 'anchor_contrast')
FRAMES = ('completion', 'instruction')


def live(identity):
    try: fields = (Path('/proc') / str(identity['pid']) / 'stat').read_text().split(') ', 1)[1].split()
    except FileNotFoundError: return False
    return fields[19] == identity['ticks'] and fields[0] not in ['Z', 'X']


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or OUT.exists(): raise ValueError('Fresh CPU-hidden ordinary-Python assessment required')
    OUT.mkdir(); expires = time.time() + 24 * 3600
    owner = dict(pid=os.getpid(), ticks=(Path('/proc') / str(os.getpid()) / 'stat').read_text().split(') ', 1)[1].split()[19])
    plan_path = RUN / 'plan.json'; producer_plan = json.loads(plan_path.read_text()); producer = producer_plan['owner']
    files = implementation_sources(); files.update({str(p): file_hash(p) for p in [plan_path,
        ROOT / 'tests/test_ordinary_python_function_cases_v1.py', ROOT / 'reports/ordinary-python-function-cases-v1-tests.log',
        *python_sources(ROOT, [Path(__file__), ROOT / 'tests/test_ordinary_python_function_cases_v1.py'])]})
    guard = FileBindings(files)
    atomic_json(OUT / 'plan.json', dict(at=datetime.now(timezone.utc).isoformat(), owner=owner, exact_producer=producer,
        endpoints=list(ENDPOINTS), original_frames=list(FRAMES), all12_original_Python_functions_each_frame=True,
        source_files_sha256=files, expires_at_unix=expires, original_cases_and_programs_and512_generation_budgets_preserved=True,
        execution_protocol='OS-confined ordinary Python stdlib module; import-style __name__; no injected globals or source edits',
        expected_answers_and_references_not_sent_to_candidate_worker=True,
        protocol_change_not_a_new_model_training_or_efficiency_result=True, neural_generations=0, model_updates=0, goal_achieved=False))
    artifacts = [OUT / 'plan.json']; summaries = {}; outcomes = {}
    tasks = {task[0]: task for task in TASKS}
    try:
        controls = []
        for task in TASKS:
            payload = request(task, 'Python', task[4], True)
            guarded = task[4] + '\n\nif __name__ == "__main__":\n    raise RuntimeError("main guard must stay inactive for module testing")\n'
            result = execute(guarded, task[0], payload['cases'])
            if not result['functional_passed'] or not result['all_cases_observed']: raise ValueError('Every canonical normal-Python main-guard control must pass the original complete finite cases')
            controls.append(dict(task=task[0], actual_reference_with_inert_main_guard=guarded, actual_original_control=result))
        old_control = restricted_execute(request(TASKS[0], 'Python', TASKS[0][4] + '\n\nif __name__ == "__main__":\n    pass\n', True))
        if old_control.get('functional_passed') is not False or old_control.get('syntax_valid') is not True:
            raise ValueError('The observed restricted-subset main-guard limitation must reproduce')
        control_path = OUT / 'actual-canonical-module-and-subset-controls.json'
        atomic_json(control_path, dict(all12_ordinary_Python_main_guard_reference_controls=controls,
            original_restricted_subset_main_guard_observation=old_control, correct_reference_program_rejected_only_by_restricted_protocol=True,
            no_model_generated_program_correctness_inferred_from_control=True)); artifacts.append(control_path)
        for endpoint in ENDPOINTS:
            outcomes[endpoint] = {}
            for frame in FRAMES:
                path = RUN / endpoint / ('code-' + frame + '.json')
                while not path.is_file():
                    if not live(producer) or (RUN / 'failure.json').exists(): raise ValueError('Exact producer ended before the next complete original code frame')
                    if time.time() > expires: raise ValueError('Declared ordinary-Python assessment lifetime exceeded; no automatic retry')
                    guard.check(full=False); time.sleep(20)
                report = json.loads(path.read_text()); checkpoint = ROOT / 'checkpoints/own-execution-contrast-full-CPU-v1' / (endpoint + '-actor.pt')
                if (report['task_count'] != 24 or len(report['results']) != 24 or report['generation_budget'] != 512
                        or report['prompt_mode'] != frame or report['checkpoint_sha256'] != file_hash(checkpoint)
                        or report['suite_sha256'] != file_hash(ROOT / 'src/slm_lab/code_tasks.py')
                        or report['inference_device_type'] != 'cpu' or report['inference_precision'] != 'fp32'
                        or [(r['language'], r['task']) for r in report['results']] != [(lang, t[0]) for lang in ['Python', 'JavaScript'] for t in TASKS]):
                    raise ValueError('Every actual original native code frame/checkpoint/function/budget required')
                selected = [r for r in report['results'] if r['language'] == 'Python']
                if len(selected) != 12: raise ValueError('Every original Python delivery must remain selected without eligibility pruning')
                def assess(row):
                    task = tasks[row['task']]; name, args, spec = task[:3]
                    code = instruction_code(row['raw_response']) if frame == 'instruction' else f'def {name}({args}):\n    """{spec}"""\n' + row['raw_response'].split('```')[0]
                    code = re.split(r'\n(?=File:)', code)[0]
                    if row['code'] != code or row['specification'] != spec: raise ValueError('Actually delivered program bytes must reconstruct unchanged')
                    payload = request(task, 'Python', code)
                    actual = execute(code, name, payload['cases'])
                    return dict(endpoint=endpoint, frame=frame, language='Python', task=name, actual_generated_response=row['raw_response'],
                        actual_generated_program=code, original_restricted_subset_execution=row['candidate'],
                        ordinary_Python_execution=actual, original_all_finite_cases_preserved=True)
                with ThreadPoolExecutor(max_workers=2) as pool: rows = list(pool.map(assess, selected))
                key = endpoint + '-' + frame; output = OUT / (key + '-all12-actual-programs.jsonl')
                output.write_text(''.join(json.dumps(r, ensure_ascii=False, separators=(',', ':')) + '\n' for r in rows))
                result = dict(Python_functions=12, syntax_valid=sum(r['ordinary_Python_execution']['syntax_valid'] for r in rows),
                    ordinary_Python_correct=sum(r['ordinary_Python_execution']['functional_passed'] for r in rows),
                    original_restricted_subset_correct=sum(bool(r['original_restricted_subset_execution'].get('functional_passed')) for r in rows),
                    runtime_protocol_only_gained=[r['task'] for r in rows if r['ordinary_Python_execution']['functional_passed'] and not r['original_restricted_subset_execution'].get('functional_passed')],
                    runtime_protocol_only_lost=[r['task'] for r in rows if not r['ordinary_Python_execution']['functional_passed'] and r['original_restricted_subset_execution'].get('functional_passed')])
                summaries[key] = result; outcomes[endpoint][frame] = rows
                receipt = OUT / (key + '-complete.json')
                atomic_json(receipt, dict(at=datetime.now(timezone.utc).isoformat(), endpoint=endpoint, frame=frame,
                    actual_outcomes=result, source_files_sha256={str(p): file_hash(p) for p in [path, checkpoint, output]},
                    no_original_program_or_case_excluded=True, new_neural_generation_or_training_updates=0,
                    not_a_new_model_or_efficiency_or_independent_final_benchmark_claim=True, goal_achieved=False))
                artifacts.append(receipt); print(json.dumps(dict(endpoint=endpoint, frame=frame, actual_outcomes=result)), flush=True)
        comparisons = {}
        for before, after in [('initial', 'anchor_CE'), ('initial', 'anchor_contrast'), ('anchor_CE', 'anchor_contrast')]:
            comparisons[before + '__to__' + after] = {}
            for frame in FRAMES:
                first, second = outcomes[before][frame], outcomes[after][frame]
                if [r['task'] for r in first] != [r['task'] for r in second]: raise ValueError('Every matched native function required')
                comparisons[before + '__to__' + after][frame] = dict(functions=12,
                    before_correct=sum(r['ordinary_Python_execution']['functional_passed'] for r in first),
                    after_correct=sum(r['ordinary_Python_execution']['functional_passed'] for r in second),
                    gained=[a['task'] for a, b in zip(first, second) if not a['ordinary_Python_execution']['functional_passed'] and b['ordinary_Python_execution']['functional_passed']],
                    lost=[a['task'] for a, b in zip(first, second) if a['ordinary_Python_execution']['functional_passed'] and not b['ordinary_Python_execution']['functional_passed']])
        guard.check(full=True)
        atomic_json(OUT / 'complete.json', dict(at=datetime.now(timezone.utc).isoformat(), complete=True,
            all72_actual_Python_programs_and12_original_reference_main_guard_controls_executed=True,
            actual_outcomes=summaries, matched_same_ordinary_Python_protocol_comparisons=comparisons,
            source_files_sha256={str(p): file_hash(p) for p in artifacts},
            original_program_bytes_cases_and_native_generation_budgets_preserved=True,
            no_new_neural_generation_training_or_empirical_efficiency_claim=True, candidate_promoted=False, goal_achieved=False))
    except BaseException as error:
        atomic_json(OUT / 'failure.json', dict(owner=owner, error=repr(error), traceback=traceback.format_exc(), automatic_retry=False, goal_achieved=False)); raise


if __name__ == '__main__': main()
