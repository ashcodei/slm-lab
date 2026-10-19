"""Freshly execute all96retained initial/final native programs without new NN.

Also runs all48Python deliveries under the separate ordinary stdlib-module
protocol. Preserves every original task/frame/program byte and the original
reported512token request. Original generation token ids were not retained by
this producer, so neither actual action counts nor EOS are certified here.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import traceback

from slm_lab.code_quality import instruction_code, request
from slm_lab.code_tasks import TASKS
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.exact_waiting_process_retirement_v1 import terminal
from slm_lab.local_source_closure_v2 import python_sources
from slm_lab.code_quality import execute as native_execute
from slm_lab.native_code_shards_v1 import reconcile
from slm_lab.ordinary_python_function_cases_v1 import execute as ordinary_execute, implementation_sources

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'reports/phase-joint-v3'
OUT = ROOT / 'reports/completed-main-native-code-CPU-v2'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or OUT.exists():
        raise ValueError('A fresh CPU-only exact completed-code reexecution is required')
    proof_path = ROOT / 'reports/completed-phase-joint-v3-CPU-readback-v1/complete.json'
    proof = json.loads(proof_path.read_text())
    if proof['actual_original_updates'] != 35284 or not proof['all_native_parameters_exactly_equal_private_full_model']:
        raise ValueError('A verified complete original main checkpoint is required')
    children = json.loads((SOURCE / 'children.json').read_text())
    workers = [p for p in children if re.fullmatch('assess-(initial|trained)-code-[0-3]', p['subject'])]
    if len(workers) != 8 or not all(terminal(p) for p in workers):
        raise ValueError('Every original initial/final code-generation worker must finish first')
    files = {str(p): file_hash(p) for p in [proof_path, SOURCE / 'plan.json',
             *python_sources(ROOT, [Path(__file__)])]}
    files.update(implementation_sources())
    for arm in ('initial', 'trained'):
        for frame in ('completion', 'instruction'):
            for rank in range(4):
                for suffix in ('.json', '.raw.json'):
                    p = SOURCE / arm / f'code-{frame}-shard-{rank}{suffix}'
                    files[str(p)] = file_hash(p)
    guard = FileBindings(files); OUT.mkdir()
    owner = dict(pid=os.getpid(), ticks=Path('/proc/self/stat').read_text().split(') ', 1)[1].split()[19])
    atomic_json(OUT / 'plan.json', dict(
        at=datetime.now(timezone.utc).isoformat(), owner=owner, source_files_sha256=files,
        original_main_updates=dict(initial=0, trained=35284), actual_native_programs=96,
        actual_secondary_ordinary_Python_programs=48, original_generation_budget_request=512,
        raw_generation_action_ids_and_EOS_not_retained_so_not_reconstructed=True,
        all_original_program_bytes_and_cases_preserved_no_repairs=True,
        secondary_stdlib_protocol_differs_from_original_math_only_prompt_contract=True,
        original_main_GPU_assessments_and_all_queues_unchanged=True,
        models_constructed=0, model_updates=0, goal_achieved=False,
    ))
    outcomes = {}; results = []; controls = []
    try:
        for task in TASKS:
            control = ordinary_execute(task[4] + '\nif __name__ == "__main__":\n    pass\n',
                                       task[0], request(task, 'Python', task[4])['cases'])
            if not control['functional_passed']:
                raise ValueError('Every original canonical ordinary-module reference must pass its full finite cases')
            controls.append(dict(task=task[0], actual_control=control))
            for language, reference_code in [('Python',task[4]),('JavaScript',task[5])]:
                checked = native_execute(request(task, language, reference_code))
                if not checked.get('functional_passed'):raise ValueError('Every canonical native Python/JavaScript control must actually pass before candidate scoring')
        atomic_json(OUT / 'all12-ordinary-module-reference-controls.json', controls)
        for arm in ('initial', 'trained'):
            folder = OUT / arm; folder.mkdir(); outcomes[arm] = {}
            checkpoint = ROOT / 'checkpoints/phase-joint-v3' / arm / 'model.pt'
            expected_sha = proof['source_files_sha256'][str(checkpoint)]
            for frame in ('completion', 'instruction'):
                report = reconcile([SOURCE / arm / f'code-{frame}-shard-{rank}.json' for rank in range(4)],
                                   folder / ('original-code-' + frame + '.json'))
                if report['checkpoint_sha256'] != expected_sha or report['generation_budget'] != 512:
                    raise ValueError('Every original delivery must use the exact verified full checkpoint and request budget')
                observed = []; ordinary = []; differences = []
                for position, case in enumerate(report['results']):
                    task = next(t for t in TASKS if t[0] == case['task']); name, args, specification = task[:3]
                    prefix = (f'def {name}({args}):\n    """{specification}"""\n' if case['language'] == 'Python'
                              else f'function {name}({args}) {{\n  // {specification}\n')
                    code = instruction_code(case['raw_response']) if frame == 'instruction' else prefix + case['raw_response'].split('```')[0]
                    if case['language'] == 'Python':
                        code = re.split(r'\n(?=File:)', code)[0]
                    if code != case['code']:
                        raise ValueError('Every actual retained program must reconstruct from its original raw response without repair')
                    actual = native_execute(request(task, case['language'], code))
                    if 'compiler_version' in actual:raise ValueError('A C++ compiler result cannot certify a Python/JavaScript execution')
                    if any(bool(actual.get(k)) != bool(case['candidate'].get(k)) for k in ['syntax_valid', 'functional_passed']):
                        differences.append(dict(position=position, task=name, language=case['language'],
                                                original=case['candidate'], fresh=actual))
                    item = dict(arm=arm, frame=frame, language=case['language'], task=name,
                        actual_code_sha256=hashlib.sha256(code.encode()).hexdigest(), actual_code=code,
                        actual_fresh_native_execution=actual, original_candidate=case['candidate'],
                        original_empirical_efficiency_pass=case['efficiency_stress_passed'])
                    if case['language'] == 'Python':
                        second = ordinary_execute(code, name, request(task, 'Python', code)['cases'])
                        item['actual_secondary_ordinary_Python_execution'] = second; ordinary.append(second)
                    observed.append(item); results.append(item)
                atomic_json(folder / (frame + '-all24-actual-program-executions.json'), observed)
                outcomes[arm][frame] = dict(tasks=len(observed),
                    syntax=sum(bool(r['actual_fresh_native_execution'].get('syntax_valid')) for r in observed),
                    functional=sum(bool(r['actual_fresh_native_execution'].get('functional_passed')) for r in observed),
                    original_empirical_efficiency=sum(bool(r['original_empirical_efficiency_pass']) for r in observed),
                    ordinary_Python_tasks=len(ordinary), ordinary_Python_functional=sum(r['functional_passed'] for r in ordinary),
                    actual_native_verdict_differences=differences)
        if len(results) != 96 or sum(len(r) for r in outcomes.values()) != 4:
            raise ValueError('Every unchanged original endpoint/frame/language/task must remain present')
        totals = {arm: {key: sum(r[key] for r in frames.values()) for key in [
            'tasks', 'syntax', 'functional', 'original_empirical_efficiency', 'ordinary_Python_tasks', 'ordinary_Python_functional']}
            for arm, frames in outcomes.items()}
        guard.check(full=True)
        artifacts = {str(p): file_hash(p) for p in OUT.rglob('*.json')}
        atomic_json(OUT / 'complete.json', dict(
            at=datetime.now(timezone.utc).isoformat(), complete=True, actual_endpoint_totals=totals, actual_frame_outcomes=outcomes,
            all96original_native_programs_freshly_reexecuted=True,
            all48original_Python_deliveries_also_executed_under_secondary_stdlib_protocol=True,
            all12original_ordinary_reference_controls_passed=True,
            real_generation_action_ids_and_EOS_not_retained_or_certified=True,
            original_generation_budget_request512_preserved=True, no_new_neural_generation_or_program_repair=True,
            no_general_correctness_asymptotic_efficiency_or_final_benchmark_independence_claim=True,
            source_files_sha256={**files, **artifacts}, original_main_and_all_queues_unchanged=True,
            models_constructed=0, model_updates=0, candidate_promoted=False, goal_achieved=False,
        ))
        print(json.dumps(dict(complete=str(OUT / 'complete.json'), actual_endpoint_totals=totals)), flush=True)
    except BaseException as error:
        atomic_json(OUT / 'failure.json', dict(owner=owner, error=repr(error), traceback=traceback.format_exc(),
            original_main_and_all_queues_untouched=True, models_constructed=0, model_updates=0, goal_achieved=False))
        raise


if __name__ == '__main__':
    main()
