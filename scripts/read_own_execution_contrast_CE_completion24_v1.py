"""Freshly execute the first complete trained24-case code frame and controls.

This is a complete local completion frame, not all48 cases or an independent
final coding benchmark. No new NN forward, training or response repair occurs.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re

from slm_lab.code_quality import execute, efficiency
from slm_lab.code_tasks import TASKS, request
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.local_source_closure_v2 import python_sources

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/own-execution-contrast-CE-completion24-v1-readback'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or OUT.exists(): raise ValueError('Fresh CPU-hidden completed-frame program readback required')
    path = ROOT / 'reports/own-execution-contrast-full-CPU-v1/anchor_CE/code-completion.json'
    report = json.loads(path.read_text()); checkpoint = ROOT / 'checkpoints/own-execution-contrast-full-CPU-v1/anchor_CE-actor.pt'
    old_path = ROOT / 'reports/own-execution-contrast-full-CPU-v2-outcome-readback/initial/complete.json'
    before = json.loads(old_path.read_text())['independent_outcomes']['native_code']['completion']
    if (report['task_count'] != 24 or len(report['results']) != 24 or report['generation_budget'] != 512
            or report['prompt_mode'] != 'completion' or report['checkpoint_sha256'] != file_hash(checkpoint)
            or report['inference_device_type'] != 'cpu' or report['inference_precision'] != 'fp32'
            or report['suite_sha256'] != file_hash(ROOT / 'src/slm_lab/code_tasks.py')
            or [(r['language'], r['task']) for r in report['results']] != [(lang, t[0]) for lang in ['Python', 'JavaScript'] for t in TASKS]):
        raise ValueError('Every actual original trained native completion case and unchanged512-budget protocol required')
    files = {str(ROOT / p): sha for p, sha in report['evaluation_provenance']['files_sha256'].items()}
    files.update({str(p): file_hash(p) for p in [path, checkpoint, old_path, *python_sources(ROOT, [Path(__file__)])]})
    guard = FileBindings(files); tasks = {t[0]: t for t in TASKS}
    def replay(row):
        task = tasks[row['task']]; name, args, specification, scaling, py, js = task
        prefix = (f'def {name}({args}):\n    """{specification}"""\n' if row['language'] == 'Python'
            else f'function {name}({args}) {{\n  // {specification}\n')
        code = prefix + row['raw_response'].split('```')[0]
        if row['language'] == 'Python': code = re.split(r'\n(?=File:)', code)[0]
        if code != row['code'] or row['specification'] != specification or row['intended_scaling'] != scaling:
            raise ValueError('Actual raw generated text must reconstruct the original executed program without repair')
        candidate = execute(request(task, row['language'], code))
        for key in ['syntax_valid', 'functional_passed', 'passed_cases', 'total_cases']:
            if candidate.get(key) != row['candidate'].get(key): raise ValueError('Fresh candidate syntax/finite-case credit differs: ' + str((row['language'], name, key)))
        reference = execute(request(task, row['language'], py if row['language'] == 'Python' else js, True))
        if not reference.get('functional_passed') or not reference.get('efficiency_completed'):
            raise ValueError('Every original canonical benchmark control must freshly pass')
        passed, measurement = efficiency(candidate, reference, scaling)
        return dict(language=row['language'], task=name, actual_raw_response=row['raw_response'], actual_code=code,
            fresh_actual_execution=candidate, fresh_original_reference_control=reference,
            empirical_efficiency_pass=passed, empirical_efficiency_measurement=measurement,
            empirical_timing_verdict_agrees_with_producer=passed == row['efficiency_stress_passed'])
    with ThreadPoolExecutor(max_workers=2) as pool: rows = list(pool.map(replay, report['results']))
    syntax = sum(bool(r['fresh_actual_execution'].get('syntax_valid')) for r in rows)
    correct = sum(bool(r['fresh_actual_execution'].get('functional_passed')) for r in rows)
    efficient = sum(r['empirical_efficiency_pass'] for r in rows)
    if syntax / 24 != report['syntax_pass_rate'] or correct / 24 != report['functional_pass_at_1']:
        raise ValueError('Every original24-case trained aggregate must independently rederive')
    guard.check(full=True); OUT.mkdir(); output = OUT / 'all24-fresh-candidate-and-reference-executions.jsonl'
    output.write_text(''.join(json.dumps(r, ensure_ascii=False, separators=(',', ':')) + '\n' for r in rows)); files[str(output)] = file_hash(output)
    atomic_json(OUT / 'complete.json', dict(at=datetime.now(timezone.utc).isoformat(), complete=True,
        actual_trained_completion_deliveries_freshly_executed=24, actual_reference_controls_freshly_executed=24,
        before_independently_executed_initial=before,
        after_independently_executed_CE40=dict(tasks=24, syntax_passes=syntax, correct=correct, empirical_efficiency_stress_passes=efficient),
        empirical_timing_verdict_disagreements=sum(not r['empirical_timing_verdict_agrees_with_producer'] for r in rows),
        every_original_program_and_case_and_native_budget_preserved=True,
        raw_code_text_not_actual_BPE_action_IDs_retained_by_producer=True,
        full48case_CE_and_contrastive_assessments_still_pending=True,
        local_development_not_independent_final_benchmark_or_any_code_efficiency_proof=True,
        source_files_sha256=files, model_forwards=0, model_updates=0, candidate_promoted=False, goal_achieved=False))
    print(json.dumps(dict(complete=str(OUT / 'complete.json'), initial_syntax=before['syntax_passes'], CE40_syntax=syntax,
        initial_correct=before['correct'], CE40_correct=correct, empirical_efficiency_passes=efficient)), flush=True)


if __name__ == '__main__': main()
