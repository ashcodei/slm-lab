"""Record an actual partial-model capture and an active matched assessment."""
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.exact_waiting_process_retirement_v1 import terminal
from slm_lab.function_repair_GPU_contract_v1 import QUEUE, ADMISSION, handoff_admitted


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/public-pretraining-interim-assessment-v1-review'
ASSESSMENT = ROOT / 'reports/public-pretraining-interim-CPU-v2'


def read(path):
    return json.loads(Path(path).read_text())


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or OUT.exists():
        raise ValueError('Fresh CUDA-hidden review required')
    captured = read(ASSESSMENT / 'captured-source.json')
    evaluation = read(ASSESSMENT / 'evaluation-plan.json')
    prior_failure = ROOT / 'reports/public-pretraining-interim-CPU-v1/failure.json'
    main_plan = read(ROOT / 'reports/public-pretraining-recovery-v1/plan.json')
    cpu_plan = read(ROOT / 'reports/full-function-repair-comparison-CPU-v1/plan.json')
    queue = read(ROOT / QUEUE)
    if (not captured.get('complete') or terminal(evaluation['owner'])
            or terminal(main_plan['owner']) or terminal(cpu_plan['owner'])
            or not handoff_admitted(ROOT, queue['owner'])
            or (ASSESSMENT / 'failure.json').exists()):
        raise ValueError('Actual complete capture and original live neural owners required')
    step = captured['actual_checkpoint_update']
    parity = captured['full_model_values_after_cold_load']
    probes = captured['actual_native_cold_probes']
    if (not 0 < step < 4447 or not parity['every_registered_network_head_and_tied_table_value_bit_exact']
            or parity['actual_counts']['total_unique'] != 984399690
            or [v['kind'] for v in probes] != ['text', 'photo', 'word']
            or [v['actual_dtype'] for v in probes] != ['torch.bfloat16', 'torch.bfloat16', 'torch.float32']
            or not all(v['finite'] for v in probes)
            or evaluation['fixed_endpoint_order'] != ['initial', 'interim']):
        raise ValueError('Complete original model and actual native precision controls required')
    files = {str(ASSESSMENT / name): file_hash(ASSESSMENT / name) for name in
             ['capture-plan.json', 'captured-source.json', 'captured-complete-source-prefix.jsonl', 'evaluation-plan.json']}
    previous = Path(read(ROOT / 'reports/completion-audit.json')['latest_full_scope_review'])
    for path in [Path(__file__), previous, prior_failure,
                 ROOT / 'reports/public-pretraining-interim-export-v1-tests.log',
                 ROOT / 'reports/public-native-cold-inference-v1-tests.log',
                 ROOT / 'reports/function-repair-evaluation-CPU-v1-tests.log']:
        files[str(path)] = file_hash(path)
    guard = FileBindings(files)
    prefix = (ROOT / 'reports/public-pretraining-recovery-v1/training-metrics.jsonl').read_bytes()
    prefix = prefix[:prefix.rfind(b'\n') + 1]
    rows = list(map(json.loads, prefix.splitlines()))
    if [v['step'] for v in rows] != list(range(1, len(rows) + 1)):
        raise ValueError('Contiguous actual completed main prefix required')
    current = dict(owner=main_plan['owner'], complete_actual_updates=len(rows), planned_updates=4447,
                   response_EOS_targets=sum(int(v['response_EOS_targets']) for v in rows),
                   complete_prefix_sha256=hashlib.sha256(prefix).hexdigest())
    progress = read(ASSESSMENT / 'progress.json')
    cpu = read(ROOT / 'reports/full-function-repair-comparison-CPU-v1/progress.json')
    at = datetime.now(timezone.utc).isoformat()
    OUT.mkdir()
    proof = dict(complete=True, at=at, actual_partial_checkpoint_update=step,
                 actual_capture=str(ASSESSMENT / 'captured-source.json'), actual_evaluation_owner=evaluation['owner'],
                 actual_evaluation_progress=progress, original_main=current, original_CPU_comparison=cpu,
                 actual_GPU_rows=subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used',
                                                         '--format=csv,noheader,nounits'], text=True),
                 full_model_cold_value_parity=parity, actual_native_cold_probes=probes,
                 matched_complete_populations=read(ASSESSMENT / 'capture-plan.json')['full_fixed_populations'],
                 previous_precision_guard_failure_preserved=str(prior_failure),
                 completed_matched_quality_endpoints=[name for name in ['initial', 'interim']
                                                     if (ASSESSMENT / (name + '-complete.json')).exists()],
                 current_goal_turn_classification='progress: real partial public-model capture, full cold-value/native-forward controls and matched CPU assessment launched',
                 original_full_training_and_final_benchmark_scope_preserved=True,
                 actual_new_neural_optimizer_updates=0, new_generated_quality_gain_established=False,
                 candidate_promoted=False, source_files_sha256=files, goal_achieved=False)
    atomic_json(OUT / 'complete.json', proof)
    text = ('# Public pretraining intermediate assessment\n\n'
            f'An actual durable public-pretraining checkpoint at update **{step}/4,447** is independently packaged for inference. '
            'All **984,399,690** model values match their pinned source after a cold load. '
            'Actual 4,096-token text, 448 px photo and real word-image forwards are finite. '
            'Text/photo logits use CPU BF16; the original word readout intentionally retains FP32.\n\n'
            'The failed first precision guard is preserved. The corrected controller does not change the model computation. '
            'Four checkpoint-reader tests, two original CPU evaluator tests and one native precision regression pass.\n\n'
            'The active assessment compares the immutable initial model with this fixed intermediate model, in that order, '
            'using two low-priority CPU threads and the same BF16 context, original prompts, budgets and scorers. '
            'Both endpoints retain all 48 local coding tasks (512 generated tokens), 24 ordinary Python executions, '
            '56 knowledge questions, 16 maths questions, 20 OCR images, 15 object images and 32 photo/caption pairs. '
            'One complete inference model is resident at a time; no GPU lease or main-training signal is used.\n\n'
            f'The original GPU run is at {len(rows)}/4,447 updates in this review. The original three-arm CPU comparison, '
            'queued full-source GPU comparison and all ten downstream human-feedback/RL/export/evaluation controllers remain live. '
            'The fixed source-validation studies and all full final benchmarks remain unchanged.\n\n'
            '**This is an active evaluation, not a measured accuracy improvement.** Last completed broader coding correctness '
            'remains 0/48; the new matched scores will be reported only after their endpoints complete. '
            'All ten final requirements remain incomplete.\n')
    (OUT / 'metrics.md').write_text(text)
    review = copy.deepcopy(read(previous))
    review.update(at=at, previous_full_scope_review=str(previous),
                  current_goal_turn_classification=proof['current_goal_turn_classification'],
                  latest_actual_interim_assessment=proof,
                  source_files_sha256={**files, str(OUT / 'complete.json'): file_hash(OUT / 'complete.json')}, goal_achieved=False)
    evidence = (f' Actual durable partial public checkpoint {step}/4447 was captured without GPU use or producer restart. '
                'Every 984399690 model value matches its pinned source after cold load; real4096-token text/448px photo '
                'and word forwards pass their original BF16/BF16/FP32 outputs. A fixed initial-versus-interim full local '
                'CPU assessment is active. No generated accuracy gain, new optimizer update or final-model completion is established.')
    for key in ['serious_full_model_training', 'high_end_llm_capability', 'any_code_correctness_and_efficiency']:
        review['requirements'][key]['evidence'] += evidence
    if any(v['status'] != 'incomplete_for_requested_final_model' for v in review['requirements'].values()):
        raise ValueError('Intermediate inference controls do not complete any final requirement')
    atomic_json(OUT / 'review.json', review)
    guard.check(full=True)
    maps = [main_plan['source_files_sha256'], cpu_plan['source_files_sha256'], queue['source_files_sha256'],
            evaluation['source_files_sha256']]
    maps += [read(ROOT / item['queue'])['source_files_sha256'] for item in read(ROOT / ADMISSION)['new_controllers']]
    for name in ['reports/completion-audit.json', 'README.md', 'RESEARCH.md']:
        if any(str(ROOT / name) in mapping for mapping in maps):
            raise ValueError('Do not change live source-bound documents')
    audit = read(ROOT / 'reports/completion-audit.json')
    audit.update(at=at, latest_full_scope_review=str(OUT / 'review.json'), requirements=review['requirements'],
                 latest_actual_main_runtime=current, latest_actual_interim_assessment=str(OUT / 'complete.json'), goal_achieved=False)
    atomic_json(ROOT / 'reports/completion-audit.json', audit)
    paragraph = (f'Intermediate public-model assessment, **{at}**: an actual **step {step}/4,447** standalone passes '
                 'cold parity for all **984,399,690** values and real text/photo/word forwards. The corrected precision '
                 'check preserves the original FP32 word readout; the failed first attempt remains recorded. '
                 'A matched initial-versus-intermediate assessment is active on two low-priority CPU threads, retaining '
                 'all 48 code tasks, 24 additional Python executions and all broad local checks. '
                 f'The original four-L40 run is at **{len(rows)}/4,447**. **No new accuracy gain is established.** '
                 'All full human/RL/final-benchmark stages remain; all ten final requirements are incomplete. '
                 '[Assessment status](reports/public-pretraining-interim-assessment-v1-review/metrics.md), '
                 '[full review](reports/public-pretraining-interim-assessment-v1-review/review.json). Earlier entries are historical snapshots.\n\n')
    for name in ['README.md', 'RESEARCH.md']:
        path = ROOT / name
        first, rest = path.read_text().split('\n', 1)
        path.write_text(first + '\n\n' + paragraph + rest.lstrip('\n'))
    print(json.dumps(dict(review=str(OUT / 'review.json'), captured_update=step,
                          assessment_owner=evaluation['owner'], goal_achieved=False)), flush=True)


if __name__ == '__main__':
    main()
