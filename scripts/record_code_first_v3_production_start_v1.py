"""Wait on the exact repaired run, then record actual first production work."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

import own_training_gpu_lease_v2 as owner
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.exact_waiting_process_retirement_v1 import terminal

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'reports/code-first-continuation-v3'
OUT = ROOT / 'reports/code-first-v3-production-start-v1.json'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or OUT.exists():
        raise ValueError('A fresh CPU-only actual production-start readback is required')
    plan_path = SOURCE/'plan.json'; plan = json.loads(plan_path.read_text()); owner.REPORTS = SOURCE
    identity = plan['owner']; owner.owned(plan)
    proofs = [json.loads((SOURCE/f'capacity-{r}.json').read_text()) for r in range(4)]
    if not json.loads((SOURCE/'capacity.json').read_text())['passed']:
        raise ValueError('Actual complete four-L40 capacity must pass first')
    for rank, proof in enumerate(proofs):
        if (proof['physical_GPU'] != rank or proof['actual_Adam_states'] != 445
                or proof['logical_updates'] != 8 or proof['physical_optimizer_calls'] != 10
                or not all(proof[k] for k in ['actual_next_source_updates_weights_Adam_all_worker_RNG_exact',
                    'invalid_source_and_private_CUDA_RNG_rejected_before_mutation', 'decoder_exact_during_word_update',
                    'complete_standalone_native_cold4096_and_word_outputs_and_every_value_exact'])):
            raise ValueError('Every actual new full-model source/recovery/cold proof must pass on all four GPUs')
    while True:
        owner.owned(plan)
        if terminal(identity) or (SOURCE/'failure.json').exists():
            raise ValueError('Exact production owner failed or ended; inspect it without restarting')
        progress_path = SOURCE/'progress.json'
        if progress_path.exists():
            progress = json.loads(progress_path.read_text())
            if progress['actual_updates'] >= 1:
                break
        time.sleep(2)
    ranks = [r for r in json.loads((SOURCE/'children.json').read_text()) if r['subject'].startswith('train-rank-')]
    if len(ranks) != 4 or any(terminal(r) for r in ranks):
        raise ValueError('All four exact production training ranks must remain live')
    baseline_path = SOURCE/'held-packed-initial.json'; baseline = json.loads(baseline_path.read_text())
    original_path = ROOT/'reports/phase-joint-v3/held-packed-35284.json'; original = json.loads(original_path.read_text())
    if baseline != original:
        raise ValueError('The new actual same128window initial retention result must exactly reproduce the complete original main')
    evidence = [Path(__file__), plan_path, SOURCE/'capacity.json', *[SOURCE/f'capacity-{r}.json' for r in range(4)],
                baseline_path, original_path, ROOT/'reports/terminal-waiter-pipeline-v1-handoff/complete.json']
    report = dict(at=datetime.now(timezone.utc).isoformat(), complete=True, actual_production_owner=identity,
        actual_four_live_training_ranks=ranks, all_four_fresh_capacity_recovery_and_cold_load_checks_passed=True,
        actual_production_updates=progress['actual_updates'], planned_production_updates=7558,
        actual_progress_snapshot=progress, initial_same_cohort_NLL=baseline['token_mean_NLL'],
        actual_initial_retention_report_byte_structure_equals_original_final=True,
        actual_GPU_observation=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used','--format=csv,noheader'],text=True),
        source_files_sha256={str(p):file_hash(p) for p in evidence},
        model_quality_gain_not_established=True, goal_achieved=False)
    atomic_json(OUT,report)
    master_path=ROOT/'reports/completion-audit.json';master=json.loads(master_path.read_text())
    if str(master_path) in plan['source_files_sha256']:
        raise ValueError('Production evidence recorded; a fixed-source audit must remain unchanged')
    master.update(updated_at=report['at'],latest_actual_code_first_production_start=str(OUT.relative_to(ROOT)),
        latest_code_first_full_size_capacity='reports/code-first-continuation-v3/capacity.json',
        current_goal_turn_classification='progress',all_ten_final_requirements_incomplete=True,goal_achieved=False)
    atomic_json(master_path,master)
    print(json.dumps(dict(report=str(OUT), actual_production_updates=progress['actual_updates'],
                          all_four_capacity_passed=True, initial_NLL=baseline['token_mean_NLL'])),flush=True)


if __name__=='__main__':main()
