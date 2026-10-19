"""Freeze the complete actual-source RL capacity workload; construct no model."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil


def main():
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19
            or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'])):
        raise ValueError('Only CUDA-hidden nice19 single-thread source preparation is allowed')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    import torch
    from slm_lab.batched_reasoning_capacity_cases_v1 import choose
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    from slm_lab.public_reasoning_train_data_v1 import PublicReasoningTrainData
    root = Path(__file__).resolve().parents[1]
    out = root / 'reports/batched-reasoning-actual-capacity-cases-CPU-v1'
    if out.exists(): raise ValueError('Preserve every previous complete or failed capacity workload')
    tests = root / 'reports/batched-reasoning-capacity-controller-v2-tests.log'
    if '5 passed' not in tests.read_text(): raise ValueError('All connected source/runtime and peer rejection checks must pass')
    source_proofs = sorted((root / 'reports/batched-reasoning-capacity-controller-v2-pytest').rglob('actual-complete-worst-source-selection.json'))
    if len(source_proofs) != 2: raise ValueError('Both complete one-epoch and three-epoch native-source proofs are required')
    data = PublicReasoningTrainData(root / 'data/public-reasoning-curriculum-v1',
        root / 'reports/public-reasoning-curriculum-v1-readback-admission/complete.json')
    plan = choose(data, seed=989113, epochs=3)
    proof = next(json.loads(p.read_text()) for p in source_proofs
        if json.loads(p.read_text())['complete_training_budget']['source_presentations'] == 7704)
    if (proof['actual_capacity_budget'] != plan['capacity_budget'] or proof['actual_capacity_entries'] != len(plan['capacity_entries'])
            or not proof['independent_native_ids_labels_all_task_and_whole_batch_maxima_checked']
            or not proof['every_physical_replica_receives_every_selected_worst_source']):
        raise ValueError('Every actual native-source maximum and complete original source budget must reconcile')
    raw = (json.dumps(plan, indent=2, allow_nan=False) + '\n').encode()
    if len(raw) > 16 * 1024**2 or shutil.disk_usage(root).free - len(raw) < 40 * 1024**3:
        raise ValueError('Complete source metadata must fit above the original other-writer reserve')
    data.guard.check(full=True)
    if torch.cuda.is_initialized(): raise ValueError('Source preparation must create no CUDAcontext')
    bound = [Path(__file__), tests, *source_proofs, root / 'src/slm_lab/batched_reasoning_capacity_cases_v1.py',
        root / 'src/slm_lab/parallel_function_repair_stage_v1.py', root / 'tests/test_batched_reasoning_capacity_controller_v2.py']
    sources = dict(data.source_files_sha256)
    sources.update({str(p): file_hash(p) for p in bound})
    out.mkdir()
    path = out / 'actual-complete-source-capacity-plan.json'
    with path.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    sources[str(path)] = file_hash(path)
    atomic_json(out / 'complete.json', dict(complete=True, at=datetime.now(timezone.utc).isoformat(),
        source_files_sha256=sources, complete_training_budget=plan['complete_training_budget'],
        actual_task_families=plan['actual_train_task_families'],
        actual_intact_original_selected_batches=len(plan['selected_intact_original_batches']),
        actual_complete_capacity_entries=len(plan['capacity_entries']), actual_capacity_budget=plan['capacity_budget'],
        all_four_replicas_receive_every_selected_worst_source=True,
        full_original_generation_horizons=[64, 256], separate_zero_gradient_control_not_an_actual_training_tail=True,
        maximum_original_prompt_positions=plan['maximum_original_prompt_positions'],
        maximum_original_complete_context=plan['maximum_original_complete_context'],
        maximum_original_response_EOS_targets=plan['maximum_original_response_EOS_targets'],
        maximum_original_prompt_plus_full_horizon=plan['maximum_original_prompt_plus_full_horizon'],
        source_selection_only=True, models_constructed=0, neural_forwards=0, GPU_contexts=0, GPU_leases=0,
        optimizer_updates=0, full_size_GPUcapacity_passed=False, training_admitted=False, goal_achieved=False))
    print(json.dumps(dict(complete=str(out / 'complete.json'), actual_intact_batches=len(plan['selected_intact_original_batches']),
        actual_capacity_entries=len(plan['capacity_entries']), actual_capacity_budget=plan['capacity_budget'],
        full_original_training_budget=plan['complete_training_budget'], models_constructed=0, GPU_contexts=0)))


if __name__ == '__main__': main()
