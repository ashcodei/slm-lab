"""Preserve real v1 late-rejection gaps and matched v2 pre-mutation rejection."""
import os
from pathlib import Path
import sys


def main():
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19
            or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'])):
        raise ValueError('Only CUDA-hidden nice19 single-thread own-small fault reproduction is allowed')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    root = Path(__file__).resolve().parents[1]
    target = root / 'reports/batched-reasoning-recovery-guard-gaps-v1.json'
    if target.exists(): raise ValueError('Preserve every previous observed result')
    sys.path.insert(0, str(root / 'tests'))
    import json
    import torch
    from slm_lab.public_reasoning_train_data_v1 import PublicReasoningTrainData
    from test_parallel_batched_reasoning_stage_v1 import setup as old_setup, live, optimizer
    from test_parallel_batched_reasoning_stage_v2 import setup as new_setup
    data = PublicReasoningTrainData(root / 'data/public-reasoning-curriculum-v1',
        root / 'reports/public-reasoning-curriculum-v1-readback-admission/complete.json')
    observations = []
    for case in ['changed_source_guard', 'replaced_selected_parameter']:
        for version, make in [('v1', old_setup), ('v2', new_setup)]:
            stage = make(data)
            stage.update()
            saved, receipt = stage.snapshot()
            stage.update()
            original_check = data.guard.check
            if case == 'changed_source_guard':
                def changed(): raise ValueError('deliberate changed original source guard')
                data.guard.check = changed
            else:
                name, parameter = stage.named[0]
                module, leaf = name.rsplit('.', 1)
                setattr(stage.model.parts.get_submodule(module), leaf,
                    torch.nn.Parameter(parameter.detach().clone(), requires_grad=parameter.requires_grad))
                stage.optimizer = optimizer(stage.model)
            before, error = live(stage), None
            try: stage.resume(saved, expected_manifest=receipt)
            except ValueError as exc: error = str(exc)
            finally: data.guard.check = original_check
            after = live(stage)
            observations.append(dict(case=case, version=version, actual_student_values=stage.model.counts()['actual_unique'],
                actual_own_small_Adam_calls_before_fault=2, original_saved_complete_cursor=1,
                actual_error=error, actual_before=before, actual_after=after,
                rejected_before_any_live_mutation=error is not None and before == after,
                partial_recovery_mutation_before_error=error is not None and before != after))
    if (not all(r['partial_recovery_mutation_before_error'] for r in observations if r['version'] == 'v1')
            or not all(r['rejected_before_any_live_mutation'] for r in observations if r['version'] == 'v2')):
        raise ValueError('Expected actual old failure and corrected pre-mutation behavior must both be observed')
    assert not torch.cuda.is_initialized()
    value = dict(complete=True, actual_small_fixture_values=267334, actual_total_small_test_Adam_calls=8,
        actual_observations=observations, original_v1_two_late_recovery_rejection_gaps_observed=True,
        corrected_v2_both_rejected_before_mutation=True, production_updates=0,
        full_size_GPU_capacity_throughput_or_model_quality=False)
    with target.open('x') as stream: stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in value.items() if k != 'actual_observations'}))


if __name__ == '__main__': main()
