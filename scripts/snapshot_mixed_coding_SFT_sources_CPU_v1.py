"""Close actual admitted mixed-source bindings; imports no neural model instance."""
from datetime import datetime, timezone
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/mixed-coding-SFT-source-bindings-CPU-v1'
CASES = ROOT / 'reports/mixed-coding-SFT-capacity-and-SUM-v1-pytest/test_every_actual_native_sourc0/actual-complete-source-capacity-plan.json'


if __name__ == '__main__':
    if (OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19
            or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'])):
        raise ValueError('Fresh CUDA-hidden nice19 one-thread data-only snapshot required')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    from record_visual_stage_and_first_repair_arm_v1 import atomic, read, sha
    owner = dict(pid=os.getpid(), ticks=Path('/proc/self/stat').read_text().split(') ', 1)[1].split()[19])
    OUT.mkdir()
    atomic(OUT / 'plan.json', dict(owner=owner, data_only=True, exact_closed_capacity_plan=str(CASES),
        exact_closed_capacity_plan_sha256=sha(CASES), models_constructed=0, neural_forwards=0, GPUcontexts=0, goal_achieved=False))
    try:
        import torch
        from slm_lab.mixed_coding_SFT_data_v1 import MixedCodingSFTData
        from slm_lab.mixed_coding_SFT_capacity_cases_v1 import choose
        torch.set_num_threads(1)
        data = MixedCodingSFTData(ROOT)
        plan = read(CASES)
        if plan != choose(data, epochs=3, seed=989923): raise ValueError('All actual source identities and intact capacity batches must reconstruct')
        files = dict(data.source_files_sha256)
        for p in [Path(__file__), CASES, ROOT / 'src/slm_lab/mixed_coding_SFT_data_v1.py',
                ROOT / 'src/slm_lab/mixed_coding_SFT_capacity_cases_v1.py']:
            files[str(p)] = sha(p)
        data.guard.check(full=True)
        if torch.cuda.is_initialized(): raise ValueError('A source binding snapshot cannot acquire a CUDAcontext')
        atomic(OUT / 'complete.json', dict(complete=True, owner=owner, at=datetime.now(timezone.utc).isoformat(),
            source_files_sha256=files, mixed_source_identity_sha256=data.identity_sha256,
            full_training_budget=plan['complete_training_budget'], actual_complete_capacity_plan_sha256=sha(CASES),
            balanced_positive_family_weights=data.family_weights,
            nominal_whole_epoch_weighted_target_mass=[.5, .4, .1], actual_epoch_loss_fraction_not_guaranteed=True,
            complete_source_population=28028, public_synthetic_test_reward_oracle_forbidden=True,
            models_constructed=0, neural_forwards=0, optimizer_updates=0, GPUcontexts=0,
            all_actual_GPUcapacity_training_export_assessment_still_required=True, goal_achieved=False))
        print(str(OUT / 'complete.json'), flush=True)
    except BaseException as exc:
        atomic(OUT / 'failure.json', dict(owner=owner, error=repr(exc), automatic_retry=False, goal_achieved=False))
        raise
