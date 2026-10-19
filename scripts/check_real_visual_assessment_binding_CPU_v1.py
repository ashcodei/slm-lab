"""Check real frozen worker/reconciler identity without running either one."""
from datetime import datetime, timezone
import hashlib
import inspect
import json
import marshal
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/visual-real-assessment-binding-CPU-v1.json'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


if __name__ == '__main__':
    if (OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19
            or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'])):
        raise ValueError('Fresh CUDA-hidden nice19 one-thread source-only function-binding check required')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    import torch
    import assess_machine_standalone_full_v2_assessment_recovery_gate_v1 as evaluator
    from slm_lab.visual_native_assessment_binding_v1 import bind, SUITES
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    original = evaluator.REPORTS, evaluator.load_student, evaluator.legacy.registered
    worker, reconcile = evaluator.worker, evaluator.reconcile_suite
    codes = [hashlib.sha256(marshal.dumps(fn.__code__)).hexdigest() for fn in [worker, reconcile]]
    sources = [inspect.getsourcefile(fn) for fn in [worker, reconcile]]
    original_GPU_report_root = evaluator.owner.REPORTS
    calls = []
    def forbidden(*args, **kwargs):
        calls.append('unexpected')
        raise ValueError('This source-only check must not load a model or register/launch any worker')
    # This is dispatch metadata only, not a model or actual capacity receipt.
    plan = dict(owner=dict(pid=os.getpid()), expected_student_values=984399690, suites=list(SUITES))
    with bind(evaluator, reports=ROOT / 'reports/nonexistent-source-only-binding-target', plan=plan,
            load_student=forbidden, register=forbidden) as functions:
        if functions != (worker, reconcile) or evaluator.worker is not worker or evaluator.reconcile_suite is not reconcile:
            raise ValueError('The actual full numerical worker/reconciler function objects changed')
        if tuple(evaluator.SUITES) != SUITES:
            raise ValueError('Every original full native assessment suite is required')
    if (original != (evaluator.REPORTS, evaluator.load_student, evaluator.legacy.registered)
            or evaluator.owner.REPORTS != original_GPU_report_root or calls or torch.cuda.is_initialized()
            or codes != [hashlib.sha256(marshal.dumps(fn.__code__)).hexdigest() for fn in [worker, reconcile]]):
        raise ValueError('The actual frozen numerical code, all old bindings and zero-GPU source scope must be preserved')
    proof = dict(complete=True, passed=True, at=datetime.now(timezone.utc).isoformat(),
        source_files_sha256={str(p): sha(p) for p in [Path(__file__), ROOT / 'src/slm_lab/visual_native_assessment_binding_v1.py', *map(Path, sources)]},
        actual_real_numerical_worker_and_reconciler_function_objects_unchanged=True,
        actual_real_function_bytecode_sha256=codes, actual_original_suites=list(SUITES),
        actual_output_model_loader_and_registration_bindings_restored=True,
        actual_existing_GPU_report_root_unchanged=True,
        actual_new_models_constructed=0, actual_neural_forwards=0, actual_GPUcontexts=0,
        actual_GPUworkers_or_leases_started=0, real_full_size_GPUdispatch_and_quality_still_pending=True,
        expected_model_count_in_dispatch_metadata_is_not_a_loaded_model=True, goal_achieved=False)
    with OUT.open('x') as stream:
        stream.write(json.dumps(proof, indent=2) + '\n')
    print(json.dumps(dict(complete=True, real_functions_unchanged=True, models=0, GPU_work=0)))
