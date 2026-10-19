"""Source-only successor for the corrected, still unlaunched reasoning driver.

The original preparer and all canonical workers remain untouched. This waits
for their exact successful completion, then retains their full source recipe.
It never imports torch, constructs a model, takes a GPU lease, or launches one.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/batched-reasoning-aligned-preparation-v2'
PREVIOUS = ROOT / 'reports/batched-reasoning-aligned-preparation-v1'
PARENT = dict(pid=165313, ticks='56047070')
CHECKS = ROOT / 'reports/batched-reasoning-owned-driver-v2/complete.json'
DIAGNOSTIC = ROOT / 'reports/batched-reasoning-aligned-GPU-v1-known-optimizer-block-v1.json'


def read(path): return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic(path, value):
    raw = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()
    if len(raw) > 64 * 1024**2:
        raise ValueError('The complete immutable source recipe must fit without truncation')
    pending = path.with_suffix(path.suffix + '.pending')
    with pending.open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    pending.replace(path)


def immutable(path):
    name = Path(path).name
    return not ('queue' in name or 'progress' in name or name in ['children.json', 'LATEST.json'] or name.endswith('.pending'))


def main():
    if (OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19
            or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'])):
        raise ValueError('Only a fresh CUDA-hidden nice19 one-thread source preparer is admitted')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    from slm_lab.exact_waiting_process_retirement_v1 import terminal
    from slm_lab.joint_visual_runtime_contract_v1 import ready
    from slm_lab.local_source_closure_v2 import python_sources
    import train_batched_reasoning_aligned_GPU_v2 as driver
    if read(PREVIOUS / 'plan.json')['owner'] != PARENT:
        raise ValueError('The exact original reasoning source preparer birth is required')
    if 'torch' in sys.modules:
        raise ValueError('Source preparation cannot import the neural runtime')
    checks = read(CHECKS)
    diagnostic = read(DIAGNOSTIC)
    if (not checks['complete'] or checks['pytest_exit_code'] != 0 or checks['actual_small_CPU_optimizer_calls'] != 6
            or diagnostic['launch_permitted'] is not False or diagnostic['actual_driver_optimizer'] != 'torch.optim.Adam'):
        raise ValueError('The preserved Adam mismatch and actual tested AdamW recovery are required')
    case = driver.CASES / 'actual-complete-source-capacity-plan.json'
    capacity = read(case)
    full = dict(updates=1926, source_presentations=7704, complete_context_tokens=735570,
        response_EOS_targets=57240, full_native_generation_horizon_sum=726336)
    if (capacity['complete_training_budget'] != full or capacity['epochs'] != driver.STAGE_OPTIONS['epochs']
            or capacity['seed'] != driver.STAGE_OPTIONS['seed'] or len(capacity['capacity_entries']) != 109
            or driver.STAGE_OPTIONS['CPU_mechanics_horizon'] is not None or driver.STAGE_OPTIONS['group_size'] != 8):
        raise ValueError('All original three-epoch sources, full native horizons and actual capacity cases are required')
    entrypoints = [Path(__file__), ROOT / 'scripts/train_batched_reasoning_aligned_GPU_v2.py',
        ROOT / 'scripts/run_batched_reasoning_owned_driver_v2_tests.py', ROOT / 'tests/test_batched_reasoning_owned_driver_v2.py']
    files = {str(p): sha(p) for p in [PREVIOUS / 'plan.json', CHECKS, DIAGNOSTIC, case, driver.ADMITTED,
        *python_sources(ROOT, entrypoints)]}
    files.update(checks['source_files_sha256'])
    files.update(checks['actual_proofs_sha256'])
    if any(sha(p) != h for p, h in files.items()):
        raise ValueError('The exact successfully tested driver/source bytes must remain unchanged')
    states = {p: (Path(p).stat().st_ino, Path(p).stat().st_size, Path(p).stat().st_mtime_ns) for p in files}
    owner = dict(pid=os.getpid(), ticks=Path('/proc/self/stat').read_text().split(') ', 1)[1].split()[19])
    OUT.mkdir()
    atomic(OUT / 'plan.json', dict(owner=owner, at=datetime.now(timezone.utc).isoformat(),
        exact_previous_source_preparer=PARENT, source_files_sha256=files, profiles=list(driver.PROFILES),
        optimizer=driver.OPTIMIZER, full_reasoning_budget=full, actual_capacity_updates_per_profile=109,
        actual_physical_capacity_updates_per_profile=110, original_bad_driver_preserved_and_not_launched=True,
        GPU_workers_never_launched_by_this_preparation=True, models_constructed=0, neural_forwards=0,
        GPU_contexts=0, GPU_leases=0, optimizer_updates=0, goal_achieved=False))
    def guard():
        if ('torch' in sys.modules or os.sched_getscheduler(0) != os.SCHED_IDLE
                or any((Path(p).stat().st_ino, Path(p).stat().st_size, Path(p).stat().st_mtime_ns) != state
                    for p, state in states.items())):
            raise ValueError('Exact tested immutable bytes and idle source-only context must remain')
    try:
        deadline = time.monotonic() + 35 * 86400
        while not terminal(PARENT):
            guard()
            if time.monotonic() > deadline:
                raise ValueError('The source preparation lifetime expired')
            atomic(OUT / 'progress.json', dict(owner=owner, phase='waiting for exact original reasoning source preparation',
                exact_previous_preparer_live=True, models_constructed=0, GPU_work_started=False))
            time.sleep(30)
        guard()
        if (PREVIOUS / 'failure.json').exists() or not read(PREVIOUS / 'complete.json')['complete']:
            raise ValueError('The previous full source preparation did not finish successfully; preserve its failure')
        prepared = read(PREVIOUS / 'ready.json')
        if not ready(ROOT, **prepared['exact_predecessors']) or prepared['full_reasoning_budget'] != full:
            raise ValueError('All exact original canonical/source scopes and the full reasoning budget must remain')
        for path, digest in prepared['source_files_sha256'].items():
            if immutable(path): files[path] = digest
        for path in [PREVIOUS / 'complete.json', PREVIOUS / 'ready.json']:
            files[str(path)] = sha(path)
        if any(sha(p) != h for p, h in files.items()):
            raise ValueError('Every retained source/initializer byte and tested optimizer recipe must match')
        budget = (driver.RESERVE + 9 * driver.MAX_STATE + 2 * prepared['maximum_native_bytes']
            + 2 * driver.MAX_ASSESSMENTS + 2 * driver.MAX_JOURNAL + 64 * 1024**2)
        result = dict(prepared, complete=True, owner=owner, at=datetime.now(timezone.utc).isoformat(),
            source_files_sha256=files, optimizer=driver.OPTIMIZER, required_free_bytes_before_launch=budget,
            observed_free_bytes=shutil.disk_usage(ROOT).free, later_fresh_normal_nice0_owned_launch_required=True,
            corrected_owned_driver=str(ROOT / 'scripts/train_batched_reasoning_aligned_GPU_v2.py'),
            original_bad_driver_preserved_and_not_launched=True, GPU_work_started=False,
            models_constructed=0, production_updates=0, goal_achieved=False)
        atomic(OUT / 'ready.json', result)
        atomic(OUT / 'complete.json', dict(complete=True, ready_sha256=sha(OUT / 'ready.json'),
            only_source_recipe_prepared=True, full_size_GPUcapacity_training_and_quality_still_pending=True, goal_achieved=False))
    except BaseException as exc:
        atomic(OUT / 'failure.json', dict(owner=owner, error=repr(exc), traceback=traceback.format_exc(),
            automatic_retry=False, existing_workers_unchanged=True, GPU_work_started=False, goal_achieved=False))
        raise


if __name__ == '__main__': main()
