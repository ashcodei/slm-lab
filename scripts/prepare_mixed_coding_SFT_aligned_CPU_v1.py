"""Idle source-only recipe after the exact full canonical/source preparation.

This process never imports neural runtime modules, obtains a GPUlease or
launches a worker. A later normal nice0 process must execute the owned driver.
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
OUT = ROOT / 'reports/mixed-coding-SFT-aligned-preparation-v1'
BINDINGS = ROOT / 'reports/mixed-coding-SFT-source-bindings-CPU-v1'
VISUAL = ROOT / 'reports/joint-visual-aligned-continuation-preparation-v1'
PARENT = dict(pid=136092, ticks='54738417')


def read(path): return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic(path, value):
    raw = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()
    if len(raw) > 64 * 1024**2: raise ValueError('Whole immutable recipe must fit without truncation')
    pending = path.with_suffix(path.suffix + '.pending')
    with pending.open('xb') as stream: stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    pending.replace(path)


def immutable(path):
    name = Path(path).name
    return not ('queue' in name or 'progress' in name or name in ['children.json', 'LATEST.json'] or name.endswith('.pending'))


def main():
    if (OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19
            or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'])):
        raise ValueError('Only a fresh CUDA-hidden nice19 one-thread idle source preparer is admitted')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    from slm_lab.exact_waiting_process_retirement_v1 import terminal
    from slm_lab.joint_visual_runtime_contract_v1 import ready
    from slm_lab.local_source_closure_v2 import python_sources
    import train_mixed_coding_SFT_aligned_GPU_v1 as driver
    if read(VISUAL / 'plan.json')['owner'] != PARENT: raise ValueError('The exact original source-only preparation birth is required')
    if 'torch' in sys.modules: raise ValueError('The idle source recipe cannot import the neural runtime')
    tests = ROOT / 'reports/mixed-coding-SFT-training-and-export-v1-tests.log'
    position_tests = ROOT / 'reports/mixed-coding-SFT-owned-driver-v1-tests.log'
    if '1 passed' not in tests.read_text() or '8 passed' not in position_tests.read_text():
        raise ValueError('All actual connected training/recovery/export and owned-driver checks must pass')
    if (not terminal(read(BINDINGS / 'plan.json')['owner']) or (BINDINGS / 'failure.json').exists()
            or not read(BINDINGS / 'complete.json')['complete']):
        raise ValueError('The exact full mixed source binding owner must finish successfully')
    binding = read(BINDINGS / 'complete.json')
    case = driver.CASES / 'actual-complete-source-capacity-plan.json'
    plan = read(case)
    if (plan['seed'] != driver.STAGE_OPTIONS['seed'] or plan['epochs'] != driver.STAGE_OPTIONS['epochs']
            or plan['complete_training_budget'] != binding['full_training_budget'] or plan['complete_mixed_source_identity_sha256'] != binding['mixed_source_identity_sha256']):
        raise ValueError('The full actual three-epoch original source/target/horizon plan must match')
    entrypoints = [Path(__file__), ROOT / 'scripts/train_mixed_coding_SFT_aligned_GPU_v1.py',
        ROOT / 'tests/test_mixed_coding_SFT_training_and_export_v1.py', ROOT / 'tests/test_mixed_coding_SFT_owned_driver_v1.py']
    files = {str(p): sha(p) for p in [VISUAL / 'plan.json', tests, position_tests, case, BINDINGS / 'plan.json', BINDINGS / 'complete.json',
        driver.ADMITTED, *python_sources(ROOT, entrypoints)]}
    states = {p: (Path(p).stat().st_ino, Path(p).stat().st_size, Path(p).stat().st_mtime_ns) for p in files}
    owner = dict(pid=os.getpid(), ticks=Path('/proc/self/stat').read_text().split(') ', 1)[1].split()[19])
    OUT.mkdir()
    atomic(OUT / 'plan.json', dict(owner=owner, exact_original_source_preparer=PARENT,
        at=datetime.now(timezone.utc).isoformat(), source_files_sha256=files, profiles=list(driver.PROFILES),
        full_training_budget=plan['complete_training_budget'], complete_original_train_sources=28028,
        original_source_families=['public_weak_synthetic_code', 'original_human_code', 'public_procedural_reasoning'],
        actual_capacity_updates_per_profile=53, actual_physical_capacity_updates_per_profile=54,
        fresh_four_replica_capacity_readback_required=True, all_original_canonical_scopes_preserved=True,
        full_candidate_suites=list(driver.SUITES), all289reserved_cases_initial_and_both_candidates=True,
        GPU_workers_never_launched_by_this_preparation=True, models_constructed=0, neural_forwards=0,
        GPU_contexts=0, GPU_leases=0, optimizer_updates=0, goal_achieved=False))
    def guard():
        if os.sched_getscheduler(0) != os.SCHED_IDLE or 'torch' in sys.modules or any(
                (Path(p).stat().st_ino, Path(p).stat().st_size, Path(p).stat().st_mtime_ns) != state for p, state in states.items()):
            raise ValueError('Every exact immutable source recipe and idle non-neural context must remain')
    try:
        deadline = time.monotonic() + 35 * 86400
        while not terminal(PARENT):
            guard()
            if time.monotonic() > deadline: raise ValueError('The source preparation lifetime expired')
            atomic(OUT / 'progress.json', dict(owner=owner, phase='waiting for exact completed canonical/source recipe',
                exact_original_preparer_live=True, models_constructed=0, GPU_work_started=False))
            time.sleep(30)
        if (VISUAL / 'failure.json').exists() or not (VISUAL / 'complete.json').exists():
            raise ValueError('The exact original source preparation did not complete; preserve its failure')
        prepared = read(VISUAL / 'ready.json')
        if not ready(ROOT, **prepared['exact_predecessors']): raise ValueError('All original canonical/source scopes must complete first')
        for path, digest in prepared['source_files_sha256'].items():
            if immutable(path): files[path] = digest
        files[str(VISUAL / 'complete.json')] = sha(VISUAL / 'complete.json')
        files[str(VISUAL / 'ready.json')] = sha(VISUAL / 'ready.json')
        files.update(read(driver.ADMITTED)['source_files_sha256'])
        files.update(binding['source_files_sha256'])
        if any(sha(p) != digest for p, digest in files.items()): raise ValueError('Every actual closed source and own initializer byte must match')
        # Four capacity slots and four training slots remain for both arms.
        # One pending state, two native exports, both full assessments and
        # both complete raw-bounded journals are additionally reserved.
        budget = driver.RESERVE + 9 * driver.MAX_STATE + 2 * prepared['maximum_native_bytes'] + 2 * driver.MAX_ASSESSMENTS + 2 * driver.MAX_JOURNAL + 64 * 1024**2
        atomic(OUT / 'ready.json', dict(complete=True, owner=owner, at=datetime.now(timezone.utc).isoformat(),
            source_files_sha256=files, exact_predecessors=prepared['exact_predecessors'],
            initializer=prepared['initializer'], cold_inputs=prepared['cold_inputs'],
            maximum_native_bytes=prepared['maximum_native_bytes'], cached_baseline=prepared['cached_baseline'],
            cached_own_reference=prepared['cached_own_reference'], pulse_sources=prepared['pulse_sources'],
            mixed_source_identity_sha256=binding['mixed_source_identity_sha256'],
            full_training_budget=plan['complete_training_budget'], required_free_bytes_before_launch=budget,
            observed_free_bytes=shutil.disk_usage(ROOT).free, later_fresh_normal_nice0_owned_launch_required=True,
            GPU_work_started=False, models_constructed=0, production_updates=0, goal_achieved=False))
        atomic(OUT / 'complete.json', dict(complete=True, ready_sha256=sha(OUT / 'ready.json'),
            only_source_recipe_prepared=True, all_actual_GPUcapacity_training_export_assessment_pending=True, goal_achieved=False))
    except BaseException as exc:
        atomic(OUT / 'failure.json', dict(owner=owner, error=repr(exc), traceback=traceback.format_exc(), automatic_retry=False,
            no_existing_worker_modified=True, GPU_work_started=False, goal_achieved=False))
        raise


if __name__ == '__main__': main()
