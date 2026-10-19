"""Freeze the full native coding recipe after exact sources and prior stages.

This owned CPU waiter cannot import torch, construct a model, acquire a GPU
lease or launch GPU workers. A later fresh normal nice0 process must prove
actual storage, exclusive ownership and full-size native capacity separately.
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
from slm_lab.metadata_observer_recovery_v1 import ORIGINAL_OUT
OUT = ORIGINAL_OUT
PARENTS = {
    'full_native_capacity_sources': ('native-human-code-suffix-actual-capacity-CPU-v2', None),
    'original_reasoning_source_recipe': ('batched-reasoning-aligned-preparation-v2', dict(pid=177810, ticks='56522852')),
}
CHECKS = ROOT / 'reports/native-prefill-and-owned-driver-v1-review'


def read(path): return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def state(path):
    s = Path(path).stat()
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


WRITE_GUARD = None


def atomic(path, value):
    from slm_lab.reserved_atomic_metadata_v1 import write_json
    def resources():
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.sched_getscheduler(0) != os.SCHED_IDLE or 'torch' in sys.modules:
            raise ValueError('The exact idle non-neural metadata owner must remain')
        if WRITE_GUARD is not None: WRITE_GUARD()
    return write_json(path,value,root=OUT,minimum_free_bytes=40*1024**3,
        maximum_metadata_bytes=64*1024**2,guard=resources)


def main(capacity_owner):
    global WRITE_GUARD
    PARENTS['full_native_capacity_sources'] = ('native-human-code-suffix-actual-capacity-CPU-v2', capacity_owner)
    if (OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19
            or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'])):
        raise ValueError('A fresh CUDA-hidden nice19 one-thread source-only preparer is required')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    from slm_lab.exact_waiting_process_retirement_v1 import terminal
    from slm_lab.joint_visual_runtime_contract_v1 import ready
    from slm_lab.local_source_closure_v2 import python_sources
    from slm_lab.native_suffix_preparation_v2 import completed_stages, assemble, merge, FULL
    import train_native_code_suffix_aligned_GPU_v2 as driver
    if 'torch' in sys.modules: raise ValueError('The entire source-only recipe must avoid the neural runtime')
    for name, (folder, expected) in PARENTS.items():
        source_folder = ROOT / 'reports' / folder
        if read(source_folder / 'plan.json')['owner'] != expected:
            raise ValueError('The exact original source owner must remain unchanged: ' + name)
        if terminal(expected) and ((source_folder / 'failure.json').exists() or not (source_folder / 'complete.json').is_file()
                or read(source_folder / 'complete.json').get('complete') is not True):
            raise ValueError('A terminal original source must retain successful complete evidence: ' + name)
    failed_path = ROOT / 'reports/native-code-suffix-aligned-preparation-CPU-v1-terminal-metadata-reserve-failure-v1.json'
    failed = read(failed_path)
    metadata_tests = ROOT / 'reports/reserved-atomic-metadata-v1-tests.log'
    if (failed['owner'] != dict(pid=212200,ticks='57912275') or not terminal(failed['owner'])
            or failed['original_preparer_failed_and_not_relabelled'] is not True
            or sha(failed['log']) != failed['log_sha256'] or '4 passed' not in metadata_tests.read_text()):
        raise ValueError('Preserve the exact failed first preparer and require all unchanged-reserve waiting checks')
    checks, review = read(CHECKS / 'complete.json'), read(CHECKS / 'review.json')
    tests = ROOT / 'reports/native-suffix-preparation-v2-tests.log'
    if (not checks.get('complete') or checks.get('review_sha256') != sha(CHECKS / 'review.json')
            or checks.get('actual_prefill_checks_passed') != 3 or checks.get('actual_owned_driver_checks_passed') != 10
            or checks.get('actual_small_physical_AdamW_calls') != 6 or '9 passed' not in tests.read_text()):
        raise ValueError('All actual source/prefill/connected-driver and preparation checks must pass')
    entrypoints = [Path(__file__), Path(driver.__file__), ROOT / 'src/slm_lab/native_suffix_preparation_v2.py',
        ROOT / 'tests/test_native_suffix_preparation_v2.py']
    files = {}
    merge(files, review['source_files_sha256'])
    merge(files, review['actual_closed_proofs_sha256'])
    paths = [CHECKS / 'complete.json', CHECKS / 'review.json', tests, failed_path, Path(failed['log']), metadata_tests,
        ROOT / 'reports/native-code-suffix-aligned-preparation-CPU-v1/plan.json',
        *[ROOT / 'reports' / folder / 'plan.json' for folder, _ in PARENTS.values()], *python_sources(ROOT, entrypoints)]
    merge(files, {str(p): sha(p) for p in paths})
    if any(sha(p) != h for p, h in files.items()): raise ValueError('Every actual tested immutable source and closed proof must agree')
    states = {p: state(p) for p in files}
    owner = dict(pid=os.getpid(), ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19])
    def guard():
        if ('torch' in sys.modules or os.sched_getscheduler(0) != os.SCHED_IDLE
                or any(state(p) != expected for p, expected in states.items())):
            raise ValueError('All tested source bytes and the exact idle non-neural context must remain')
    WRITE_GUARD = guard
    OUT.mkdir()
    atomic(OUT / 'plan.json', dict(owner=owner, at=datetime.now(timezone.utc).isoformat(), source_files_sha256=files,
        exact_original_source_parents={k: v[1] for k, v in PARENTS.items()}, full_native_budget=FULL,
        previous_failed_metadata_preparer_preserved=failed, unchanged40GiB_floor_waited_for_before_metadata_writes=True,
        initializer='fixed completed own reasoning anchor_RL native; chosen before future native-code quality',
        cached_baseline='same fixed reasoning anchor_RL full native assessments',
        original_canonical_visual_mixed_reasoning_scopes_preserved=True,
        all_original_QA16961_AI2D3088x3_MBPP600_human4642_words3429x3_code_broad_and289reasoning_retained=True,
        native_stage_options=driver.STAGE_OPTIONS, native_prefill=driver.NATIVE_PREFILL,
        models_constructed=0, neural_forwards=0, GPU_contexts=0, GPU_leases=0, production_updates=0,
        GPU_workers_never_launched_by_this_preparation=True, automatic_retry=False, goal_achieved=False))
    captured = {}
    try:
        deadline = time.monotonic() + 35 * 86400
        while any(not terminal(expected) for _, expected in PARENTS.values()):
            guard()
            if time.monotonic() > deadline: raise ValueError('The declared source-only preparation lifetime expired')
            atomic(OUT / 'progress.json', dict(owner=owner, phase='waiting for exact full source recipes',
                original_source_owners_live={k: not terminal(v[1]) for k, v in PARENTS.items()},
                models_constructed=0, GPU_work_started=False, full_runtime_recipe_prepared=False))
            time.sleep(30)
        for folder, _ in PARENTS.values():
            path = ROOT / 'reports' / folder
            if (path / 'failure.json').exists() or not read(path / 'complete.json')['complete']:
                raise ValueError('An exact original complete source recipe failed or stopped without successful evidence')
            merge(files, {str(p): sha(p) for p in [path / 'complete.json', path / 'plan.json']})
        original_folder = ROOT / 'reports/batched-reasoning-aligned-preparation-v2'
        original = read(original_folder / 'ready.json')
        merge(files, {str(original_folder / 'ready.json'): sha(original_folder / 'ready.json')})
        if not ready(ROOT, **original['exact_predecessors']):
            raise ValueError('The actual complete canonical pipeline and every original source scope must remain')
        capacity_folder = ROOT / 'reports/native-human-code-suffix-actual-capacity-CPU-v2'
        capacity = read(capacity_folder / 'complete.json')
        while True:
            guard()
            stages = completed_stages(ROOT, captured)
            if stages is not None: break
            if time.monotonic() > deadline: raise ValueError('The declared prior-stage observation lifetime expired')
            atomic(OUT / 'progress.json', dict(owner=owner, phase='waiting for all actual full visual/mixed/reasoning experiments',
                observed_actual_stage_plans=captured, missing_future_plans_are_not_live_jobs=True,
                models_constructed=0, GPU_work_started=False, full_runtime_recipe_prepared=False))
            time.sleep(30)
        prepared = assemble(ROOT, original, capacity, stages, owner=owner, source_files=files, driver=driver)
        guard()
        observed = shutil.disk_usage(ROOT).free
        prepared.update(at=datetime.now(timezone.utc).isoformat(), observed_free_bytes=observed,
            storage_funded_at_preparation=observed >= prepared['required_free_bytes_before_launch'],
            actual_storage_shortfall_bytes=max(0, prepared['required_free_bytes_before_launch'] - observed))
        atomic(OUT / 'ready.json', prepared)
        atomic(OUT / 'complete.json', dict(complete=True, ready_sha256=sha(OUT / 'ready.json'),
            only_complete_source_recipe_prepared=True, all_prior_full_populations_and_fixed_baseline_preserved=True,
            actual_storage_funding_and_owned_GPUcapacity_training_export_still_required=True,
            models_constructed=0, neural_forwards=0, GPU_contexts=0, GPU_leases=0, production_updates=0, goal_achieved=False))
    except BaseException as exc:
        atomic(OUT / 'failure.json', dict(owner=owner, error=repr(exc), traceback=traceback.format_exc(),
            automatic_retry=False, original_workers_and_all_partial_outputs_preserved=True,
            GPU_work_started=False, goal_achieved=False))
        raise


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capacity-pid',type=int,required=True)
    parser.add_argument('--capacity-ticks',required=True)
    args = parser.parse_args()
    main(dict(pid=args.capacity_pid,ticks=args.capacity_ticks))
