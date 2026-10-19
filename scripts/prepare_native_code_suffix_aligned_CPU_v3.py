"""Wait for complete original scopes and close the indexed GPU runtime recipe.

This idle source-only owner cannot import torch, acquire GPUs or launch any
worker. It preserves the earlier recipe and all failed/completed evidence.
The later normal-scheduled GPU owner must separately fund and measure every
full-size source/state/record/horizon bound before complete training.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/native-code-suffix-aligned-preparation-CPU-v3'
ORIGINAL = ROOT / 'reports/native-code-suffix-aligned-preparation-CPU-v2'
PROOF = ROOT / 'reports/native-indexed-training-v2-proof-v1'


def main():
    if (OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19
            or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'])):
        raise ValueError('A fresh CUDA-hidden nice19 one-thread source-only owner is required')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0)); os.sched_setaffinity(0, {22})
    from slm_lab.exact_waiting_process_retirement_v1 import terminal
    from slm_lab.local_source_closure_v2 import python_sources
    from slm_lab.native_suffix_preparation_v2 import read, sha, merge
    from slm_lab.native_suffix_preparation_v3 import assemble
    from slm_lab.reserved_atomic_metadata_v1 import write_json
    import train_native_code_suffix_aligned_GPU_v3 as driver
    if 'torch' in sys.modules: raise ValueError('The complete indexed recipe must remain non-neural')
    parents = {str(path): read(path / 'plan.json')['owner'] for path in [ORIGINAL, PROOF]}
    # An already terminal predecessor is allowed only with its successful
    # evidence. A live exact birth remains a wait, never an inferred failure.
    for path, owner in parents.items():
        if terminal(owner) and not (Path(path) / 'complete.json').exists():
            raise ValueError('An exact predecessor stopped without successful complete evidence')
    log = ROOT / 'reports/native-indexed-runtime-v3-tests.log'
    if 'passed' not in log.read_text() or 'failed' in log.read_text():
        raise ValueError('Full-scope source assembly, storage and no-neural-import checks must pass first')
    entrypoints = [Path(__file__).resolve(), Path(driver.__file__).resolve(),
        ROOT / 'src/slm_lab/native_suffix_preparation_v3.py', ROOT / 'tests/test_native_indexed_runtime_v3.py']
    runtime = {str(p): sha(p) for p in python_sources(ROOT, entrypoints)}
    files = dict(runtime)
    merge(files, {str(p): sha(p) for p in [log, ORIGINAL / 'plan.json', PROOF / 'plan.json']})
    for folder in [ORIGINAL, PROOF]: merge(files, read(folder / 'plan.json')['source_files_sha256'])
    owner = dict(pid=os.getpid(), ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19])
    def guard():
        if ('torch' in sys.modules or os.environ.get('CUDA_VISIBLE_DEVICES') != ''
                or os.sched_getscheduler(0) != os.SCHED_IDLE or os.nice(0) != 19
                or any(sha(p) != h for p, h in files.items())):
            raise ValueError('All original and added immutable bytes and the idle source-only context must remain')
    def publish(name, value):
        return write_json(OUT / name, value, root=ROOT, minimum_free_bytes=40 * 1024**3,
            maximum_metadata_bytes=64 * 1024**2, guard=guard)
    guard(); OUT.mkdir()
    publish('plan.json', dict(at=datetime.now(timezone.utc).isoformat(), owner=owner, exact_predecessors=parents,
        source_files_sha256=files, added_runtime_sources=runtime,
        full_original_source_scope_and_all_assessments_preserved=True,
        models_constructed=0, GPU_contexts=0, GPU_leases=0, production_updates=0,
        automatic_retry=False, goal_achieved=False))
    try:
        deadline = time.monotonic() + 35 * 86400
        while any(not terminal(identity) for identity in parents.values()):
            guard()
            for path, identity in parents.items():
                if terminal(identity) and ((Path(path) / 'failure.json').exists()
                        or not (Path(path) / 'complete.json').is_file()):
                    raise ValueError('An exact predecessor stopped without complete successful evidence: ' + path)
            if time.monotonic() > deadline: raise ValueError('The declared source-only observation lifetime expired')
            publish('progress.json', dict(owner=owner, phase='waiting for exact full original preparation and indexed proof',
                exact_owners_live={path: not terminal(identity) for path, identity in parents.items()},
                GPU_work_started=False, full_indexed_recipe_prepared=False))
            time.sleep(30)
        for folder in [ORIGINAL, PROOF]:
            if not (folder / 'complete.json').is_file() or (folder / 'failure.json').exists():
                raise ValueError('An exact original predecessor failed or lacks complete successful evidence')
            merge(files, {str(folder / 'complete.json'): sha(folder / 'complete.json')})
        merge(files, {str(ORIGINAL / 'ready.json'): sha(ORIGINAL / 'ready.json'),
            str(PROOF / 'pytest/test_actual_four_replica_compl0/complete.json'):
                sha(PROOF / 'pytest/test_actual_four_replica_compl0/complete.json')})
        prepared = assemble(ROOT, owner=owner, original_path=ORIGINAL / 'ready.json',
            source_files=files, runtime_sources=runtime, driver=driver)
        observed = shutil.disk_usage(ROOT).free
        prepared.update(at=datetime.now(timezone.utc).isoformat(), observed_free_bytes=observed,
            storage_funded_at_preparation=observed >= prepared['required_free_bytes_before_launch'],
            actual_storage_shortfall_bytes=max(0, prepared['required_free_bytes_before_launch'] - observed))
        publish('ready.json', prepared)
        publish('complete.json', dict(complete=True, ready_sha256=sha(OUT / 'ready.json'),
            source_recipe_only=True, all_original_complete_scopes_and_actual_CPU_proof_retained=True,
            actual_full_size_GPU_capacity_training_export_and_assessments_still_required=True,
            GPU_contexts=0, GPU_leases=0, production_updates=0, goal_achieved=False))
    except BaseException as exc:
        publish('failure.json', dict(owner=owner, error=repr(exc), traceback=traceback.format_exc(),
            automatic_retry=False, all_original_workers_and_partial_outputs_preserved=True,
            GPU_work_started=False, goal_achieved=False))
        raise


if __name__ == '__main__':
    main()
