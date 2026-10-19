"""Fresh normal source-only parent; prove its new idle workers cannot lower it."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import runpy


def main():
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.sched_getscheduler(0) != os.SCHED_OTHER or os.nice(0) != 0):
        raise ValueError('A fresh CUDA-hidden normal nice0source-only parent is required for this isolation proof')
    root = Path(__file__).resolve().parents[1]
    source = root / 'src/slm_lab/source_execution_threads_v1.py'
    scope = runpy.run_path(str(source))
    before = scope['observation']()
    with ThreadPoolExecutor(max_workers=2, initializer=scope['initialize']) as pool:
        workers = list(pool.map(lambda _: scope['observation'](), range(4)))
    after = scope['observation']()
    if before != after or not all(w['actual_nice'] == 19 and w['actual_scheduler'] == 5 and not w['thread_is_main'] for w in workers):
        raise ValueError('Actual worker native IDs must be idle/nice19and the parent must remain normal/nice0')
    value = dict(complete=True, actual_normal_main_before=before, actual_normal_main_after=after,
        actual_private_source_worker_observations=workers, actual_parent_priority_and_identity_unchanged=True,
        source_files_sha256={str(source):hashlib.sha256(source.read_bytes()).hexdigest(),str(Path(__file__)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        models_constructed=0, neural_forwards=0, GPUcontexts=0, existing_neural_workers_not_targeted=True,
        actual_GPU_speedup_not_established=True)
    path = root / 'reports/private-source-execution-thread-scheduler-v1.json'
    with path.open('x') as stream: stream.write(json.dumps(value, indent=2) + '\n')
    print(json.dumps(dict(complete=True,parent_normal_nice0_unchanged=True,new_workers_idle_nice19=True,models_constructed=0)))


if __name__ == '__main__': main()
