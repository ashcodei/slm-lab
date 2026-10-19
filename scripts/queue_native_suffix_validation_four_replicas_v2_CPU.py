"""Wait for exact owned CPU trials, then run the full V2 four-replica trial once."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/native-suffix-validation-four-replicas-v2-queue'
PREDECESSORS = [dict(pid=178844, ticks='56558134'), dict(pid=180812, ticks='56636391')]


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic(path, value):
    pending = path.with_suffix(path.suffix + '.pending')
    with pending.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    pending.replace(path)


def identity(pid):
    fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=pid, ticks=fields[19], state=fields[0], nice=int(fields[16]), scheduler=os.sched_getscheduler(pid))


def main():
    if (OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19
            or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'])):
        raise ValueError('A fresh CUDA-hidden nice19 one-thread owned CPU queue is required')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    from slm_lab.exact_waiting_process_retirement_v1 import terminal
    from slm_lab.local_source_closure_v2 import python_sources
    seeds = [Path(__file__), ROOT / 'scripts/run_native_suffix_validation_four_replicas_v2_tests.py',
        ROOT / 'tests/test_native_suffix_validation_four_replicas_v2.py']
    sources = {str(p):sha(p) for p in python_sources(ROOT, seeds)}
    if 'torch' in sys.modules:
        raise ValueError('The waiting parent cannot import the neural runtime')
    OUT.mkdir()
    atomic(OUT / 'plan.json', dict(owner=identity(os.getpid()), at=datetime.now(timezone.utc).isoformat(),
        exact_predecessors=PREDECESSORS, source_files_sha256=sources, only_owned_small_CPU_test_will_be_launched=True,
        original_GPUworkers_never_modified=True, GPUleases=0, GPUcontexts=0, automatic_retry=False, goal_achieved=False))
    child = None
    try:
        deadline = time.monotonic() + 86400
        while True:
            status = [dict(p, terminal=terminal(p)) for p in PREDECESSORS]
            if all(p['terminal'] for p in status): break
            if time.monotonic() > deadline:
                raise ValueError('The owned CPU test queue lifetime expired; preserve all current workers')
            if any(sha(p) != h for p,h in sources.items()) or 'torch' in sys.modules:
                raise ValueError('The exact source-only queue and full declared trial bytes must remain')
            atomic(OUT / 'progress.json', dict(owner=identity(os.getpid()), phase='waiting for exact existing CPU trials',
                exact_predecessors=status, models_constructed=0, neural_forwards=0, GPU_work_started=False))
            time.sleep(15)
        first_log = ROOT / 'reports/parallel-native-suffix-stage-v1-tests.log'
        prior_log = ROOT / 'reports/native-suffix-validation-stage-v2-tests.log'
        if '12 passed' not in prior_log.read_text():
            raise ValueError('Every actual V2 source/byte-mutation/thread/update/recovery check must pass before this full trial')
        if any(sha(p) != h for p,h in sources.items()):
            raise ValueError('The full unchanged source trial must match before launch')
        log = ROOT / 'reports/native-suffix-validation-four-replicas-v2-tests.log'
        with log.open('x') as stream:
            child = subprocess.Popen([str(ROOT / '.venv/bin/python'), '-u', str(seeds[1])], cwd=ROOT,
                env={**os.environ, 'CUDA_VISIBLE_DEVICES':''}, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            atomic(OUT / 'child.json', dict(actual_owned_CPU_test=identity(child.pid),
                previous_V1_full_trial_passed='9 passed' in first_log.read_text(),
                previous_V1_trial_log_sha256=sha(first_log), V2_preceding_checks_log_sha256=sha(prior_log),
                no_predecessor_failure_removed_or_waived=True, no_old_worker_restarted=True, GPU_work_started=False))
            code = child.wait()
        if code != 0 or '1 passed' not in log.read_text():
            raise ValueError('The actual full four-replica V2 trial failed; preserve all observations and do not retry automatically')
        if any(sha(p) != h for p,h in sources.items()):
            raise ValueError('The full trial source changed during execution')
        atomic(OUT / 'complete.json', dict(complete=True, actual_test_exit_code=code, actual_test_log_sha256=sha(log),
            full_size_GPUcapacity_or_model_quality=False, original_workers_unchanged=True, goal_achieved=False))
    except BaseException as exc:
        atomic(OUT / 'failure.json', dict(error=repr(exc), traceback=traceback.format_exc(), automatic_retry=False,
            owned_child_pid=None if child is None else child.pid, existing_workers_unchanged=True, goal_achieved=False))
        raise


if __name__ == '__main__': main()
