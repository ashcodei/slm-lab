"""Normal nice0 CUDA-hidden observer handing off the four full owned stages."""
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import traceback

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/full-owned-experiment-launch-queue-v1'


def main():
    from slm_lab.full_experiment_launch_queue_v1 import STAGES, context, identity, read, sha, run
    from slm_lab.local_source_closure_v2 import python_sources
    from slm_lab.reserved_atomic_metadata_v1 import write_json
    context()
    if OUT.exists(): raise ValueError('Preserve the exact existing full experiment queue')
    tests = ROOT / 'reports/full-experiment-launch-queue-v1-tests.log'
    if 'passed' not in tests.read_text() or 'failed' in tests.read_text():
        raise ValueError('Actual non-neural handoff/source/funding/failure checks must pass')
    entrypoints = [Path(__file__).resolve(), ROOT / 'tests/test_full_experiment_launch_queue_v1.py',
        *[ROOT / 'scripts' / stage['driver'] for stage in STAGES]]
    files = {str(path): sha(path) for path in python_sources(ROOT, entrypoints)}
    files[str(tests)] = sha(tests)
    captures = {}
    for spec in STAGES:
        path = ROOT / 'reports' / spec['preparation'] / 'plan.json'
        plan = read(path); captures[spec['name']] = dict(owner=plan['owner'], plan_sha256=sha(path))
        files[str(path)] = sha(path)
    def guard():
        context()
        if any(sha(path) != digest for path, digest in files.items()):
            raise ValueError('Every actual full owned driver/source closure and captured preparer must remain unchanged')
    guard(); OUT.mkdir()
    def publish(name, value):
        write_json(OUT / name, value, root=ROOT, minimum_free_bytes=40 * 1024**3,
            maximum_metadata_bytes=64 * 1024**2, guard=guard)
    with (OUT / 'queue.lock').open('x') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        publish('plan.json', dict(owner=identity(os.getpid()), at=datetime.now(timezone.utc).isoformat(),
            source_files_sha256=files, exact_original_preparers=captures, declared_stages=list(STAGES),
            no_existing_worker_signals_or_scheduling_changes=True, no_GPU_lease_or_context_in_observer=True,
            all_future_GPU_work_requires_its_own_complete_owned_driver_admission=True,
            automatic_retry=False, goal_achieved=False))
        try:
            result = run(ROOT, OUT, captures, guard=guard, publish=publish)
            publish('complete.json', result)
        except BaseException as exc:
            publish('failure.json', dict(error=repr(exc), traceback=traceback.format_exc(), automatic_retry=False,
                all_existing_and_launched_owned_driver_outputs_preserved=True,
                original_and_owned_driver_watchdog_lifecycles_unchanged=True, goal_achieved=False))
            raise


if __name__ == '__main__': main()
