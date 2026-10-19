"""Replace only the terminal metadata-failed serial observer in scratch.

The old observer and both source failures remain untouched. Actual source
preparation, ROOT-volume full experiment funding, all capacity/training and
every original assessment still gate each unchanged owned GPUdriver.
"""
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import traceback

ROOT = Path(__file__).resolve().parents[1]


def main():
    from slm_lab.full_experiment_launch_queue_v1 import STAGES as ORIGINAL_STAGES, context, identity, read, sha
    from slm_lab.full_experiment_launch_queue_v2 import STAGES, run
    from slm_lab.metadata_observer_recovery_v1 import QUEUE_OUT, INDEXED_OUT, capture
    from slm_lab.local_source_closure_v2 import python_sources
    from slm_lab.reserved_atomic_metadata_v1 import write_json
    context()
    if QUEUE_OUT.exists():
        raise ValueError('Preserve the actual existing replacement observer;there is no automatic retry')
    previous = ROOT / 'reports/full-owned-experiment-launch-queue-v1'
    recovery = capture(previous, previous.with_suffix('.log'), kind='serial')
    old = read(previous / 'plan.json')
    if old['declared_stages'] != list(ORIGINAL_STAGES):
        raise ValueError('Every original full declared experiment and scope must remain')
    original = read(INDEXED_OUT / 'plan.json')
    if original.get('versioned_metadata_recovery', {}).get('phase') != 'indexed':
        raise ValueError('The actual new full-scope indexed source-recovery owner is required')
    captures = dict(old['exact_original_preparers'])
    captures['native_code'] = dict(owner=original['owner'], plan_sha256=sha(INDEXED_OUT / 'plan.json'))
    files = dict(recovery['source_files_sha256'])
    paths = [Path(__file__), ROOT / 'scripts/train_native_code_suffix_aligned_GPU_v5.py',
        ROOT / 'scripts/recover_native_recipe_observers_CPU_v1.py']
    incoming = {str(p): sha(p) for p in python_sources(ROOT, paths)}
    incoming[str(INDEXED_OUT / 'plan.json')] = sha(INDEXED_OUT / 'plan.json')
    for path, digest in incoming.items():
        if path in files and files[path] != digest:
            raise ValueError('Every exact original and recovered source byte must agree')
        files[path] = digest
    def guard():
        context()
        if any(sha(p) != h for p, h in files.items()):
            raise ValueError('Every original and replacement source and captured plan must remain unchanged')
    guard(); QUEUE_OUT.mkdir()
    def publish(name, value):
        return write_json(QUEUE_OUT / name, value, root=QUEUE_OUT, minimum_free_bytes=40 * 1024**3,
            maximum_metadata_bytes=64 * 1024**2, guard=guard)
    with (QUEUE_OUT / 'queue.lock').open('x') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        publish('plan.json', dict(owner=identity(os.getpid()), at=datetime.now(timezone.utc).isoformat(),
            source_files_sha256=files, exact_original_preparers=captures, declared_stages=list(STAGES),
            preserved_original_metadata_failure=recovery, metadata_filesystem=str(QUEUE_OUT), original_experiment_root=str(ROOT),
            only_native_source_location_and_parent_selector_rebound=True, all_original_neural_scopes_and_GPUfunding_unchanged=True,
            no_existing_worker_signals_or_scheduling_changes=True, no_GPU_lease_or_context_in_observer=True, automatic_retry=False, goal_achieved=False))
        try:
            result = run(ROOT, QUEUE_OUT, captures, guard=guard, publish=publish)
            publish('complete.json', result)
        except BaseException as exc:
            publish('failure.json', dict(error=repr(exc), traceback=traceback.format_exc(), automatic_retry=False,
                all_original_and_launched_owned_outputs_preserved=True, goal_achieved=False))
            raise


if __name__ == '__main__':
    main()
