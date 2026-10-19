"""Observe scheduler transitions on this new tiny control process only."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/visual-launch-scheduler-capability-CPU-v1.json'


if __name__ == '__main__':
    if OUT.exists() or os.nice(0) != 19 or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Fresh CUDA-hidden nice19 own-only scheduler probe required')
    owner = dict(pid=os.getpid(), ticks=Path('/proc/self/stat').read_text().split(') ', 1)[1].split()[19])
    before = os.sched_getscheduler(0)
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    error = None
    try:
        os.sched_setscheduler(0, os.SCHED_OTHER, os.sched_param(0))
    except PermissionError as exc:
        error = dict(type=type(exc).__name__, errno=exc.errno, message=str(exc))
    observed = dict(complete=True, at=datetime.now(timezone.utc).isoformat(), actual_own_new_process=owner,
        before_policy=before, requested_policy=os.SCHED_IDLE, actual_final_policy=os.sched_getscheduler(0),
        own_nice=os.nice(0), RLIMIT_NICE=list(resource.getrlimit(resource.RLIMIT_NICE)),
        return_from_IDLE_to_OTHER_allowed=error is None, actual_own_transition_error=error,
        no_other_process_policy_priority_affinity_or_lifetime_changed=True,
        no_GPU_context_model_or_lease=True, goal_achieved=False)
    with OUT.open('x') as stream:
        stream.write(json.dumps(observed, indent=2) + '\n')
    print(json.dumps(observed))
