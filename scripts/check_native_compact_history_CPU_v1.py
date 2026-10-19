"""Owned, CUDA-hidden small-neural qualification; never launches GPU training."""
import hashlib
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'reports/native-compact-neural-v3-proof-v1'


def write(name, value):
    (REPORT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    os.chdir(ROOT)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
        os.environ[key] = '1'
    os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
    os.setpriority(os.PRIO_PROCESS, 0, 19)
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    if 22 not in os.sched_getaffinity(0):
        raise ValueError('The existing admitted idle CPU 22 must remain available')
    os.sched_setaffinity(0, {22})
    REPORT.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__).resolve(),
        ROOT / 'src/slm_lab/native_reference_records_v1.py',
        ROOT / 'src/slm_lab/native_suffix_compact_stage_v3.py',
        ROOT / 'tests/test_native_suffix_compact_stage_v3.py',
        ROOT / 'tests/test_native_suffix_capacity_controller_v2.py',
        ROOT / 'tests/test_parallel_native_suffix_stage_v1.py',
        ROOT / 'reports/full-native-token-suffix-qualification-gzip-recovery-CPU-v3/closed-source-bindings.json']
    bindings = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    owner = dict(pid=os.getpid(), ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19])
    plan = dict(at=datetime.now(timezone.utc).isoformat(), owner=owner, source_files_sha256=bindings,
        actual_small_CPU_replicas=4, profiles=['anchor_CE', 'anchor_execution'],
        planned_physical_small_AdamW_calls=56, repeated_human_programs=1,
        fixture_source_presentations_per_arm=9, fixture_action_horizon=5, original_action_horizon=512,
        actual_closed_source_map_used_for_history_geometry_only=True,
        no_full_source_data_admission_or_GPU_capacity_claim=True, production_updates=0,
        priority=dict(nice=os.getpriority(os.PRIO_PROCESS, 0), scheduler=os.sched_getscheduler(0), affinity=[22]),
        goal_achieved=False)
    write('plan.json', plan)
    print('Starting source-bound four-replica compact-history neural proof', owner, flush=True)
    import pytest
    code = pytest.main(['-q', '-s', 'tests/test_native_suffix_compact_stage_v3.py',
        '--basetemp=' + str(REPORT / 'pytest')])
    unchanged = all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in bindings.items())
    outcome = dict(owner=owner, finished_at=datetime.now(timezone.utc).isoformat(), pytest_exit_code=int(code),
        all_original_source_bytes_unchanged=unchanged, production_updates=0, new_quality_gain=False,
        goal_achieved=False)
    write('complete.json' if code == 0 and unchanged else 'failure.json', outcome)
    if not unchanged:
        raise ValueError('The owned proof sources changed during its actual execution')
    return int(code)


if __name__ == '__main__':
    sys.exit(main())
