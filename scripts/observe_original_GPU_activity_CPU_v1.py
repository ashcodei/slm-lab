"""Bounded read-only NVML/process observation; no CUDA model/context or signal."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/original-human-GPU-activity-CPU-v1'
OWNERS = [(151931, '55253729'), (151932, '55253730'), (151933, '55253731'), (151934, '55253731')]


def process(pid, birth):
    f = Path('/proc', str(pid), 'stat').read_text().split(') ', 1)[1].split()
    if f[19] != birth or f[0] in ['X', 'Z']: raise ValueError('The exact canonical neural owner changed; inspect the handoff')
    return dict(pid=pid, ticks=birth, state=f[0], user_ticks=int(f[11]), system_ticks=int(f[12]),
        nice=os.getpriority(os.PRIO_PROCESS, pid), scheduler=os.sched_getscheduler(pid))


if __name__ == '__main__':
    if OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19:
        raise ValueError('Fresh CUDA-hidden low-priority read-only observer required')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    OUT.mkdir()
    rows = []
    for sample in range(16):
        rows.append(dict(at=datetime.now(timezone.utc).isoformat(), monotonic=time.monotonic(),
            actual_original_owners=[process(p, b) for p, b in OWNERS],
            raw_GPU_query=subprocess.check_output(['nvidia-smi',
                '--query-gpu=index,utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm',
                '--format=csv,noheader,nounits'], text=True),
            raw_compute_process_query=subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                '--format=csv,noheader,nounits'], text=True)))
        if sample != 15: time.sleep(2)
    parsed = [[list(map(float, line.split(','))) for line in r['raw_GPU_query'].strip().split('\n')] for r in rows]
    summaries = []
    for gpu in range(4):
        actual = [next(row for row in sample if row[0] == gpu) for sample in parsed]
        summaries.append(dict(index=gpu, samples=len(actual), average_NVML_utilization_percent=sum(r[1] for r in actual)/len(actual),
            minimum_NVML_utilization_percent=min(r[1] for r in actual), maximum_NVML_utilization_percent=max(r[1] for r in actual),
            average_memory_utilization_percent=sum(r[2] for r in actual)/len(actual),
            minimum_allocated_MiB=min(r[3] for r in actual), maximum_allocated_MiB=max(r[3] for r in actual),
            average_power_W=sum(r[4] for r in actual)/len(actual)))
    value = dict(complete=True, actual_sample_count=16, actual_observation_seconds=rows[-1]['monotonic']-rows[0]['monotonic'],
        actual_GPU_summary=summaries, all_actual_rows=rows, clock_ticks_per_second=os.sysconf('SC_CLK_TCK'),
        neural_workers_unchanged=True, source_only_observer=True, CUDA_contexts_or_leases_created=0,
        NVML_utilization_not_a_direct_tensor_core_occupancy_measure=True,
        no_CUDA_kernel_profiler_or_measured_causal_throughput_conclusion=True, goal_achieved=False)
    (OUT / 'complete.json').write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    print(json.dumps(summaries), flush=True)
