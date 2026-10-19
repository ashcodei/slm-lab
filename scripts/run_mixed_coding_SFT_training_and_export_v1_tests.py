import json
import os
from pathlib import Path


if __name__ == '__main__':
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19
            or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'])):
        raise ValueError('CUDA-hidden nice19 one-thread explicit own-small CPU mechanics required')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    p = Path('reports/mixed-coding-SFT-training-and-export-v1-owner.json')
    with p.open('x') as stream:
        json.dump(dict(owner=dict(pid=os.getpid(), ticks=Path('/proc/self/stat').read_text().split(') ', 1)[1].split()[19]),
            nice=os.nice(0), scheduler=os.sched_getscheduler(0), affinity=sorted(os.sched_getaffinity(0)),
            actual_CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'], own_small_CPUmechanics_only=True), stream, indent=2)
    import pytest
    raise SystemExit(pytest.main(['-q', 'tests/test_mixed_coding_SFT_training_and_export_v1.py',
        '--basetemp', 'reports/mixed-coding-SFT-training-and-export-v1-pytest', '--tb=short']))
