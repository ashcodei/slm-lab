"""Isolated real-file spawn runner; never a full model/GPUtraining launcher."""
import os


if __name__ == '__main__':
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19
            or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'])):
        raise ValueError('CUDA-hidden nice19 one-thread CPU mechanics required')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    import pytest
    raise SystemExit(pytest.main(['-q', 'tests/test_parallel_mixed_coding_SFT_stage_v1.py',
        '--basetemp', 'reports/mixed-coding-SFT-stage-v1-pytest', '--tb=short']))
