"""Preserve the original failed Python-bound-method identity control and fixtures."""
import os


if __name__ == '__main__':
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19
            or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'])):
        raise ValueError('CUDA-hidden nice19 one-thread own-small worker required')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    import pytest
    raise SystemExit(pytest.main(['-q', 'tests/test_frozen_linear_cast_cache_v2.py',
        '--basetemp', 'reports/frozen-linear-cast-cache-v2-pytest', '--tb=short']))
