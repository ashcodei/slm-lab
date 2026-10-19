import os


if __name__ == '__main__':
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19
            or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'])):
        raise ValueError('Own-small checks require CUDA-hidden nice19 and single-thread math')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    import pytest
    raise SystemExit(pytest.main(['-q', 'tests/test_batched_reasoning_capacity_controller_v1.py',
        '--basetemp', 'reports/batched-reasoning-capacity-controller-v1-pytest', '--tb=short']))
