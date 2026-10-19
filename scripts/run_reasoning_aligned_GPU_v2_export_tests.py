import os


if __name__ == '__main__':
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19
            or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'])):
        raise ValueError('Only CUDA-hidden nice19 single-thread own-small export checks are allowed')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    import pytest
    raise SystemExit(pytest.main(['-q', 'tests/test_reasoning_aligned_GPU_v2.py', '-k',
        'actual_trained_parts_export_complete_native_values_critic_and_real_cold_outputs',
        '--basetemp', 'reports/reasoning-aligned-GPU-v2-export-pytest', '--tb=short']))
