"""Actual importable CPU-only strict visual snapshot/recovery controls."""
import os


if __name__ == '__main__':
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19
            or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'])):
        raise ValueError('One-thread CUDA-hidden nice19 small-own-neural recovery test worker required')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    import pytest
    raise SystemExit(pytest.main(['-q', 'tests/test_joint_visual_private_checkpoint_v2.py',
        'tests/test_joint_visual_durable_state_v2.py', 'tests/test_joint_visual_snapshot_guards_v2.py',
        '--basetemp', 'reports/joint-visual-recovery-v2-pytest', '--tb=short']))
