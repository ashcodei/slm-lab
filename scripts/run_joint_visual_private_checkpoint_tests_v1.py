"""Use an importable real main file for the actual four-replica CPU tests."""
import os


if __name__ == '__main__':
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19
            or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'])):
        raise ValueError('One-thread CUDA-hidden nice19 own-small-neural test worker required')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    import pytest
    raise SystemExit(pytest.main(['-q', 'tests/test_joint_visual_private_checkpoint_v1.py',
        '--basetemp', 'reports/joint-visual-private-checkpoint-v1-file-launch-pytest', '--tb=short']))
