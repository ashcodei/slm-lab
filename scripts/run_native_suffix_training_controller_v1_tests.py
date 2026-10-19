import os


if __name__ == '__main__':
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19
            or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'])):
        raise ValueError('Native training-controller mechanics require CUDA hidden, nice19 and one thread')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    import pytest
    import sys
    args = sys.argv[1:]
    if args not in [[], ['--gates-only']]: raise ValueError('Unknown native controller test scope')
    suffix = 'gates' if args else 'full'
    extra = ['-k', 'not actual_four_replica'] if args else []
    raise SystemExit(pytest.main(['-q', 'tests/test_native_suffix_training_controller_v1.py',
        '--basetemp', 'reports/native-suffix-training-controller-v1-' + suffix + '-pytest', '--tb=short'] + extra))
