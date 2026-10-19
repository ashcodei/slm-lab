import os


if __name__ == '__main__':
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19
            or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'])):
        raise ValueError('Actual native controller checks require CUDA hidden, nice19 and one CPU thread')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    import pytest
    import sys
    extra = ['-k', 'corrected_fixture_complete_cold_probe'] if sys.argv[1:] == ['--cold-only'] else []
    if sys.argv[1:] not in [[], ['--cold-only']]: raise ValueError('Unknown native capacity test scope')
    raise SystemExit(pytest.main(['-q', 'tests/test_native_suffix_capacity_controller_v2.py',
        '--basetemp', 'reports/native-suffix-capacity-controller-v2-pytest', '--tb=short'] + extra))
