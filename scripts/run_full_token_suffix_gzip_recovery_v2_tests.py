import os


if __name__ == '__main__':
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19:
        raise ValueError('CUDA-hidden nice19 source-only checks required')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    import pytest
    raise SystemExit(pytest.main(['-q', 'tests/test_full_token_suffix_gzip_recovery_v2.py',
        '--basetemp', 'reports/full-token-suffix-gzip-recovery-v2-pytest', '--tb=short']))
