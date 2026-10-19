import os


if __name__ == '__main__':
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19:
        raise ValueError('CUDA-hidden idle source-only controls required')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    import pytest
    raise SystemExit(pytest.main(['-q', 'tests/test_independent_token_suffix_readback_v2.py', '--tb=short']))
