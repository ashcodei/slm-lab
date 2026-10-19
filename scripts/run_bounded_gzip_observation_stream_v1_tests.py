import os


if __name__ == '__main__':
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19:
        raise ValueError('CUDA-hidden nice19 source-only checks required')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    import pytest
    raise SystemExit(pytest.main(['-q', 'tests/test_bounded_gzip_observation_stream_v1.py',
        '--basetemp', 'reports/bounded-gzip-observation-stream-v1-pytest', '--tb=short']))
