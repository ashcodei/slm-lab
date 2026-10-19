import os


if __name__ == '__main__':
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19:
        raise ValueError('Only CUDA-hidden nice19 own-small source checks are allowed')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    import pytest
    raise SystemExit(pytest.main(['-q', 'tests/test_reasoning_native_position_export_guard_v1.py',
        '--basetemp', 'reports/reasoning-native-position-export-guard-v1-pytest', '--tb=short']))
