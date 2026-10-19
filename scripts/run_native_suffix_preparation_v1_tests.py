import os
from pathlib import Path
import sys


if __name__ == '__main__':
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19
            or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'])):
        raise ValueError('Source-preparation checks require CUDA hidden, nice19 and one CPU thread')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import pytest
    raise SystemExit(pytest.main(['-q', 'tests/test_native_suffix_preparation_v1.py',
        '--basetemp', 'reports/native-suffix-preparation-v1-pytest', '--tb=short']))
