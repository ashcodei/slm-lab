import os
from pathlib import Path
import sys


if __name__ == '__main__':
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19
            or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'])):
        raise ValueError('Native driver mechanics require CUDA hidden, nice19 and one CPU thread')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    import pytest
    raise SystemExit(pytest.main(['-q', 'tests/test_native_code_suffix_owned_driver_v1.py',
        '--basetemp', 'reports/native-code-suffix-owned-driver-v1-pytest', '--tb=short']))
