"""Run owned-small CPU integration and retain the exact tested-source receipt."""
import hashlib
import json
import os
from pathlib import Path
import time


if __name__ == '__main__':
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19
            or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'])):
        raise ValueError('CUDA-hidden nice19 one-thread own-small CPU driver checks required')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    root = Path(__file__).resolve().parents[1]
    out = root / 'reports/batched-reasoning-owned-driver-v2'
    out.mkdir()
    import pytest
    from slm_lab.local_source_closure_v2 import python_sources
    entrypoints = [Path(__file__), root / 'scripts/train_batched_reasoning_aligned_GPU_v2.py',
        root / 'scripts/prepare_batched_reasoning_aligned_CPU_v2.py', root / 'tests/test_batched_reasoning_owned_driver_v2.py']
    def sha(path):
        with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()
    sources = {str(p): sha(p) for p in python_sources(root, entrypoints)}
    start = time.monotonic()
    code = int(pytest.main(['-q', 'tests/test_batched_reasoning_owned_driver_v2.py',
        '--basetemp', 'reports/batched-reasoning-owned-driver-v2-pytest', '--tb=short']))
    proofs = sorted((root / 'reports/batched-reasoning-owned-driver-v2-pytest').rglob('actual-owned-optimizer-*.json'))
    rows = [json.loads(p.read_text()) for p in proofs]
    valid = (code == 0 and len(rows) == 2 and {r['profile'] for r in rows} == {'anchor_CE', 'anchor_RL'}
        and all(r['actual_optimizer_class'] == 'AdamW' and r['actual_physical_optimizer_calls'] == 3
            and r['next_actual_actions_verifiers_weights_AdamW_replayed_exactly'] for r in rows)
        and all(sha(p) == h for p, h in sources.items()))
    value = dict(complete=valid, pytest_exit_code=code, elapsed_seconds=time.monotonic() - start,
        source_files_sha256=sources, actual_proofs_sha256={str(p): sha(p) for p in proofs},
        actual_small_CPU_optimizer_calls=sum(r['actual_physical_optimizer_calls'] for r in rows),
        full_size_GPUcapacity_or_training=False, quality_gain_established=False, production_updates=0, goal_achieved=False)
    destination = out / ('complete.json' if valid else 'failure.json')
    with destination.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    raise SystemExit(0 if valid else 1)
