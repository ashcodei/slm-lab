"""Measure the existing actual source guard; no models or gradients are run."""
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/native-suffix-source-guard-profile-CPU-v1'
CONTROL = ROOT / 'reports/native-suffix-CPU-mechanics-qualification-v1/closed-qualified-controls.json'


def counters():
    return {k:int(v) for k,v in [line.split(':') for line in Path('/proc/self/io').read_text().splitlines()]}


def main():
    if (OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19
            or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'])):
        raise ValueError('Fresh CUDA-hidden nice19 one-thread source-only measurement required')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    import torch
    from slm_lab.codenet_checkpoint_v1 import FileBindings
    assert not torch.cuda.is_initialized()
    proof = json.loads(CONTROL.read_text())
    files = proof['source_files_sha256']
    sizes = {p:Path(p).stat().st_size for p in files}
    def measure(action):
        before = counters(); wall = time.monotonic(); cpu = time.process_time()
        result = action()
        elapsed_cpu = time.process_time() - cpu; elapsed_wall = time.monotonic() - wall
        after = counters()
        return result, dict(wall_seconds=elapsed_wall, CPU_seconds=elapsed_cpu,
            proc_character_read_bytes=after['rchar']-before['rchar'], proc_physical_read_bytes=after['read_bytes']-before['read_bytes'],
            proc_read_syscalls=after['syscr']-before['syscr'])
    guard, construction = measure(lambda: FileBindings(files))
    _, checked = measure(guard.check)
    with CONTROL.open('rb') as stream: control_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    with Path(__file__).open('rb') as stream: script_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    result = dict(complete=True, actual_closed_source_proof_sha256=control_sha,
        actual_source_files=len(files), actual_bound_payload_bytes=sum(sizes.values()),
        actual_guard_constructor=construction, actual_single_unchanged_guard_check=checked,
        unchanged_check_small_payload_bytes=sum(v for v in sizes.values() if v <= 8 * 1024**2),
        actual_measured_existing_API=True, concurrent_original_workers_and_CPUtest_unchanged=True,
        observed_wall_time_not_an_isolated_hardware_benchmark=True,
        models_constructed=0, neural_forwards=0, GPUcontexts=0, optimizer_updates=0, goal_achieved=False,
        source_files_sha256={str(CONTROL):control_sha,str(Path(__file__).resolve()):script_sha})
    assert not torch.cuda.is_initialized()
    OUT.mkdir()
    with (OUT/'complete.json').open('x') as stream: json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['source_files_sha256']}))


if __name__ == '__main__': main()
