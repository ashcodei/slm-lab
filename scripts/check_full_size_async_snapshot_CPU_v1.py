"""Full-size CPU snapshot/I/O proof using the actual completed dense state.

No model is constructed or optimized, and CUDA remains hidden. Capture a
private CPU copy, promptly release the original checkpoint mmap, then test
background serialization and whole-state readback in volatile scratch. This
does not admit full-size GPU asynchronous recovery or prove a speedup.
"""
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'reports/durable-distributed-code-v1'
OUT = ROOT/'reports/full-size-async-snapshot-CPU-v1'
SCRATCH = Path('/tmp/slm-full-size-async-snapshot-CPU-v1')
GIB = 1024**3


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(name, value):
    p = OUT/name; q = p.with_suffix(p.suffix+'.pending')
    with q.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(q, p)


def main():
    if OUT.exists() or SCRATCH.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('A fresh explicit CPU-only full-size snapshot proof is required')
    os.nice(10); allowed = sorted(os.sched_getaffinity(0)); os.sched_setaffinity(0, {allowed[min(2, len(allowed)-1)]})
    import torch
    from slm_lab.async_durable_checkpoint_v2 import BoundedAsyncCheckpoint, freeze_cpu_bounded
    from slm_lab.codenet_checkpoint_v1 import atomic_checkpoint, capture_rng
    from slm_lab.local_source_closure_v2 import python_sources
    from slm_lab.parallel_function_repair_stage_v1 import value_digest
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    stat = dict(line.split()[:2] for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())
    available = int(Path('/sys/fs/cgroup/memory.max').read_text()) - sum(int(stat[k]) for k in ['anon', 'shmem', 'kernel'])
    if available < 96*GIB or shutil.disk_usage('/tmp').free < 52*GIB or shutil.disk_usage(ROOT).free < 8*GIB:
        raise ValueError('Full 48GiB own/RSS plus48GiB other-job funding, scratch and persistent reserves required')
    resource.setrlimit(resource.RLIMIT_AS, (64*GIB, 64*GIB))
    marker_path = RUN/'durable-state.json'; marker = json.loads(marker_path.read_text())
    if marker['profile'] != 'rate3e5' or marker['completed'] != 230:
        raise ValueError('Capture the actual second-arm final state before its slot is legitimately reused')
    source = Path(marker['checkpoint']); tests = ROOT/'reports/distributed-async-dense-state-v1-final-tests.log'
    if '3 passed' not in tests.read_text():
        raise ValueError('The actual four-replica overlap/recovery/failure controls must pass first')
    files = {str(p): sha(p) for p in [tests, RUN/'study.json', *python_sources(ROOT, [Path(__file__).resolve()])]}
    OUT.mkdir(); SCRATCH.mkdir(); start = time.perf_counter(); rng = value_digest(capture_rng('cpu'))
    write('plan.json', dict(at=datetime.now(timezone.utc).isoformat(), owner=dict(pid=os.getpid(),
        birth=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]),
        original_checkpoint_marker=marker, source_files_sha256=files, maximum_private_snapshot_tensor_bytes=12*GIB,
        maximum_process_virtual_bytes=64*GIB, minimum_other_job_host_bytes=48*GIB,
        no_model_optimizer_or_CUDA_operations=True, CPU_affinity=sorted(os.sched_getaffinity(0)),
        source_checkpoint_slot_may_legitimately_change_after_private_copy=True,
        output_checkpoint_volatile=True, GPU_recovery_and_throughput_admission_still_required=True))

    def capture():
        if sha(source) != marker['sha256']:
            raise ValueError('The actual original completed slot changed before capture')
        state = torch.load(source, map_location='cpu', weights_only=True, mmap=True)
        common = state['common']; native = common['model']
        if (state['world_size'] != 4 or len(state['private_rngs']) != 4 or common['completed'] != 230
                or common['planned'] != 230 or len(common['metrics']) != 230
                or native['actual_counts']['total_unique'] != 984399690
                or len(common['optimized_names']) != 317 or len(common['optimizer']['state']) != 317
                or native['sources']['profile_sha256'] != hashlib.sha256(b'rate3e5').hexdigest()):
            raise ValueError('The actual full-size trained model/all317Adam/allfourRNG state is required')
        positions = [p for row in common['metrics'] for p in row['actual_positions']]
        if sorted(positions) != list(range(3675)):
            raise ValueError('Every original once-only train view must survive in the actual complete state')
        for adam in common['optimizer']['state'].values():
            if float(adam['step']) != 230 or any(not bool(torch.isfinite(adam[k]).all()) for k in ['exp_avg', 'exp_avg_sq']):
                raise ValueError('Every actual final finite Adam moment and clock is required')
        private, tensor_bytes = freeze_cpu_bounded(state, 12*GIB)
        if sha(source) != marker['sha256']:
            raise ValueError('The canonical slot changed before its private CPU copy closed')
        return private, tensor_bytes

    private, tensor_bytes = capture(); gc.collect()
    # capture() has returned: no tensor backed by the canonical mmap is kept.
    expected = value_digest(private)
    write('captured.json', dict(actual_source_sha256=marker['sha256'], immutable_private_tensor_bytes=tensor_bytes,
        expected_complete_value_digest=expected, original_mmap_released=True, seconds=time.perf_counter()-start))
    entered = threading.Event(); release = threading.Event()
    def delayed(path, state, maximum, reserve):
        entered.set()
        if not release.wait(120):
            raise ValueError('The explicit snapshot-mutation control did not release I/O')
        atomic_checkpoint(path, state, maximum, reserve)
    writer = BoundedAsyncCheckpoint(maximum_snapshot_tensor_bytes=12*GIB, writer=delayed)
    path = SCRATCH/'state.pt'; submitted = time.perf_counter()
    try:
        writer.submit(path, private, 12*GIB, 40*GIB)
        submit_seconds = time.perf_counter()-submitted
        if not entered.wait(30) or path.exists() or writer.check() is not None:
            raise ValueError('The complete snapshot must be frozen before background I/O is released')
        # Mutate only the caller's disposable CPU copy, never a live model,
        # original checkpoint, registered worker, source pack or oracle.
        private['common']['model']['network']['embedding.weight'].view(-1)[0].add_(1)
        private['common']['optimizer']['state'][0]['exp_avg'].view(-1)[0].add_(1)
        private['common']['metrics'][0]['actual_positions'][0] = -12345
        private['private_rngs'][0]['torch'][0].bitwise_xor_(1)
        if value_digest(private) == expected or writer.pending.done():
            raise ValueError('The disposable copy must change while the full private writer snapshot stays in flight')
        release.set(); receipt = writer.flush()
    finally:
        release.set(); writer.close()
    saved = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    if value_digest(saved) != expected:
        raise ValueError('Every actual full-size original model/Adam/metric/four-private-RNG value must read back exactly')
    table = saved['common']['model']['network']
    if table['embedding.weight'].data_ptr() != table['lm_head.weight'].data_ptr():
        raise ValueError('The complete saved checkpoint must preserve one actual shared vocabulary storage')
    if torch.cuda.is_initialized() or value_digest(capture_rng('cpu')) != rng:
        raise ValueError('No CUDA context or caller CPU random-state change is allowed')
    for name, digest in files.items():
        if sha(name) != digest:
            raise ValueError('An original snapshot implementation or control source changed')
    write('complete.json', dict(complete=True, actual_parameter_values=984399690, complete_language_Adam_states=317,
        original_completed_updates=230, all_four_original_private_RNG_records=4,
        original_checkpoint_sha256=marker['sha256'], complete_value_digest=expected,
        original_mmap_released_before_background_IO=True, immutable_private_tensor_bytes=tensor_bytes,
        durable_scratch_receipt=receipt, tied_vocabulary_storage_preserved=True,
        caller_copy_model_Adam_metric_and_rng_mutations_did_not_change_saved_state=True,
        submit_copy_seconds=submit_seconds, seconds=time.perf_counter()-start,
        peak_process_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        source_files_sha256=files, no_actual_model_optimizer_update_or_neural_forward=True,
        CUDA_private_rng_bytes_preserved_not_executed=True, no_GPU_context_created=True,
        full_GPU_capacity_recovery_and_speedup_still_unverified=True,
        no_checkpoint_promoted_or_training_scope_changed=True, goal_achieved=False))


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        if OUT.exists():
            write('failure.json', dict(error=repr(exc), traceback=traceback.format_exc(), goal_achieved=False))
        raise
