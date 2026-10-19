"""Measure ordinary checkpoint serialization using real full-size actor/Adam.

The original trained disposable tensors keep their real two-update clocks.
Oversized ASCII metadata stresses the complete future receipt population.
Bytes stream to a counted SHA sink, never a usable checkpoint. This is a size
measurement, not native training, recovery admission or a model-quality gain.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/indexed-full-actor-state-size-CPU-v1'


class Counter:
    def __init__(self, maximum):
        self.maximum, self.bytes, self.calls = maximum, 0, 0
        self.digest = hashlib.sha256()

    def write(self, raw):
        if self.bytes + len(raw) > self.maximum:
            raise ValueError('The complete ordinary serialization exceeds the declared state limit')
        self.digest.update(raw); self.bytes += len(raw); self.calls += 1
        return len(raw)

    def tell(self): return self.bytes

    def flush(self): pass


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or OUT.exists():
        raise ValueError('A fresh CUDA-hidden source-bound size measurement is required')
    os.setpriority(os.PRIO_PROCESS, 0, 19)
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0)); os.sched_setaffinity(0, {20})
    import torch
    from slm_lab.codenet_checkpoint_v1 import capture_rng
    from slm_lab.corpus import file_hash
    from slm_lab.parallel_function_repair_stage_v1 import value_digest
    from slm_lab.reserved_atomic_metadata_v1 import write_json
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    original = ROOT / 'reports/dense-human-actor-full-CPU-v1'
    inputs = [Path(__file__).resolve(), original / 'complete.json', original / 'independent-complete-readback.json', original / 'resume-step2.pt']
    bindings = {str(p): file_hash(p) for p in inputs}
    def guard():
        if torch.cuda.is_initialized() or any(file_hash(p) != h for p, h in bindings.items()):
            raise ValueError('Every exact original input and the CPU-only context must remain')
    done = json.loads(inputs[1].read_text()); reader = json.loads(inputs[2].read_text())
    if (done['actual_counts']['actual_unique'] != 999706442 or done['actual_actor_values'] != 15305216
            or done['retained_step2_sha256'] != bindings[str(inputs[3])]
            or reader['original_complete_sha256'] != bindings[str(inputs[1])]
            or reader['all_saved_policy_tensors_exact_shape_dtype_finite_and_unused_critic_zero'] is not True):
        raise ValueError('The actual closed full-size own actor/Adam and independent readback are required')
    state = torch.load(inputs[3], map_location='cpu', weights_only=True, mmap=True)
    if (state['completed'] != 2 or state['groups'] != ['actor'] or len(state['optimizer']['state']) != 561
            or sum(t.numel() for t in state['model']['parts'].values()) != 15306752
            or any(float(v['step']) != 2 for v in state['optimizer']['state'].values())):
        raise ValueError('Original actual actor/critic counts and every real two-update Adam clock must remain')
    rng = value_digest(capture_rng('cpu'))
    OUT.mkdir()
    def publish(name, value):
        write_json(OUT / name, value, root=ROOT, minimum_free_bytes=40 * 1024**3,
            maximum_metadata_bytes=4 * 1024**2, guard=guard)
    owner = dict(pid=os.getpid(), ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19])
    publish('plan.json', dict(at=datetime.now(timezone.utc).isoformat(), owner=owner, source_files_sha256=bindings,
        real_actor_values=15305216, real_Adam_states=561, real_saved_optimizer_clock=2,
        stress_receipt_population=11172, oversized_ASCII_bytes_per_receipt=32 * 1024,
        oversized_reference_blocks=4, ASCII_bytes_per_reference_block=16 * 1024**2,
        extra_miscellaneous_metadata_bytes=128 * 1024**2, maximum_ordinary_serialized_bytes=1024**3,
        new_neural_optimizer_calls=0, durable_checkpoint_or_recovery_admission=False, goal_achieved=False))
    def ascii_bytes(key, size):
        return hashlib.shake_256(key.encode()).digest(size // 2).hex().encode('ascii')
    payload = dict(format='unusable-source-shaped-full-actor-serialization-size-measurement-v1',
        actual_two_update_model=state['model'], actual_two_update_Adam=state['optimizer'], actual_original_CPU_RNG=state['rng'],
        size_only_11172_independent_receipt_blocks=[ascii_bytes('receipt-' + str(i), 32 * 1024) for i in range(11172)],
        size_only_reference_headroom=[ascii_bytes('reference-' + str(i), 16 * 1024**2) for i in range(4)],
        size_only_miscellaneous_headroom=ascii_bytes('miscellaneous', 128 * 1024**2),
        not_a_training_state=True, production_training_admitted=False)
    counter = Counter(1024**3)
    torch.save(payload, counter)
    guard()
    if value_digest(capture_rng('cpu')) != rng: raise ValueError('The size measurement must preserve original global RNG')
    result = dict(complete=True, at=datetime.now(timezone.utc).isoformat(), owner=owner,
        actual_complete_torch_save_bytes=counter.bytes, actual_complete_torch_save_sha256=counter.digest.hexdigest(),
        actual_stream_write_calls=counter.calls, actual_original_actor_values=15305216, actual_original_Adam_states=561,
        original_two_update_tensor_values_and_clocks_unchanged=True, actual_oversized_receipt_population=11172,
        entire_output_streamed_without_retaining_serialized_bytes=True, persistent_checkpoint_bytes_written=0,
        declared_state_limit_bytes=1024**3, actual_remaining_serialization_headroom_bytes=1024**3-counter.bytes,
        every_original_input_and_all_CPU_RNG_unchanged=True, actual_future_native_full_state_still_requires_runtime_measurement=True,
        full_size_GPU_capacity_or_recovery_or_quality_proved=False, new_neural_optimizer_calls=0, goal_achieved=False)
    publish('complete.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__': main()
