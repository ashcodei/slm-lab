"""Charge full-size joint selected tensors and oversized metadata to a stream.

Use actual closed native visual/word/matching tensors and actual two-update
actor/Adam tensors. Additional visual moment buffers are independently
allocated size controls, not learned moments or a valid resumable state.
No neural model, optimizer, GPU context or persistent checkpoint is created.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import traceback

ROOT = Path(__file__).resolve().parents[1]
RESERVE = 40 * 1024**3
LIMIT = 2 * 1024**3


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main(output):
    output = Path(output).absolute()
    if (output.exists() or output.resolve() != output or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19
            or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'])
            or shutil.disk_usage(output.parent).free < RESERVE + 4 * 1024**2):
        raise ValueError('A fresh funded CUDA-hidden one-thread CPU size measurement is required')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    import torch
    from slm_lab.codenet_checkpoint_v1 import capture_rng, FileBindings
    from slm_lab.local_source_closure_v2 import python_sources
    from slm_lab.parallel_function_repair_stage_v1 import value_digest
    from measure_indexed_actor_checkpoint_size_CPU_v1 import Counter
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    dense = ROOT / 'checkpoints/full-actor-fold-dense-update-CPU-v1/trained-native-student.pt'
    dense_proof = ROOT / 'reports/full-actor-fold-dense-update-CPU-v1/complete.json'
    actor = ROOT / 'reports/dense-human-actor-full-CPU-v1/resume-step2.pt'
    actor_proof = ROOT / 'reports/dense-human-actor-full-CPU-v1/complete.json'
    actor_reader = ROOT / 'reports/dense-human-actor-full-CPU-v1/independent-complete-readback.json'
    paths = [dense, dense_proof, actor, actor_proof, actor_reader, *python_sources(ROOT, [Path(__file__)])]
    files = {str(p): sha(p) for p in paths}; guard = FileBindings(files)
    if (files[str(dense)] != read(dense_proof)['source_files_sha256'][str(dense)]
            or files[str(actor)] != read(actor_proof)['retained_step2_sha256']
            or read(actor_reader)['original_complete_sha256'] != files[str(actor_proof)]):
        raise ValueError('Both actual completed original full-size sources and their readback must match')
    native = torch.load(dense, map_location='cpu', weights_only=True, mmap=True)['complete_dense']
    raw = torch.load(actor, map_location='cpu', weights_only=True, mmap=True)
    if (native['actual_counts']['total_unique'] != 984399690 or native['config'] != raw['model']['config']
            or native['settings']['word_expansion'] != 5 or raw['completed'] != 2 or raw['groups'] != ['actor']
            or len(raw['optimizer']['state']) != 561 or any(float(v['step']) != 2 for v in raw['optimizer']['state'].values())):
        raise ValueError('Actual compatible35layer native and561real two-update actor Adam states are required')
    parts = {n: t for n, t in raw['model']['parts'].items() if not n.startswith('critic.')}
    visual = {'vision.' + n.removeprefix('vision.'): t for n, t in native['network'].items() if n.startswith('vision.')}
    visual.update({'pointing.' + n.removeprefix('pointing.'): t for n, t in native['heads'].items() if n.startswith('pointing.')})
    visual.update({'word.' + n.removeprefix('word.'): t for n, t in native['heads'].items() if n.startswith('word.') and n != 'word.height.bias'})
    actor_values = sum(t.numel() for t in parts.values()); visual_values = sum(t.numel() for t in visual.values())
    if actor_values != 15305216 or actor_values + visual_values != 68123713 or len(parts) != 561 or len(visual) != 129:
        raise ValueError('Every actual full-size690leaf/68123713value joint selection must match')
    rng = value_digest(capture_rng('cpu')); output.mkdir()
    owner = dict(pid=os.getpid(), ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19])

    def publish(name, value):
        data = (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
        if len(data) > 4 * 1024**2 or shutil.disk_usage(output).free - len(data) < RESERVE:
            raise ValueError('Only bounded size metadata may be written above the unchanged scratch reserve')
        with (output / name).open('xb') as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())

    publish('plan.json', dict(owner=owner, at=datetime.now(timezone.utc).isoformat(), source_files_sha256=files,
        real_native_parameters=984399690, actual_actor_values=actor_values, actual_visual_word_matching_values=visual_values,
        actual_selected_values=actor_values + visual_values, actual_real_actor_Adam_states=561,
        independently_allocated_size_only_visual_moment_tensors=2 * len(visual),
        full_original_receipt_population=5362, oversized_ASCII_bytes_per_receipt=128 * 1024,
        oversized_reference_blocks=4, ASCII_bytes_per_reference_block=16 * 1024**2, miscellaneous_headroom_bytes=256 * 1024**2,
        ordinary_serialized_byte_limit=LIMIT, usable_training_checkpoint=False, optimizer_updates=0, goal_achieved=False))
    try:
        # Charge real separate storages; neither Adam bank aliases weights or
        # another bank. These zero buffers explicitly assert no learned state.
        moments = {n: dict(size_only_first=torch.zeros_like(t), size_only_second=torch.zeros_like(t)) for n, t in visual.items()}
        def ascii_bytes(key, size):
            return hashlib.shake_256(key.encode()).digest(size // 2).hex().encode('ascii')
        payload = dict(format='unusable-complete-joint-selected-tensor-and-metadata-size-control-v1',
            actual_two_update_actor_parts=parts, actual_two_update_actor_Adam=raw['optimizer'],
            actual_original_native_visual_word_matching=visual, size_only_independent_visual_moment_buffers=moments,
            original_actual_CPU_private_generator=raw['rng'],
            oversized_original5362receipt_blocks=[ascii_bytes('receipt-' + str(i), 128 * 1024) for i in range(5362)],
            oversized_reference_metadata=[ascii_bytes('reference-' + str(i), 16 * 1024**2) for i in range(4)],
            miscellaneous_metadata=ascii_bytes('miscellaneous', 256 * 1024**2),
            visual_buffers_are_not_learned_moments=True, usable_as_training_checkpoint=False, GPU_admission=False)
        counter = Counter(LIMIT); torch.save(payload, counter)
        guard.check(full=True)
        if value_digest(capture_rng('cpu')) != rng or torch.cuda.is_initialized():
            raise ValueError('Every original private CPU generator and hidden CUDA context must remain unchanged')
        publish('complete.json', dict(complete=True, owner=owner, at=datetime.now(timezone.utc).isoformat(), source_files_sha256=files,
            actual_complete_torch_save_bytes=counter.bytes, actual_complete_torch_save_sha256=counter.digest.hexdigest(), actual_stream_writes=counter.calls,
            actual_original_actor_values=actor_values, actual_original_visual_word_matching_values=visual_values,
            actual_full_selected_values=actor_values + visual_values, actual_selected_parameter_leaves=690,
            actual_original_two_update_actor_Adam_states=561, actual_additional_size_only_visual_moment_tensors=2 * len(visual),
            all_original5362oversized_receipt_blocks_retained=True, ordinary_serialized_byte_limit=LIMIT,
            actual_serialization_headroom_bytes=LIMIT - counter.bytes, persistent_checkpoint_bytes_written=0,
            visual_buffers_not_learned_Adam_and_no_joint_update_or_resume_claim=True,
            unchanged_original_sources_and_all_private_CPU_RNG=True, future_actual_full_GPUstate_must_still_fit_enforced_limit=True,
            full_size_GPU_capacity_recovery_training_funding_or_quality_proved=False, optimizer_updates=0, goal_achieved=False))
        print(json.dumps(dict(complete=True, actual_complete_torch_save_bytes=counter.bytes, limit=LIMIT, output=str(output))), flush=True)
    except BaseException as exc:
        publish('failure.json', dict(error=repr(exc), traceback=traceback.format_exc(), automatic_retry=False, goal_achieved=False))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
