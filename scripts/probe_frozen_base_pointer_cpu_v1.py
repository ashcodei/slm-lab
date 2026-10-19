"""Disposable actual706.8M CPU proof for a fully frozen own backbone/head."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import torch

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.frozen_base_pointer_stage_v1 import FrozenBasePointerStage, make_head, make_optimizer
from slm_lab.point_grounding_data_v1 import PointerData
from slm_lab.point_grounding_evaluation_v1 import state_digest
from slm_lab.training import load_checkpoint

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports/frozen-base-pointer-full-CPU-v1'
BASE = ROOT / 'checkpoints/extended-curriculum-v1/joint/model.pt'
DATA = ROOT / 'data/localized-pointer-candidate-v1'


def main():
    if REPORTS.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise ValueError('Fresh disposable CPU proof with GPU visibility disabled required')
    torch.set_num_threads(2); torch.manual_seed(839032); torch.use_deterministic_algorithms(True)
    files = {str(path): file_hash(path) for path in [BASE, BASE.parent / 'tokenizer.json', DATA / 'admission.json',
        DATA / 'manifest.json', Path(__file__).resolve(), ROOT / 'src/slm_lab/frozen_base_pointer_stage_v1.py',
        ROOT / 'src/slm_lab/frozen_base_pointer_inference_v1.py', ROOT / 'reports/frozen-base-pointer-kernel-v1-tests.log']}
    guard = FileBindings(files)
    if '4 passed' not in (ROOT / 'reports/frozen-base-pointer-kernel-v1-tests.log').read_text():
        raise ValueError('Actual four-worker freezing/recovery/integrated-code-cache and reload checks required')
    REPORTS.mkdir()
    plan = dict(created_at=datetime.now(timezone.utc).isoformat(), source_files_sha256=files,
        CPU_threads=2, precision='CPU/FP32', disposable_original_train_query_updates=1,
        production_updates=0, whole_backbone_frozen=True, only_two_projection_matrices_train=True,
        no_GPU_capacity_or_held_gain_claimed=True, goal_achieved=False)
    atomic_json(REPORTS / 'plan.json', plan)
    started = time.monotonic()
    base, metadata = load_checkpoint(BASE, 'cpu', training_state=False)
    head = make_head(base, rank=256); data = PointerData(DATA, 'train', 'optimization')
    opts = dict(seed=839032, epochs=1, world_size=1, microbatch_per_rank=1, accumulation=1,
        global_queries_per_update=1, steps=len(data), planned_complete_query_exposures=len(data), rounding_repeated_queries=0,
        vision_learning_rate=0., head_learning_rate=2e-4, betas=[.9, .95], weight_decay=.01,
        gradient_norm_clip=1., warmup_fraction=1e-5, cosine_floor_fraction=.1,
        image_size=448, maximum_whole_query_tokens=34)
    sources = dict(base_checkpoint_sha256=file_hash(BASE), tokenizer_sha256=metadata['tokenizer_sha256'],
        run_plan_sha256=file_hash(REPORTS / 'plan.json'), data_admission_sha256=file_hash(DATA / 'admission.json'))
    kernel = FrozenBasePointerStage(head, make_optimizer(head, opts), data, opts, sources, 'cpu')
    original_versions = [p._version for p in head.base.parameters()]
    selected = kernel.positions(0, 0, 0)[0]; batch = data.batch([selected], 'cpu')
    head.eval()
    with torch.no_grad():
        before = head.base.generate(batch['input_ids'], batch['pixel_values'], max_new_tokens=2)
    record = kernel.update(); kernel.validate_Adam()
    kernel.save(REPORTS / 'disposable-reference-resume.pt', 16 * 1024**2, 26 * 1024**3)
    head.eval()
    with torch.no_grad():
        after = head.base.generate(batch['input_ids'], batch['pixel_values'], max_new_tokens=2)
    original = torch.load(BASE, map_location='cpu', weights_only=True, mmap=True)
    if (any(not torch.equal(t, original['model'][name]) for name, t in head.base.state_dict().items())
            or not torch.equal(before, after) or original_versions != [p._version for p in head.base.parameters()]
            or state_digest(head.base) != kernel.frozen_base_digest or torch.cuda.is_initialized()):
        raise ValueError('Every original base byte/version, native image generation and GPU isolation must remain unchanged')
    inventory = head.trainable_inventory()
    if inventory['total_unique_parameters'] != 706799104 or inventory['actual_trainable_parameters'] != 786432:
        raise ValueError('Actual full original backbone and exactly786432head parameters required')
    guard.check(full=True)
    snapshot = REPORTS / 'disposable-reference-resume.pt'
    complete = dict(completed_at=datetime.now(timezone.utc).isoformat(), actual_inventory=inventory,
        actual_disposable_updates=1, complete_Adam_states=len(kernel.optimizer.state),
        actual_original_train_query_id=data.rows[selected]['id'], original_source_position=selected,
        actual_image_shape=list(batch['pixel_values'].shape), actual_whole_query_tokens=int(batch['attention_mask'].sum()),
        actual_source_attention_samples=int(batch['targets'].sum()), update=record,
        all_original332_base_tensors_byte_equal=True, all_original_base_parameter_versions_unchanged=True,
        native_image_generation_bit_identical=True, frozen_base_digest=kernel.frozen_base_digest,
        full_backbone_reference_not_copied_into_snapshot=True, actual_reference_snapshot_bytes=snapshot.stat().st_size,
        actual_reference_snapshot_sha256=file_hash(snapshot), CUDA_initialized=False, CPU_threads=2,
        elapsed_seconds=time.monotonic() - started, production_updates=0,
        actual_full_GPU_capacity_or_grounding_gain_established=False, goal_achieved=False)
    atomic_json(REPORTS / 'complete.json', complete)
    print(json.dumps(complete, indent=2), flush=True)


if __name__ == '__main__':
    main()
