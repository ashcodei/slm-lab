"""Actual sub1B shared architecture probe; grounding weights are untrained.

Load completed own code training, attach a fresh head only to verify physical
sharing/counts/route parity, and export no model or trained-head manifest.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import torch

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.frozen_base_pointer_stage_v1 import make_head
from slm_lab.own_assistant_v1 import OwnAssistant
from slm_lab.own_code_training_inference_v1 import load
from slm_lab.point_grounding_data_v1 import PointerData
from slm_lab.point_grounding_evaluation_v1 import state_digest

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports/own-assistant-full-CPU-v1'
CODE = ROOT / 'checkpoints/own-code-generation-repair-v1/model.pt'
CODE_REPORTS = ROOT / 'reports/own-code-generation-repair-v1'
BASE = ROOT / 'checkpoints/extended-curriculum-v1/joint/model.pt'
TOKEN = BASE.parent / 'tokenizer.json'
INITIAL_PLAN = ROOT / 'reports/own-code-delta-v1/plan.json'


def main():
    if REPORTS.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise ValueError('Fresh GPU-hidden CPU structural probe required')
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    if '3 passed' not in (ROOT / 'reports/own-assistant-v1-tests.log').read_text():
        raise ValueError('Actual single-base trained-fixture loader/route/cache/provenance checks required')
    training = json.loads((CODE_REPORTS / 'plan.json').read_text())
    receipt = json.loads((CODE_REPORTS / 'training-complete.json').read_text())
    if receipt['actual_updates'] != 815:
        raise ValueError('Actual complete current own-code components required')
    names = ['src/slm_lab/own_assistant_v1.py', 'scripts/probe_own_assistant_full_cpu_v1.py',
        'tests/test_own_assistant_v1.py', 'reports/own-assistant-v1-tests.log',
        'src/slm_lab/frozen_base_pointer_stage_v1.py', 'src/slm_lab/own_code_training_inference_v1.py',
        'src/slm_lab/own_code_delta_inference_v1.py', 'src/slm_lab/own_code_delta_v1.py',
        'reports/own-code-generation-repair-v1/training-complete.json',
        'reports/frozen-base-pointer-full-CPU-v1/complete.json', 'data/localized-pointer-candidate-v1/admission.json']
    files = {str(ROOT / name): file_hash(ROOT / name) for name in names}
    for path in [BASE, TOKEN, CODE, INITIAL_PLAN, CODE_REPORTS / 'plan.json']:
        files[str(path)] = file_hash(path)
    guard = FileBindings(files); REPORTS.mkdir()
    plan = dict(created_at=datetime.now(timezone.utc).isoformat(), source_files_sha256=files,
        actual_code_training_updates=815, grounding_head_random_initialization_only=True,
        actual_grounding_production_weights_loaded=False, CPU_threads=2, precision='CPU/FP32',
        public_sources_role='One actual original pointer train query/photo; no held evaluation gradients or outputs',
        scope='Actual complete shared architecture/counts/route/cache parity only; no learned grounding or high-end gain claim.',
        model_weights_exported=False, goal_achieved=False)
    atomic_json(REPORTS / 'plan.json', plan); started = time.monotonic()
    language, metadata = load(CODE, BASE, TOKEN, INITIAL_PLAN, CODE_REPORTS / 'plan.json',
        expected_checkpoint_sha256=receipt['trained_components_sha256'], expected_base_sha256=training['base_checkpoint_sha256'],
        expected_target_sha256=training['own_code_target_sha256'], expected_initial_plan_sha256=training['initial_plan_sha256'],
        expected_training_plan_sha256=file_hash(CODE_REPORTS / 'plan.json'), device='cpu')
    data = PointerData(ROOT / 'data/localized-pointer-candidate-v1', 'train', 'optimization')
    earlier = json.loads((ROOT / 'reports/frozen-base-pointer-full-CPU-v1/complete.json').read_text())
    position = earlier['original_source_position']; batch = data.batch([position], 'cpu')
    ids, pixels = batch['input_ids'], batch['pixel_values']
    base_before = state_digest(language.base); code_before = state_digest(language.parts)
    versions = [(id(p), p._version, p.data_ptr()) for p in language.base.parameters()]
    with torch.no_grad():
        outputs = {route: language.generate(ids, pixels, route=route, max_new_tokens=2).clone() for route in ['general', 'code']}
        cache = language.code.prefill(ids)['cache']
        next_ids = torch.tensor([[4]])
        cached_expected = language.code.decode(next_ids, cache)['logits'].clone()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(839032)
        head = make_head(language.base, rank=256)
    combined = OwnAssistant(language, head).eval(); counts = combined.counts()
    if counts != dict(shared_original_backbone=706012672, additional_code=279651840,
                      additional_grounding=786432, total_unique=986450944):
        raise ValueError('Every actual shared original/code/head parameter must fit below1B')
    with torch.no_grad():
        for route in ['general', 'code']:
            if not torch.equal(outputs[route], combined.generate(ids, pixels, route=route, max_new_tokens=2)):
                raise ValueError('Actual full-model original image/general/code generation must remain exact')
        cached = combined.language.code.decode(next_ids, cache)['logits']
        if not torch.equal(cached_expected, cached):
            raise ValueError('The actual full-model pre-existing conditional-code cache must remain valid/exact')
        located = combined.locate(ids, pixels, attention_mask=batch['attention_mask'], valid_cells=batch['valid_cells'])
    if (not bool(torch.isfinite(located['logits']).all()) or base_before != state_digest(language.base)
            or code_before != state_digest(language.parts)
            or versions != [(id(p), p._version, p.data_ptr()) for p in language.base.parameters()]
            or torch.cuda.is_initialized()):
        raise ValueError('Every actual base/code byte/version, finite structural head and CUDA isolation must hold')
    guard.check(full=True)
    complete = dict(completed_at=datetime.now(timezone.utc).isoformat(), actual_counts=counts,
        actual_original_base_object_shared=combined.language.base is combined.grounding.base,
        actual_original_vision_object_shared=combined.language.code.vision is combined.grounding.base.vision,
        actual_code_training_updates=815, actual_grounding_head_updates=0,
        grounding_head_random_initialization_only=True, no_trained_grounding_file_or_manifest_created=True,
        original_train_query_id=data.rows[position]['id'], original_train_image_shape=list(pixels.shape),
        original_query_tokens=int(batch['attention_mask'].sum()),
        full_original_general_and_code_image_generation_bit_identical=True,
        original_base_and_trained_code_components_byte_identical=True, original_base_versions_and_pointers_unchanged=True,
        pre_existing_full_model_code_cache_decode_bit_identical=True,
        ground_output_shape=list(located['logits'].shape), ground_logits_finite=True,
        ground_logits_are_random_head_structural_output_not_accuracy=True,
        original_base_digest=base_before, trained_code_parts_digest=code_before,
        full_extra_code_source_loaded=False, external_model_or_teacher_loaded=False,
        model_weights_exported=False, additional_neural_updates=0, CUDA_initialized=False,
        elapsed_seconds=time.monotonic() - started, full_actual_GPU_joint_capacity_and_grounding_quality_unproven=True,
        candidate_promoted=False, goal_achieved=False)
    atomic_json(REPORTS / 'complete.json', complete)
    print(json.dumps(complete, indent=2), flush=True)


if __name__ == '__main__':
    main()
