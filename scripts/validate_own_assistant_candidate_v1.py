"""Actual trained986M factory/route/cache CPU validation, without new updates."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import torch

from slm_lab import own_assistant_v1 as factory
from slm_lab import own_code_delta_inference_v1 as code_loader
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.frozen_base_pointer_stage_v1 import make_head
from slm_lab.point_grounding_data_v1 import PointerData
from slm_lab.point_grounding_evaluation_v1 import state_digest

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports/own-assistant-candidate-v1'
MANIFEST = ROOT / 'checkpoints/own-assistant-candidate-v1/manifest.json'


def main():
    if (REPORTS / 'validation-complete.json').exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise ValueError('Fresh GPU-hidden complete trained-candidate validation required')
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    spec = json.loads(MANIFEST.read_text())
    files = {str(MANIFEST): file_hash(MANIFEST)}
    for name in ['base_checkpoint', 'tokenizer', 'code_checkpoint', 'code_initial_plan', 'code_training_plan', 'grounding_head', 'grounding_plan']:
        files[str((MANIFEST.parent / spec[name]).resolve())] = spec[name + '_sha256']
    for path in [Path(__file__).resolve(), ROOT / 'src/slm_lab/own_assistant_v1.py',
                 ROOT / 'tests/test_own_assistant_v1.py', ROOT / 'reports/own-assistant-v1-tests.log',
                 ROOT / 'reports/frozen-base-pointer-full-CPU-v1/complete.json',
                 ROOT / 'data/localized-pointer-candidate-v1/admission.json']:
        files[str(path)] = file_hash(path)
    guard = FileBindings(files)
    atomic_json(REPORTS / 'validation-plan.json', dict(created_at=datetime.now(timezone.utc).isoformat(),
        source_files_sha256=files, precision='CPU/FP32', CPU_threads=2,
        actual_trained_code_updates=815, actual_trained_grounding_updates=1137,
        no_fresh_random_head_used=True, original_train_photo_query_only=True,
        scope='Full actual trained factory sharing/counts/route/cache parity; not a new held capability assessment.',
        model_updates=0, goal_achieved=False))
    data = PointerData(ROOT / 'data/localized-pointer-candidate-v1', 'train', 'optimization')
    previous = json.loads((ROOT / 'reports/frozen-base-pointer-full-CPU-v1/complete.json').read_text())
    position = previous['original_source_position']; batch = data.batch([position], 'cpu')
    ids, pixels = batch['input_ids'], batch['pixel_values']
    native_load = code_loader.load_checkpoint; original_attach = factory.attach_grounding
    calls = []; proofs = []
    def counted_load(path, *args, **kwargs):
        calls.append(str(Path(path).resolve()))
        return native_load(path, *args, **kwargs)
    def checked_attach(language, checkpoint, *args, **kwargs):
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if payload['actual_steps'] != 1137:
            raise ValueError('Actual complete trained head required; no seeded placeholder')
        versions = [(id(p), p._version, p.data_ptr()) for p in language.base.parameters()]
        base_before = state_digest(language.base); code_before = state_digest(language.parts)
        with torch.random.fork_rng(devices=[]):
            baseline_head = make_head(language.base, rank=payload['head_rank'])
        baseline_head.query_projection.load_state_dict({'weight': payload['projections']['query_projection.weight']})
        baseline_head.patch_projection.load_state_dict({'weight': payload['projections']['patch_projection.weight']})
        baseline_head.eval(); language.eval()
        with torch.no_grad():
            expected = {route: language.generate(ids, pixels, route=route, max_new_tokens=2).clone() for route in ['general', 'code']}
            expected_ground = baseline_head(ids, pixels, attention_mask=batch['attention_mask'], valid_cells=batch['valid_cells'])['logits'].clone()
            cached = language.code.prefill(ids)['cache']; next_token = torch.tensor([[4]])
            expected_decode = language.code.decode(next_token, cached)['logits'].clone()
        rng = torch.get_rng_state().clone()
        combined, metadata = original_attach(language, checkpoint, *args, **kwargs)
        if not torch.equal(rng, torch.get_rng_state()):
            raise ValueError('Attaching a completed head must preserve the caller CPU RNG')
        with torch.no_grad():
            for route in ['general', 'code']:
                if not torch.equal(expected[route], combined.generate(ids, pixels, route=route, max_new_tokens=2)):
                    raise ValueError('Every actual general/code image-generation probe must match before/after attachment')
            if not torch.equal(expected_ground, combined.locate(ids, pixels, attention_mask=batch['attention_mask'], valid_cells=batch['valid_cells'])['logits']):
                raise ValueError('Actual completed standalone and attached grounding predictions must match')
            if not torch.equal(expected_decode, combined.language.code.decode(next_token, cached)['logits']):
                raise ValueError('Actual pre-existing full code cache must stay valid and produce exact logits')
        if (versions != [(id(p), p._version, p.data_ptr()) for p in language.base.parameters()]
                or base_before != state_digest(language.base) or code_before != state_digest(language.parts)):
            raise ValueError('All original/learned code bytes, original pointers and versions must remain unchanged')
        proofs.append(dict(original_base_digest=base_before, trained_code_parts_digest=code_before,
            completed_head_steps=payload['actual_steps'], general_image_code_generation_and_ground_logits_exact=True,
            pre_existing_code_cache_logits_exact=True, original_base_versions_and_pointers_unchanged=True,
            original_and_code_weight_bytes_unchanged=True, caller_CPU_RNG_unchanged=True))
        return combined, metadata
    code_loader.load_checkpoint = counted_load; factory.attach_grounding = checked_attach
    try:
        model, metadata = factory.load_manifest(MANIFEST, expected_manifest_sha256=file_hash(MANIFEST), device='cpu')
    finally:
        code_loader.load_checkpoint = native_load; factory.attach_grounding = original_attach
    if (len(calls) != 1 or len(proofs) != 1 or model.counts()['total_unique'] != 986450944
            or model.language.base is not model.grounding.base or torch.cuda.is_initialized()):
        raise ValueError('One actual original base and complete shared sub1B count with no CUDA initialization required')
    guard.check(full=True)
    complete = dict(completed_at=datetime.now(timezone.utc).isoformat(), manifest_sha256=file_hash(MANIFEST),
        actual_counts=model.counts(), original_backbone_load_calls=calls, full_backbones_loaded_by_attachment=0,
        actual_trained_code_updates=815, actual_trained_grounding_updates=1137,
        no_random_head_placeholder=True, actual_all_three_routes_validated=True,
        original_train_query_id=data.rows[position]['id'], actual_pixel_shape=list(pixels.shape),
        route_parity_proof=proofs[0], full_code_source_policy_or_external_model_loaded=False,
        full_GPU_joint_inference_not_exercised=True, CUDA_initialized=False, model_updates=0,
        component_scores_are_not_new_standardized_joint_model_benchmarks=True,
        fresh_genuine_human_alignment_of_composite_not_complete=True, candidate_promoted=False, goal_achieved=False)
    atomic_json(REPORTS / 'validation-complete.json', complete)
    print(json.dumps(complete, indent=2), flush=True)


if __name__ == '__main__':
    main()
