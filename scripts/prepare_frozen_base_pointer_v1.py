"""Bind the full original head-only ablation without altering active code work."""
from datetime import datetime, timezone
import json
from pathlib import Path

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from train_frozen_base_pointer_v1 import ROOT, REPORTS, OUTPUT, DATA, BASE, TOKEN, PREVIOUS, PREVIOUS_QUEUE


def main():
    if REPORTS.exists() or OUTPUT.exists():
        raise ValueError('Preserve prior head-only hypothesis and artifacts; no implicit overwrite')
    checks = [('frozen-base-pointer-kernel-v1-tests.log', '4 passed'), ('frozen-base-pointer-driver-v1-tests.log', '3 passed')]
    for name, expected in checks:
        if expected not in (ROOT / 'reports' / name).read_text():
            raise ValueError('Actual head-only mean/recovery/base/code-cache/reload and owned resource checks required')
    cpu = json.loads((ROOT / 'reports/frozen-base-pointer-full-CPU-v1/complete.json').read_text())
    if (cpu['actual_inventory']['total_unique_parameters'] != 706799104
            or cpu['actual_inventory']['actual_trainable_parameters'] != 786432
            or not cpu['all_original332_base_tensors_byte_equal']
            or not cpu['all_original_base_parameter_versions_unchanged']
            or not cpu['native_image_generation_bit_identical'] or cpu['CUDA_initialized']):
        raise ValueError('Actual full-model GPU-hidden CPU proof required')
    previous = json.loads((ROOT / 'reports/point-grounding-joint-v2-queue.json').read_text())
    files = dict(previous['source_files_sha256'])
    names = ['src/slm_lab/frozen_base_pointer_stage_v1.py', 'src/slm_lab/frozen_base_pointer_inference_v1.py',
        'src/slm_lab/frozen_base_pointer_readback_v1.py', 'scripts/train_frozen_base_pointer_v1.py',
        'scripts/prepare_frozen_base_pointer_v1.py', 'scripts/probe_frozen_base_pointer_cpu_v1.py',
        'scripts/train_own_code_generation_repair_v1.py', 'tests/test_frozen_base_pointer_stage_v1.py',
        'tests/test_frozen_base_pointer_inference_v1.py', 'tests/test_frozen_base_pointer_driver_v1.py',
        'tests/test_deterministic_vision_resize_v1.py', 'tests/test_point_grounding_stage_v1.py',
        'tests/test_own_code_delta_v1.py', 'reports/frozen-base-pointer-kernel-v1-tests.log',
        'reports/frozen-base-pointer-driver-v1-tests.log', 'reports/frozen-base-pointer-full-CPU-v1/plan.json',
        'reports/frozen-base-pointer-full-CPU-v1/complete.json', 'reports/point-grounding-joint-v2/complete-neural-source-readback.json',
        'reports/point-grounding-joint-v2/matched-attention-and-native-comparison.json',
        'src/slm_lab/point_grounding_readback_v1.py', 'src/slm_lab/apps_native_metrics_v1.py']
    for name in names:
        files[str(ROOT / name)] = file_hash(ROOT / name)
    for path in [BASE, TOKEN, PREVIOUS_QUEUE]:
        files[str(path)] = file_hash(path)
    guard = FileBindings(files); guard.check(full=True)
    code_plan = json.loads((PREVIOUS / 'plan.json').read_text())
    images = code_plan['evaluation_image_files_sha256']; FileBindings(images).check(full=True)
    options = dict(seed=839032, epochs=1, world_size=4, microbatch_per_rank=2, accumulation=4,
        global_queries_per_update=32, steps=1137, planned_complete_query_exposures=36384, rounding_repeated_queries=30,
        vision_learning_rate=0., head_learning_rate=2e-4, betas=[.9, .95], weight_decay=.01,
        gradient_norm_clip=1., warmup_fraction=.1, cosine_floor_fraction=.1,
        image_size=448, maximum_whole_query_tokens=34)
    hypothesis = dict(format='own-frozen-base-original-human-attention-head-v1',
        prepared_at=datetime.now(timezone.utc).isoformat(), source_files_sha256=files,
        base_checkpoint=str(BASE), base_checkpoint_sha256=file_hash(BASE), tokenizer_sha256=file_hash(TOKEN),
        data_admission_sha256=file_hash(DATA / 'admission.json'), frozen_base_digest=cpu['frozen_base_digest'],
        head_rank=256, total_unique_parameters=706799104, actual_trainable_parameters=786432,
        compatible_combined_code_and_head_parameter_budget=986450944,
        full_combined_model_not_yet_instantiated_or_assessed=True,
        training_options=options, native_image_files_sha256=images,
        scientific_motivation='Joint attention supervision improved held mouse-grid binding but regressed original native object answers46to5/164. Freeze the entire own original base and learn only query/patch projections.',
        ablation='Same original train query order/seed, head initialization/rank, global32source mean,1epoch and head Adam schedule as joint-vision experiment; original vision learning rate is exactly0 and all base parameters are frozen.',
        capacity_required_before_production='All4actual L40s; two complete head Adam states; actual reference snapshot exact CUDA next-update/head/Adam/base-version recovery;448px/34token maximum-shape backward; discard probes.',
        all_original_train_queries=36354, original_held_validation_queries=2027,
        all_original_queries_photos_and_soft_sample_targets_unchanged=True,
        full_native_OCR_object_caption_outputs_must_remain_exact_on_matched_frozen_base_view=True,
        frozen_original_backbone_never_copied_into_reference_checkpoint=True,
        all_raw_attention_logits_and_original_pixel_query_target_prior_interventions_retained=True,
        approximate_human_attention_not_object_boxes_or_OCR_truth=True,
        original_development_validation_not_certified_final_independence=True,
        genuine_human_preference_RLHF_not_created=True, no_external_encoder_or_teacher=True,
        owned_GPU_window_only_after_exact_existing_code_owner_and_registered_workers_end=True,
        actual_new_production_updates=0, full_GPU_capacity_unproven=True, candidate_promoted=False, goal_achieved=False)
    REPORTS.mkdir(); atomic_json(REPORTS / 'hypothesis.json', hypothesis)
    print(json.dumps(dict(hypothesis=str(REPORTS / 'hypothesis.json'), source_files=len(files), native_images=len(images),
                         actual_parameters=706799104, actual_trainable=786432, planned_updates=1137,
                         entire_original_backbone_frozen=True, full_GPU_capacity_unproven=True), indent=2), flush=True)


if __name__ == '__main__':
    main()
