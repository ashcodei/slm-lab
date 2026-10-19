"""Seal exact source draws/objectives and the owned next coding experiment."""
from datetime import datetime, timezone
import json
from pathlib import Path

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.evaluation_provenance import capture
from slm_lab.model import ModelConfig
from slm_lab.own_code_training_readback_v1 import exposure_budget
from train_own_code_generation_repair_v1 import ROOT, REPORTS, OUTPUT, INITIAL, BASE, TOKEN, INITIAL_PLAN, FOLDERS, PREVIOUS_QUEUE, streams


def main():
    if REPORTS.exists() or OUTPUT.exists():
        raise ValueError('Preserve original preparation/run; no implicit overwrite')
    for name, expected in [('own-code-training-kernel-v1-tests.log', '3 passed'), ('own-code-training-driver-v1-tests.log', '3 passed')]:
        if expected not in (ROOT / 'reports' / name).read_text():
            raise ValueError('Actual complete-kernel/recovery/loader and owned-driver checks required')
    config = ModelConfig(**json.loads((ROOT / 'configs/model.json').read_text()))
    data = streams(config)
    if {name: len(stream) for name, stream in data.items()} != dict(generation=3548, repair=6520, variety=112609):
        raise ValueError('Exact independently admitted source populations required')
    options = dict(seed=624319, world_size=4, microbatch_per_rank=1, accumulation=2,
        global_examples_per_stream=8, steps=815, learning_rate=2e-5, betas=[.9, .95], weight_decay=.01,
        gradient_norm_clip=1., warmup_fraction=.1, cosine_floor_fraction=.1,
        coefficients=dict(generation=1., repair=.5, variety=.25))
    budget = exposure_budget(data, options)
    if (budget['repair']['unique_original_positions'] != 6520 or budget['repair']['repeated_original_positions']
            or budget['generation']['unique_original_positions'] != 3548
            or budget['generation']['complete_passes'] != 1 or budget['generation']['next_pass_remainder'] != 2972
            or set(budget['variety']['sampled_original_languages']) != set(data['variety'].language_positions)
            or budget['variety']['full_original_corpus_pass']):
        raise ValueError('Exact full repair/original generation and every-language sampled replay required')
    files = {str(ROOT / name): digest for name, digest in capture()['files_sha256'].items()}
    names = ['scripts/train_own_code_generation_repair_v1.py', 'scripts/prepare_own_code_generation_repair_v1.py',
        'scripts/train_point_grounding_joint_v1.py', 'scripts/assess_joint_java_comparison_v1.py',
        'src/slm_lab/own_code_delta_v1.py', 'src/slm_lab/own_code_delta_inference_v1.py',
        'src/slm_lab/own_code_training_stage_v1.py', 'src/slm_lab/own_code_training_inference_v1.py',
        'src/slm_lab/own_code_training_readback_v1.py', 'src/slm_lab/codenet_stage_v1.py',
        'src/slm_lab/codenet_runner_v1.py', 'src/slm_lab/codenet_checkpoint_v1.py', 'src/slm_lab/codenet_token_pack_v1.py',
        'src/slm_lab/codenet_replay_v1.py', 'src/slm_lab/codenet_replay_schedule_v1.py',
        'src/slm_lab/point_grounding_evaluation_v1.py', 'src/slm_lab/point_grounding_head_v1.py',
        'src/slm_lab/point_grounding_data_v1.py', 'src/slm_lab/point_grounding_protocol_v1.py',
        'src/slm_lab/apps_native_metrics_v1.py', 'tests/test_own_code_training_stage_v1.py',
        'tests/test_own_code_training_inference_v1.py', 'tests/test_own_code_training_driver_v1.py',
        'tests/test_own_code_delta_v1.py', 'reports/own-code-training-kernel-v1-tests.log',
        'reports/own-code-training-driver-v1-tests.log', 'reports/own-code-delta-public-v1/comparison.json',
        'reports/own-code-generation-repair-v1-preflight-failure/failure-review.json',
        'reports/own-code-delta-v1/complete.json', 'configs/model.json',
        'data/knowledge-eval-v1/questions.jsonl', 'data/public-code-assessment-v1/questions.jsonl',
        'data/public-code-assessment-v1/manifest.json', 'data/alignment-curriculum-v1/images.valid.jsonl',
        'data/alignment-curriculum-v1/contrastive.valid.jsonl', 'data/object-qa-v1/valid.jsonl']
    for name in names:
        files[str(ROOT / name)] = file_hash(ROOT / name)
    for path in [INITIAL, BASE, TOKEN, INITIAL_PLAN, PREVIOUS_QUEUE]:
        files[str(path)] = file_hash(path)
    for shard in range(4):
        for path in [ROOT / 'reports/own-code-delta-public-v1' / f'candidate-{shard}.json',
                     ROOT / 'reports/own-code-delta-public-v1/inputs' / f'shard-{shard}' / 'questions.jsonl',
                     ROOT / 'reports/own-code-delta-public-v1/inputs' / f'shard-{shard}' / 'manifest.json']:
            files[str(path)] = file_hash(path)
    provenance = {}
    for name, folder in FOLDERS.items():
        manifest = json.loads((folder / 'manifest.json').read_text())
        for file, digest in manifest['files_sha256'].items():
            files[str(folder / file)] = digest
        for path in [folder / 'manifest.json', folder / 'admission.json']:
            files[str(path)] = file_hash(path)
        admission = json.loads((folder / 'admission.json').read_text())
        inherited = admission['source_files_sha256'] if name != 'variety' else json.loads(Path(admission['source_admission_report']).read_text())['source_files_sha256']
        for path, digest in inherited.items():
            if path in provenance and provenance[path] != digest:
                raise ValueError('Conflicting immutable upstream source evidence: ' + path)
            provenance[path] = digest
        if name == 'variety':
            p = Path(admission['source_admission_report'])
            provenance[str(p)] = admission['source_admission_report_sha256']
    images = {}
    for name in ['data/alignment-curriculum-v1/images.valid.jsonl', 'data/alignment-curriculum-v1/contrastive.valid.jsonl', 'data/object-qa-v1/valid.jsonl']:
        source = ROOT / name
        for row in map(json.loads, source.read_text().splitlines()):
            image = Path(row['image'])
            if not image.is_absolute():
                image = source.parent / image
            images[str(image.resolve())] = file_hash(image)
    provenance.update(images)
    FileBindings(files).check(full=True)
    FileBindings(provenance).check(full=True)
    import torch
    state = torch.load(INITIAL, map_location='cpu', weights_only=True, mmap=True)
    counts = dict(total_unique=state['total_unique_parameters'], frozen_general=state['base_parameters'],
        trainable_code=state['additional_code_parameters'], reserved_extra_head=state['reserved_point_head_parameters'],
        total_with_reserved_head=state['total_with_reserved_point_head'])
    if counts != dict(total_unique=985664512, frozen_general=706012672, trainable_code=279651840,
                      reserved_extra_head=786432, total_with_reserved_head=986450944):
        raise ValueError('Actual complete initialized conditional-code sub1B inventory required')
    hypothesis = dict(format='own-sub1B-original-generation-repair-with-language-balanced-replay-v1',
        prepared_at=datetime.now(timezone.utc).isoformat(), training_options=options, predeclared_exposure_budget=budget,
        base_checkpoint=str(BASE), base_checkpoint_sha256=file_hash(BASE), initial_component_checkpoint_sha256=file_hash(INITIAL),
        initial_plan_sha256=file_hash(INITIAL_PLAN), own_code_target_sha256=state['own_code_source_checkpoint_sha256'],
        tokenizer_sha256=file_hash(TOKEN), actual_model_counts=counts,
        data_admissions_sha256={name: stream.admission_sha256 for name, stream in data.items()},
        source_files_sha256=files, provenance_files_sha256=provenance, evaluation_image_files_sha256=images,
        gradient_reduction='fixed_parameter_order_worker_average_v1',
        source_guard_scope='Actual packed gradients/runtime checked each update; complete original source/image provenance fully rehashed before acquisition/training and at endpoint.',
        source_qualification='Verified APPS programs pass only their finite original source cases. CodeNet targets are historical Accepted human programs, not uniformly reexecuted.',
        objectives='1.0 generation +0.5 repair +0.25 language-balanced CodeNet; each is separately global response/EOS-token normalized.',
        general_vision_and_critic_frozen=True, no_external_model_or_teacher=True, machine_execution_data_not_genuine_human_preferences=True,
        capacity_before_training='Actual four-L40 mixed source updates,386complete Adam states, byte-exact next-update/component/Adam recovery from a complete collective snapshot, then actual longest4096token original-source backward; discard all probe weights.',
        fixed_evaluations=['Original175supported public-Python cases, original greedy512CUDA/BF16; reuse complete initialized baseline and rerun all candidate shards.',
            '24local Python/JavaScript tasks in both completion/instruction formats, with functional/empirical-efficiency execution.',
            '12local C++ tasks with compilation/function/empirical-efficiency execution.',
            'Original559generated domain questions plus OCR170/object164/caption128 on the unchanged explicit general route at448px.',
            'Complete export/Adam/four-worker RNG/source-draw/target/immutable-general reconciliation.'],
        full_native_math_assessment_still_a_separate_obligation=True, automatic_routing_unproven=True,
        full_prior_source_pass=False, new_production_code_updates=0, candidate_promoted=False, goal_achieved=False)
    REPORTS.mkdir()
    atomic_json(REPORTS / 'hypothesis.json', hypothesis)
    print(json.dumps(dict(hypothesis=str(REPORTS / 'hypothesis.json'), actual_model_counts=counts,
                         exposure_budget=budget, fast_bound_files=len(files), original_provenance_files=len(provenance)), indent=2), flush=True)


if __name__ == '__main__':
    main()
