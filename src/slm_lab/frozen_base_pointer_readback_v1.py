"""Complete original-query head accounting and matched native-image retention."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random

import torch

from .apps_native_metrics_v1 import matched, summarize as native_summary
from .corpus import file_hash
from .diverse_images import atomic_json
from .frozen_base_pointer_stage_v1 import FORMAT, PROJECTIONS
from .point_grounding_readback_v1 import mapping_digest, recount_attention, bootstrap_photo_gain


def reconcile(root, reports, output, data):
    root, reports, output, data = map(Path, [root, reports, output, data])
    plan = json.loads((reports / 'plan.json').read_text()); options = plan['training_options']
    receipt = json.loads((reports / 'training-complete.json').read_text())
    if (receipt['actual_steps'] != 1137 or file_hash(output / 'resume.pt') != receipt['resume_sha256']
            or file_hash(output / 'head.pt') != receipt['head_sha256']
            or file_hash(plan['base_checkpoint']) != plan['base_checkpoint_sha256']):
        raise ValueError('Actual complete reference-only own head and immutable original base required')
    saved = torch.load(output / 'resume.pt', map_location='cpu', weights_only=True)
    export = torch.load(output / 'head.pt', map_location='cpu', weights_only=True)
    base = torch.load(plan['base_checkpoint'], map_location='cpu', weights_only=True, mmap=True)
    base_digest = mapping_digest(base['model'])
    if (saved['format'] != FORMAT or saved['steps'] != 1137 or saved['planned_steps'] != 1137
            or saved['world_size'] != 4 or saved['training_options'] != options
            or saved['sources']['run_plan_sha256'] != file_hash(reports / 'plan.json')
            or saved['sources']['base_checkpoint_sha256'] != plan['base_checkpoint_sha256']
            or saved['frozen_base_digest'] != base_digest or export['frozen_base_digest'] != base_digest
            or receipt['frozen_base_digest_before'] != base_digest or receipt['frozen_base_digest_after'] != base_digest
            or export['sources'] != saved['sources'] or export['actual_steps'] != 1137
            or tuple(saved['projections']) != PROJECTIONS or tuple(export['projections']) != PROJECTIONS
            or sum(t.numel() for t in saved['projections'].values()) != 786432):
        raise ValueError('Entire original backbone reference and both actual trained projections must reconcile')
    for name in PROJECTIONS:
        if not torch.equal(saved['projections'][name], export['projections'][name]) or not bool(torch.isfinite(export['projections'][name]).all()):
            raise ValueError('Both finite exported learned head matrices must equal resume state')
    optimizer = saved['optimizer']; groups = optimizer['param_groups']
    if len(groups) != 2 or groups[0]['params'] or groups[0]['lr'] != 0. or len(groups[1]['params']) != 2 or len(optimizer['state']) != 2:
        raise ValueError('No original backbone tensor may have Adam state')
    for identity, tensor in zip(groups[1]['params'], saved['projections'].values()):
        entry = optimizer['state'][identity]
        if (float(entry['step']) != 1137 or any(entry[k].shape != tensor.shape or not bool(torch.isfinite(entry[k]).all())
                                              for k in ['exp_avg', 'exp_avg_sq'])):
            raise ValueError('Both exact1137step projection Adam states must be complete and finite')
    if len(saved['worker_rng_states']) != 4:
        raise ValueError('All original worker RNG payloads required')
    for rng in saved['worker_rng_states']:
        random.Random().setstate(rng['python']); torch.Generator(device='cpu').set_state(rng['torch'])
        if rng['cuda'].dtype != torch.uint8 or not rng['cuda'].numel():
            raise ValueError('Original actual CUDA RNG bytes must be retained')
    train = list(map(json.loads, (data / 'train.jsonl').read_text().splitlines()))
    order = list(range(len(train))); random.Random(options['seed']).shuffle(order)
    metrics = list(map(json.loads, (reports / 'training-metrics.jsonl').read_text().splitlines()))
    if len(train) != 36354 or len(metrics) != 1137 or metrics != saved['metrics']:
        raise ValueError('Every actual original query and completed update metric required')
    draws = []; samples = 0; tokens = 0
    for step, record in enumerate(metrics):
        point_count = 0; token_count = 0
        if record['step'] != step + 1 or len(record['workers']) != 4 or record['vision_learning_rate'] != 0.:
            raise ValueError('Exact head-only original collective update history required')
        for micro in range(4):
            for rank, worker in enumerate(record['workers']):
                positions = [order[(step * 32 + (micro * 4 + rank) * 2 + i) % len(order)] for i in range(2)]
                actual = worker['micros'][micro]
                points = sum(train[p]['source_point_count'] for p in positions)
                length = sum(len(train[p]['query_input_ids']) for p in positions)
                if (worker['rank'] != rank or actual['positions'] != positions
                        or actual['original_attention_samples'] != points or actual['complete_query_tokens'] != length):
                    raise ValueError('Every original query position/sample/whole-token source draw must reconstruct')
                draws.extend(positions); point_count += points; token_count += length
        if (record['original_attention_samples'] != point_count or record['whole_query_tokens'] != token_count
                or not math.isfinite(record['query_mean_nll'])
                or not math.isclose(record['query_mean_nll'], sum(w['query_mean_nll'] for w in record['workers']) / 4, abs_tol=1e-12)):
            raise ValueError('Original per-query mean objective and all target accounting must match')
        samples += point_count; tokens += token_count
    cursor = dict(completed_steps=1137, population=36354, seed=options['seed'], source_exposures=36384,
        order_sha256=hashlib.sha256(json.dumps(order, separators=(',', ':')).encode()).hexdigest())
    if saved['sampler_state'] != cursor or draws[:36354] != order or draws[36354:] != order[:30]:
        raise ValueError('Exactly one complete original query epoch and30explicit repeats required')
    atomic_json(reports / 'complete-head-source-readback.json', dict(completed_at=datetime.now(timezone.utc).isoformat(),
        actual_steps=1137, original_unique_train_queries=36354, actual_exposures=36384, rounding_repeats=30,
        original_attention_samples=samples, whole_query_tokens=tokens, actual_learned_head_parameters=786432,
        actual_projection_tensors=2, complete_finite_Adam_states=2, original_base_sha256=plan['base_checkpoint_sha256'],
        original_base_canonical_digest=base_digest, entire_original_backbone_reference_unchanged=True,
        exported_projections_equal_complete_resume=True, all_original_draws_and_targets_reconciled=True,
        all_four_worker_RNG_payloads_retained=True, reference_snapshot_bytes=(output / 'resume.pt').stat().st_size,
        head_artifact_bytes=(output / 'head.pt').stat().st_size,
        head_sha256=file_hash(output / 'head.pt'), resume_sha256=file_hash(output / 'resume.pt'), goal_achieved=False))
    held = list(map(json.loads, (data / 'validation.jsonl').read_text().splitlines()))
    photos = {r['image_id']: r for r in map(json.loads, (data / 'photos.jsonl').read_text().splitlines()) if r['split'] == 'validation'}
    pairs = list(map(json.loads, (data / 'validation-control-pairs.jsonl').read_text().splitlines()))
    prior = json.loads((data / 'training-position-prior.json').read_text()); full = {}; mapped = {}
    before_plan = json.loads((reports / 'baseline/grounding/plan.json').read_text())
    after_plan = json.loads((reports / 'candidate/grounding/plan.json').read_text())
    if {k: v for k, v in before_plan.items() if k not in ['created_at', 'actual_model_state_sha256']} != {
            k: v for k, v in after_plan.items() if k not in ['created_at', 'actual_model_state_sha256']}:
        raise ValueError('Exactly matched complete original attention/runtime/control protocol required')
    for phase in ['baseline', 'candidate']:
        journal = reports / phase / 'grounding/observations.jsonl'
        observed = list(map(json.loads, journal.read_text().splitlines()))
        full[phase], mapped[phase] = recount_attention(observed, held, photos, pairs, prior)
        certificate = json.loads((journal.parent / 'complete.json').read_text())
        if certificate['observations_sha256'] != file_hash(journal) or certificate['conditions'] != full[phase]['conditions']:
            raise ValueError('Every original raw query/pixel/target/control prediction must reconstruct')
    final_map = {**{'base.' + name: tensor for name, tensor in base['model'].items()}, **export['projections']}
    capacity = json.loads((reports / 'capacity.json').read_text())
    if (before_plan['actual_model_state_sha256'] != capacity['initial_head_state_digest']
            or after_plan['actual_model_state_sha256'] != mapping_digest(final_map)):
        raise ValueError('Actual capacity/baseline/final original frozen-base and learned-head states must match')
    gains = {condition: bootstrap_photo_gain(mapped['candidate']['proper'], mapped['candidate'][condition])
             for condition in ['blank_image', 'wrong_image', 'uniform', 'center', 'training_position', 'wrong_query']}
    gains['before_to_after_proper'] = bootstrap_photo_gain(mapped['candidate']['proper'], mapped['baseline']['proper'])
    native = {}
    for name in ['ocr.json', 'objects.json', 'vision.json']:
        before, after = [json.loads((reports / phase / name).read_text()) for phase in ['baseline', 'candidate']]
        matched(before, after, name)
        key = 'samples' if name == 'vision.json' else 'results'
        if before[key] != after[key] or native_summary(before, name) != native_summary(after, name):
            raise ValueError('The frozen original native image view must retain every matched raw output and score')
        native[name] = dict(baseline=native_summary(before, name), candidate=native_summary(after, name),
                            every_retained_raw_native_output_and_metric_equal=True)
    atomic_json(reports / 'matched-attention-and-native-comparison.json', dict(completed_at=datetime.now(timezone.utc).isoformat(),
        attention=full, paired_original_photo_gain_intervals=gains, matched_native_image_assessments=native,
        every_raw_original_query_pixel_target_control_and_score_reconstructed=True,
        every_original_native_image_output_retained=True, learned_attention_not_object_box_OCR_or_general_visual_reasoning=True,
        development_validation_not_certified_independent_final_test=True, automatic_promotion=False, goal_achieved=False))
