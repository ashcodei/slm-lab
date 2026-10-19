"""Actual whole-budget capacity, private replay and independent actor cold.

This internal controller owns no GPUlease and never starts a production run.
A caller must admit resources and a fixed complete own initializer separately.
Independent cold loads the own base plus the committed actor parts; final
standalone packaging/export and full assessments remain separate requirements.
"""
import copy
import random
import time

import numpy as np
import torch

from .batched_reasoning_capacity_cases_v1 import choose
from .codenet_checkpoint_v1 import capture_rng
from .own_dense_actor_v1 import OwnDenseActor
from .parallel_batched_reasoning_stage_v2 import ParallelBatchedReasoningStage
from .parallel_function_repair_stage_v1 import manifest, value_digest
from .parallel_repair_durable_state_v1 import commit, read_latest, restore_latest
from .whole_human_sft_evidence_v1 import tensor_digest

FORMAT = 'own-batched-reasoning-whole-capacity-private-replay-actor-cold-v1'


def live(stage):
    return dict(actor=tensor_digest(stage.model.parts.named_parameters()),
        Adam=value_digest(stage.optimizer.state_dict()), RNG=value_digest(capture_rng(stage.device)),
        metrics=value_digest(stage.metrics), completed=stage.completed, failed=stage.failed,
        selected_tensor_ids=[(id(p), p.data_ptr(), p._version) for p in stage.parameters],
        Adam_tensor_ids=[(id(p), k, id(t), t.data_ptr(), t._version)
            for p, row in stage.optimizer.state.items() for k, t in row.items()])


def draws(device):
    result = dict(python=random.random(), numpy=float(np.random.random()), CPU=torch.rand(4).tolist())
    if device.type == 'cuda': result['CUDA'] = torch.rand(4, device=device).tolist()
    return result


def sync(stage):
    if stage.device.type == 'cuda': torch.cuda.synchronize(stage.device)


def _cold(stage, independent_actor, saved, inputs):
    error = None
    other = report = None
    before_rng = value_digest(capture_rng(stage.device))
    try:
        if (not callable(independent_actor) or set(inputs) != {'text', 'photo', 'word'}
                or set(inputs['text']) != {'input_ids', 'attention_mask'}
                or set(inputs['photo']) != {'input_ids', 'attention_mask', 'pixel_values'}
                or set(inputs['word']) != {'word_pixels'}):
            raise ValueError('Independent own actor and complete text/photo/word inputs are required')
        other = independent_actor()
        if (type(other) is not OwnDenseActor or other.counts() != stage.model.counts()
                or other.settings != stage.model.settings or other.config != stage.model.config
                or any(p.device != stage.device or p.dtype != torch.float32 or p.grad is not None for p in other.parameters())
                or {p.data_ptr() for p in other.parameters()} & {p.data_ptr() for p in stage.model.parameters()}
                or value_digest(capture_rng(stage.device)) != before_rng):
            raise ValueError('A separately loaded identical own base/actor graph must preserve all private RNGs and share no storage')
        for name in ['text', 'photo']:
            ids, mask = inputs[name]['input_ids'], inputs[name]['attention_mask']
            if (ids.ndim != 2 or ids.shape[0] != 1 or ids.dtype != torch.long or mask.dtype != torch.bool
                    or ids.shape != mask.shape or not bool(mask.all()) or ids[0, 0] != 1
                    or not bool(((0 <= ids) & (ids < other.config.vocab_size)).all())):
                raise ValueError('Every actual complete cold native text position/mask must be retained')
        photo, word = inputs['photo']['pixel_values'], inputs['word']['word_pixels']
        if (photo.ndim != 4 or photo.shape[:2] != (1, 3) or photo.shape[2] != photo.shape[3]
                or word.ndim != 4 or word.shape[:3] != (1, 3, 64)
                or any(t.dtype != torch.float32 or not bool(torch.isfinite(t).all()) for t in [photo, word])
                or stage.mode == 'capacity' and (inputs['text']['input_ids'].shape != (1, 4096)
                    or photo.shape != (1, 3, 448, 448) or word.shape != (1, 3, 64, 448))):
            raise ValueError('Actual full-size capacity retains4096text/448RGB/64x448whole-word inputs')
        other.import_export(saved['model'], expected_sources=stage.sources)
        hot = dict(stage.model.named_parameters())
        if (list(hot) != [n for n, _ in other.named_parameters()]
                or any(not torch.equal(hot[n], p) for n, p in other.named_parameters())):
            raise ValueError('Every original base/head/policy/critic value must exactly match independent cold recovery')
        stage.model.eval()
        other.eval()
        batches = {n: {k: v.to(stage.device) for k, v in b.items()} for n, b in inputs.items()}
        output_digests = {}
        with torch.no_grad(), torch.autocast(stage.device.type, enabled=False):
            for name in ['text', 'photo']:
                wanted = stage.model(**batches[name])['logits']
                actual = other(**batches[name])['logits']
                if (wanted.dtype != torch.float32 or not bool(torch.isfinite(wanted).all())
                        or not torch.equal(wanted, actual)):
                    raise ValueError('Every complete FP32text/photo actor output must cold-load bit exactly')
                output_digests[name] = tensor_digest([('logits', actual)])
                del wanted, actual
            wanted = stage.model.base.word_logits(batches['word']['word_pixels'])
            actual = other.base.word_logits(batches['word']['word_pixels'])
            if wanted.dtype != torch.float32 or not bool(torch.isfinite(wanted).all()) or not torch.equal(wanted, actual):
                raise ValueError('The complete original FP32whole-word readout must remain cold-load bit exact')
            output_digests['word'] = tensor_digest([('logits', actual)])
        if value_digest(capture_rng(stage.device)) != before_rng:
            raise ValueError('All actual independent cold loads and complete forward probes must preserve private RNGs')
        report = dict(complete=True, mode=stage.mode, mechanics_only=stage.mode == 'mechanics',
            actual_text_positions=inputs['text']['input_ids'].shape[1], actual_photo_side=photo.shape[-1],
            actual_word_shape=list(word.shape[-2:]), actual_word_readout_dtype='torch.float32',
            every_base_head_actor_critic_parameter_equal=True, no_shared_parameter_storage=True,
            complete_text_photo_word_outputs_bit_exact=True, all_private_RNGs_unchanged=True,
            actual_complete_output_sha256=output_digests, original_own_base_plus_committed_actor_parts_loaded=True,
            final_standalone_export_and_final_inference_still_required=True)
    except Exception as exc:
        error = repr(exc)
    stage.coordinated(error, 'independent complete actor cold')
    del other
    return report


def run_capacity(stage, directory, *, capacity_plan, independent_actor, cold_inputs,
                 maximum_bytes, minimum_free_bytes, allow_mechanics=False):
    error = None
    try:
        if (type(stage) is not ParallelBatchedReasoningStage or stage.completed or stage.failed or stage.world != 4
                or len(stage.entries) < 2 or stage.entries != capacity_plan['capacity_entries']
                or not callable(independent_actor)
                or stage.mode != 'capacity' and not (allow_mechanics is True and stage.mode == 'mechanics')):
            raise ValueError('A fresh actual four-replica complete-capacity stage is required')
        if stage.mode == 'capacity':
            if capacity_plan != choose(stage.data, seed=stage.options['seed'], epochs=stage.options['epochs']):
                raise ValueError('Every complete original worst-source batch and all4rank rotations must match')
        elif capacity_plan.get('mechanics_only') is not True:
            raise ValueError('Own-small controller fixtures must be explicitly mechanics-only')
        stage.intact()
        common = dict(plan=value_digest(capacity_plan), binding=stage.initial_binding)
    except Exception as exc:
        error = repr(exc)
    stage.coordinated(error, 'complete capacity preflight')
    if any(v != common for v in stage.gather(common)):
        raise ValueError('Every actual replica must run the identical whole capacity workload')
    if stage.device.type == 'cuda': torch.cuda.reset_peak_memory_stats(stage.device)
    timings, physical_calls = [], 0
    def step():
        nonlocal physical_calls
        sync(stage)
        began = time.monotonic()
        value = stage.update()
        sync(stage)
        physical_calls += 1
        timings.append(dict(actual_global_step=stage.completed, actual_seconds=time.monotonic() - began,
            actual_generated_actions=sum(t['kernel']['actual_sampled_action_tokens'] for t in value['actual_rank_observations'] if t['kernel'] is not None)))
        return value
    def save(): return commit(stage, directory, maximum_bytes=maximum_bytes, minimum_free_bytes=minimum_free_bytes)
    step()
    first = save()
    error = None
    try:
        saved, receipt, pointer = read_latest(directory)
        if first != receipt: raise ValueError('The first complete shared file and all private worker records must reconcile')
    except Exception as exc:
        error = repr(exc)
    stage.coordinated(error, 'actual first durable readback')
    wanted_draws = draws(stage.device)
    wanted = step()
    expected = live(stage)
    # Malformed private generators remain invalid even under a new checksum.
    error = None
    try:
        bad = copy.deepcopy(saved)
        bad['private_workers'][3]['rng']['torch'] = torch.zeros(1, dtype=torch.uint8)
    except Exception as exc:
        error = repr(exc)
    stage.coordinated(error, 'invalid peer RNG fixture before recovery')
    rejected = False
    try: stage.resume(bad, expected_manifest=manifest(bad))
    except ValueError: rejected = True
    stage.coordinated(None if rejected and live(stage) == expected else 'Malformed peer RNG changed a live value', 'invalid peer RNG rejection proof')
    if stage.device.type == 'cuda':
        bad = copy.deepcopy(saved)
        bad['private_workers'][3]['rng']['cuda'] = torch.zeros(1, dtype=torch.uint8)
        rejected = False
        try: stage.resume(bad, expected_manifest=manifest(bad))
        except ValueError: rejected = True
        stage.coordinated(None if rejected and live(stage) == expected else 'Malformed peer CUDA RNG changed a live value', 'invalid peer CUDA RNG rejection proof')
    bad = copy.deepcopy(saved)
    if stage.rank == 2: bad['private_workers'][0]['rng'] = copy.deepcopy(saved['private_workers'][1]['rng'])
    rejected = False
    try: stage.resume(bad, expected_manifest=receipt['semantic_manifest'])
    except ValueError: rejected = True
    stage.coordinated(None if rejected and live(stage) == expected else 'Valid foreign private RNG changed a live value', 'valid foreign RNG rejection proof')
    restore_latest(stage, directory, expected_pointer_sha256=pointer)
    actual_draws = draws(stage.device)
    actual = step()
    current = live(stage)
    error = None
    if (actual_draws != wanted_draws or actual != wanted
            or any(current[k] != expected[k] for k in ['actor', 'Adam', 'RNG', 'metrics', 'completed', 'failed'])):
        error = 'Actual next private draws/actions/verifiers/logps/unclipped gradients/weights/Adam/RNG differ'
    stage.coordinated(error, 'exact next global source group replay')
    while stage.completed < len(stage.entries): step()
    final = save()
    error = None
    try:
        last, last_receipt, final_pointer = read_latest(directory)
        if (last_receipt != final or last['completed'] != len(stage.entries)
                or value_digest(last['metrics']) != value_digest(stage.metrics)
                or tensor_digest(last['model']['parts'].items()) != tensor_digest(stage.model.parts.named_parameters())):
            raise ValueError('Every final actual capacity source and selected value must match the committed file')
    except Exception as exc:
        error = repr(exc)
    stage.coordinated(error, 'complete final capacity readback')
    cold = _cold(stage, independent_actor, last, cold_inputs)
    stage.intact()
    actual_rows = [a['kernel'] for r in stage.metrics for a in r['actual_rank_observations'] if a['kernel'] is not None]
    samplers = [r['complete_batched_sampling_observation'] for r in actual_rows]
    workers = stage.gather(dict(actual_rank=stage.rank, private_RNG_sha256=value_digest(capture_rng(stage.device)),
        actor_sha256=tensor_digest(stage.model.parts.named_parameters()), Adam_sha256=value_digest(stage.optimizer.state_dict()),
        actual_intervals=timings))
    if (len({v['private_RNG_sha256'] for v in workers}) != 4 or len({v['actor_sha256'] for v in workers}) != 1
            or len({v['Adam_sha256'] for v in workers}) != 1):
        raise ValueError('All4actual workers must retain distinct private RNGs and one identical model/Adam')
    result = dict(format=FORMAT, complete=True, passed=True, rank=stage.rank, world=4, mode=stage.mode,
        actual_device_type=stage.device.type, actual_device_index=stage.device.index,
        actual_research_actor_parameters=stage.model.counts()['actual_unique'],
        actual_native_parameters=stage.model.base.counts()['total_unique'], profile=stage.profile,
        original_source_bindings=copy.deepcopy(stage.sources), original_objective_options=copy.deepcopy(stage.options),
        original_own_reference_binding=copy.deepcopy(stage.reference_binding), capacity_plan_sha256=value_digest(capacity_plan),
        actual_complete_capacity_updates=stage.completed, actual_physical_optimizer_calls=physical_calls,
        actual_source_budget=stage.data.budget(stage.entries), actual_Adam_states=len(stage.optimizer.state),
        actual_complete_capacity_source_positions=copy.deepcopy(stage.entries),
        actual_complete_source_families=sorted({r['complete_original_source']['task'] for r in actual_rows}),
        actual_committed_prefix_trajectory_count=sum(len(r['trajectories']) for r in actual_rows),
        actual_committed_prefix_generated_action_count=sum(r['actual_sampled_action_tokens'] for r in actual_rows),
        actual_all_physical_global_generated_actions_including_replay=sum(t['actual_generated_actions'] for t in timings),
        actual_genuine_EOS_stops=sum(t['actual_stopped_on_EOS'] for r in actual_rows for t in r['trajectories']),
        actual_reward1_trajectories=sum(x == 1. for r in actual_rows for x in r['actual_strict_rewards']),
        actual_reward_varying_groups=sum(len(set(r['actual_strict_rewards'])) > 1 for r in actual_rows),
        actual_maximum_cached_full_logp_difference=max(s['cached_full_max_absolute_difference'] for s in samplers),
        actual_peak_logical_KV_bytes=max(s['actual_peak_logical_KVcache_bytes'] for s in samplers),
        actual_conservative_full_horizon_logical_KV_bytes=max(s['conservative_full_horizon_logical_KVcache_bytes'] for s in samplers),
        actual_per_rank_intervals=workers, all_actual_next_actions_verifiers_unclipped_gradients_Adam_private_RNG_exact=True,
        invalid_peer_CPU_RNG_rejected_before_mutation=True,
        invalid_peer_CUDA_RNG_rejected_before_mutation=stage.device.type == 'cuda',
        peer_only_valid_foreign_RNG_rejected_before_mutation=True, independent_complete_actor_cold=cold,
        final_actual_shared_checkpoint_sha256=final['checkpoint_sha256'], final_actual_pointer_sha256=final_pointer,
        actual_GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(stage.device) if stage.device.type == 'cuda' else None,
        actual_GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved(stage.device) if stage.device.type == 'cuda' else None,
        full_size_GPU_capacity_passed=stage.mode == 'capacity', additional_learned_values=0,
        production_updates=0, new_model_quality_gain=False, final_standalone_and_full_assessments_still_required=True,
        candidate_promoted=False, goal_achieved=False)
    return result


def require_full_capacity_reports(proofs, *, capacity_plan, profile, expected_sources, expected_options, expected_reference):
    """Strict report completeness; source-state readback/ownership remain gates."""
    if (not isinstance(proofs, list) or len(proofs) != 4 or profile not in ['anchor_CE', 'anchor_RL']
            or capacity_plan.get('mechanics_only') is not False or capacity_plan.get('source_selection_only') is not True
            or capacity_plan.get('actual_original_train_source_population') != 2568
            or len(capacity_plan.get('actual_train_task_families', [])) != 12
            or capacity_plan.get('full_original_action_budgets') != [64, 256]
            or capacity_plan.get('all_four_replicas_receive_each_selected_source') is not True
            or not expected_sources or not expected_reference or expected_options.get('CPU_mechanics_horizon') is not None):
        raise ValueError('All4actual full-source, full-horizon and exact-initializer capacity reports are required')
    for rank, report in enumerate(proofs):
        if (report.get('format') != FORMAT or report.get('complete') is not True or report.get('passed') is not True
                or report.get('rank') != rank or report.get('world') != 4 or report.get('mode') != 'capacity'
                or report.get('actual_device_type') != 'cuda' or report.get('actual_device_index') != rank
                or report.get('actual_research_actor_parameters') != 999706442 or report.get('actual_native_parameters') != 984399690
                or report.get('profile') != profile or report.get('original_source_bindings') != expected_sources
                or report.get('original_objective_options') != expected_options or report.get('original_own_reference_binding') != expected_reference
                or report.get('capacity_plan_sha256') != value_digest(capacity_plan)
                or report.get('actual_complete_capacity_source_positions') != capacity_plan['capacity_entries']
                or report.get('actual_source_budget') != capacity_plan['capacity_budget']
                or report.get('actual_complete_source_families') != capacity_plan['actual_train_task_families']
                or report.get('actual_complete_capacity_updates') != len(capacity_plan['capacity_entries'])
                or report.get('actual_physical_optimizer_calls') != len(capacity_plan['capacity_entries']) + 1
                or report.get('actual_Adam_states') != 561
                or report.get('actual_committed_prefix_trajectory_count') != capacity_plan['capacity_budget']['source_presentations'] * expected_options['group_size']
                or report.get('full_size_GPU_capacity_passed') is not True or report.get('production_updates') != 0
                or not all(report.get(k) is True for k in ['all_actual_next_actions_verifiers_unclipped_gradients_Adam_private_RNG_exact',
                    'invalid_peer_CPU_RNG_rejected_before_mutation', 'invalid_peer_CUDA_RNG_rejected_before_mutation',
                    'peer_only_valid_foreign_RNG_rejected_before_mutation'])
                or not 0 <= report.get('actual_maximum_cached_full_logp_difference', float('inf')) <= expected_options['cache_absolute_bound']
                or not 0 < report.get('actual_GPU_peak_allocated_bytes', 0) <= report.get('actual_GPU_peak_reserved_bytes', 0)):
            raise ValueError('Every actual full-size GPU, complete workload, ordinary Adam and private recovery report must agree')
        cold = report.get('independent_complete_actor_cold', {})
        if (cold.get('complete') is not True or cold.get('mode') != 'capacity' or cold.get('mechanics_only') is not False
                or cold.get('actual_text_positions') != 4096 or cold.get('actual_photo_side') != 448
                or cold.get('actual_word_shape') != [64, 448] or cold.get('actual_word_readout_dtype') != 'torch.float32'
                or not all(cold.get(k) is True for k in ['every_base_head_actor_critic_parameter_equal', 'no_shared_parameter_storage',
                    'complete_text_photo_word_outputs_bit_exact', 'all_private_RNGs_unchanged'])):
            raise ValueError('Every actual full-size original text/photo/word and independent actor value must cold-load exactly')
    if len({p['final_actual_shared_checkpoint_sha256'] for p in proofs}) != 1:
        raise ValueError('All4reports must bind the same complete shared capacity checkpoint')
    return dict(all_four_complete_full_size_capacity_report_schemas_match=True,
        independent_complete_capacity_state_readback_and_owned_training_admission_still_required=True,
        production_training_admitted=False, goal_achieved=False)
