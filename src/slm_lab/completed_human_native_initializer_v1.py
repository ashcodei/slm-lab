"""Read completed human-policy states and build one independent own native.

This is CPU source reconstruction, not optimizer/RNG resume or GPU admission.
Retained CUDA generator bytes are hashed without interpreting them on a GPU.
Original training data and neural preference likelihoods are not rerun here.
"""
import copy
from pathlib import Path
import shutil

import torch

from .codenet_checkpoint_v1 import atomic_checkpoint, capture_rng
from .code_residual_checkpoint_v1 import _validate_adam, _validate_rng
from .corpus import file_hash
from .joint_native_fold_v1 import fold
from .own_dense_actor_v1 import OwnDenseActor
from .own_dense_standalone_v1 import export as native_export, load
from .parallel_function_repair_stage_v1 import value_digest
from .shared_joint_initializer_v1 import assemble
from .whole_human_sft_evidence_v1 import tensor_digest

FORMAT = 'own-completed-human-DPO-and-own-foundation-joint-initializer-v1'


def verify_private_states(actor, states, *, options, population, sources, allow_mechanics=False):
    full = (type(actor) is OwnDenseActor and actor.counts()['actual_unique'] == 999706442
        and actor.base.counts()['total_unique'] == 984399690 and actor.settings['rank'] == 16
        and options['steps'] == 5362 and population == 85786)
    small = (type(actor) is OwnDenseActor and actor.counts()['actual_unique'] < 10**7 and allow_mechanics is True)
    if (not full and not small or actor.training or torch.cuda.is_initialized()
            or any(p.device.type != 'cpu' or p.dtype != torch.float32 or p.grad is not None for p in actor.parameters())
            or options.get('epochs') != 1 or options.get('DPO_weight') != 1):
        raise ValueError('A completed own CPU human-DPO actor or explicit small control is required')
    before = (tensor_digest(actor.named_parameters()), value_digest(capture_rng('cpu')))
    parts = tensor_digest(actor.parts.named_parameters()); selected = actor.named_selected(('actor',))
    frozen = actor.export(sources)['frozen_dense_sha256']
    steps = options['steps']; common = set(); private = set(); ranks = []; observations = []
    for rank, state in enumerate(states):
        outer = state.get('core', {}); core = outer.get('core', {})
        cursor = dict(completed_updates=steps, source_population=population, complete_passes=1,
            next_pass_sources=0, actual_pair_draws=population, repeated_rounding_pairs=0,
            seed=options['seed'], world_size=4, replica_rank=rank)
        if (rank >= 4 or state.get('format') != 'own-private-replica-complete-human-DPO-global-normalization-v2'
                or state.get('replicas') != 4 or state.get('replica_rank') != rank
                or state.get('private_state_is_not_whole_distributed_checkpoint') is not True
                or state.get('goal_achieved') is not False
                or outer.get('format') != 'own-dense-complete-human-DPO-epoch-Adam-RNG-v1'
                or outer.get('source_population') != population or outer.get('options') != options
                or outer.get('frozen_dense_sha256') != frozen
                or core.get('format') != 'own-dense-policy-critic-selected-Adam-RNG-source-cursor-v1'
                or core.get('completed') != steps or core.get('planned') != steps or core.get('world_size') != 1
                or core.get('groups') != ['actor'] or core.get('optimized_names') != [n for n, _ in selected]
                or core.get('cursor') != cursor or len(core.get('metrics', [])) != steps):
            raise ValueError('All four original terminal human states, complete cursor and selected inventory must match')
        actor.validate_export(core['model'], expected_sources=sources)
        if tensor_digest(core['model']['parts'].items()) != parts:
            raise ValueError('The completed exported actor must equal every actual final private actor, including its unused critic')
        _validate_adam(core['optimizer'], [p for _, p in selected], steps)
        group = core['optimizer']['param_groups'][0]
        if (group['lr'] != options['learning_rate'] or list(group['betas']) != list(options['betas'])
                or group['weight_decay'] != options['weight_decay'] or group['eps'] != 1e-8):
            raise ValueError('The original complete ordinary human AdamW settings must remain')
        positions = []
        for step, row in enumerate(core['metrics'], 1):
            ps = row['original_source_positions']; pairs = row['original_human_pairs']
            if (row.get('step') != step or not ps or len(pairs) != len(ps)
                    or [p['original_train_position'] for p in pairs] != ps
                    or row.get('DPO_weight') != 1 or row.get('reference_queried') is not True
                    or row.get('global_pair_and_chosen_token_denominators') is not True
                    or row.get('per_replica_original_pairs') != [len(ps[r::4]) for r in range(4)]
                    or row.get('gradient_reduction') != 'SUM_without_second_world_average'):
                raise ValueError('Every original full human-pair record and actual empty-tail denominator must remain')
            positions.extend(ps)
        if len(positions) != population or set(positions) != set(range(population)):
            raise ValueError('Every original human source must occur exactly once in the completed state')
        rng = core['rng']; _validate_rng(dict(rng, cuda=None), 'cpu')
        if full and (not isinstance(rng['cuda'], torch.Tensor) or rng['cuda'].dtype != torch.uint8
                or rng['cuda'].device.type != 'cpu' or rng['cuda'].ndim != 1 or not rng['cuda'].numel()):
            raise ValueError('Retain every original actual CUDA generator byte without initializing CUDA')
        common_digest = value_digest(dict(parts=parts, Adam=core['optimizer'], metrics=core['metrics']))
        common.add(common_digest); private.add(value_digest(rng)); ranks.append(rank)
        observations.append(dict(rank=rank, actual_completed_updates=steps, actual_human_pair_presentations=len(positions),
            actual_Adam_states=len(core['optimizer']['state']), actual_private_RNG_sha256=value_digest(rng),
            actual_common_actor_Adam_metrics_sha256=common_digest))
    if ranks != [0, 1, 2, 3] or len(common) != 1 or len(private) != 4:
        raise ValueError('All four actual final actors/Adam/metrics must agree and each private generator must remain distinct')
    if before != (tensor_digest(actor.named_parameters()), value_digest(capture_rng('cpu'))):
        raise ValueError('Completed-state inspection cannot change any source value or caller RNG')
    return dict(all_four_final_private_states_match_exported_actor=True, actual_rank_observations=observations,
        actual_common_actor_Adam_metrics_sha256=next(iter(common)), original_pair_population=population,
        actual_completed_human_updates=steps, source_and_CPU_RNG_unchanged=True, new_Adam_calls=0,
        CUDA_generator_bytes_retained_but_not_replayed=True, original_training_corpus_or_neural_likelihoods_not_rerun=True,
        actual_full_size_completed_human_state_readback=full, GPU_training_admitted=False)


def materialize(actor, folder, tokenizer_path, *, sources, verified, probes, word_pixels,
                working_budget_bytes, maximum_checkpoint_bytes, minimum_free_bytes, preflight, allow_mechanics=False):
    folder = Path(folder)
    if (folder.exists() or not probes or verified.get('all_four_final_private_states_match_exported_actor') is not True
            or verified.get('GPU_training_admitted') is not False or minimum_free_bytes < 40 * 1024**3
            or not (verified.get('actual_full_size_completed_human_state_readback') is True or allow_mechanics is True)):
        raise ValueError('A fresh funded native output and actual completed human-state readback are required')
    preflight(); before = value_digest(capture_rng('cpu'))
    text, folding = fold(actor, sources=sources, working_budget_bytes=working_budget_bytes)
    native, assembly = assemble(text, actor.base, sources=sources, working_budget_bytes=working_budget_bytes,
        allow_mechanics=allow_mechanics)
    del text
    observations = []
    for name, batch in probes:
        preflight(); args = {k: batch[k] for k in ['input_ids', 'attention_mask', 'pixel_values'] if k in batch}
        with torch.no_grad():
            hot = actor(**args)['logits']; output = native(**args)['logits']
        maximum = float((hot - output).abs().max())
        if not bool(torch.isfinite(hot).all()) or not bool(torch.isfinite(output).all()) or maximum > .002:
            raise ValueError('Actual complete source probes must satisfy the declared FP32 folding bound')
        observations.append(dict(name=name, complete_text_positions=batch['input_ids'].shape[1],
            actual_pixel_shape=list(batch['pixel_values'].shape) if 'pixel_values' in batch else None,
            maximum_FP32_actor_to_initializer_error=maximum, actual_argmax_differences=int((hot.argmax(-1) != output.argmax(-1)).sum())))
        del hot, output
    with torch.no_grad():
        word = native.word_logits(word_pixels)
        if not torch.equal(actor.base.word_logits(word_pixels), word) or word.dtype != torch.float32:
            raise ValueError('Every actual native FP32 whole-word output must survive assembly')
    folder.mkdir(); shutil.copyfile(tokenizer_path, folder / 'tokenizer.json'); path = folder / 'model.pt'
    provenance = dict(kind=FORMAT, fixed_text_source='completed own human_DPO; not selected by future joint quality',
        fixed_visual_source='that same completed own public-pretraining foundation',
        completed_human_updates=verified['actual_completed_human_updates'], joint_optimizer_updates=0,
        completed_human_state_readback=copy.deepcopy(verified), actual_assembly=assembly, actual_language_fold=folding)
    preflight()
    atomic_checkpoint(path, native_export(native, sources=sources, provenance=provenance, completed_updates=0,
        tokenizer_sha256=file_hash(tokenizer_path)), maximum_checkpoint_bytes, minimum_free_bytes)
    cold, metadata = load(path, expected_checkpoint_sha256=file_hash(path), expected_sources=sources)
    if tensor_digest(cold.named_parameters()) != tensor_digest(native.named_parameters()):
        raise ValueError('Every complete initializer value must independently cold-load exactly')
    for name, batch in probes:
        preflight(); args = {k: batch[k] for k in ['input_ids', 'attention_mask', 'pixel_values'] if k in batch}
        with torch.no_grad():
            if not torch.equal(native(**args)['logits'], cold(**args)['logits']):
                raise ValueError('Every actual complete native text/photo probe must cold-load bit-exactly')
    with torch.no_grad():
        if not torch.equal(word, cold.word_logits(word_pixels)):
            raise ValueError('The actual complete FP32 whole-word output must cold-load exactly')
    preflight()
    if before != value_digest(capture_rng('cpu')) or torch.cuda.is_initialized():
        raise ValueError('Initializer export cannot advance private CPU generators or initialize CUDA')
    return dict(format=FORMAT, complete=True, checkpoint=str(path), checkpoint_sha256=file_hash(path),
        tokenizer_sha256=file_hash(folder / 'tokenizer.json'), sources=copy.deepcopy(sources), actual_counts=metadata['actual_counts'],
        actual_assembly=assembly, actual_language_fold=folding, completed_human_state_readback=verified,
        actual_FP32_source_probes=observations, actual_word_pixel_shape=list(word_pixels.shape), actual_word_output_shape=list(word.shape),
        actual_word_dtype=str(word.dtype), all_native_values_and_actual_text_photo_word_cold_outputs_exact=True,
        inference_needs_only_native_checkpoint_and_packaged_tokenizer=True,
        actual_full_size_CPU_initializer=assembly['actual_full_size_value_assembly'], joint_optimizer_updates=0,
        source_CPU_RNG_unchanged=True, GPU_contexts=0, full_size_GPU_capacity_or_training_admitted=False,
        full_original_baseline_and_final_assessments_still_required=True, candidate_promoted=False, goal_achieved=False)
