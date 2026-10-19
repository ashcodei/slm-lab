"""Synthetic full-group native KV/CE/actor-gradient allocation without updates.

Fixed non-EOS actions keep every cache row active through the entire original
horizon. They are explicitly not sampled policy actions, execution rewards,
training targets or capability evidence. Callers coordinate failures across
ranks and own resource admission; this helper grants neither GPU nor training.
"""
import copy
from pathlib import Path
from .corpus import file_hash

import torch

from .anchored_code_objective_v1 import reference_anchor
from .batched_own_native_rollouts_v2 import signature
from .codenet_checkpoint_v1 import capture_rng
from .native_suffix_batched_history_stage_v5 import NativeSuffixBatchedHistoryStageV5
from .parallel_function_repair_stage_v1 import value_digest

FORMAT = 'own-native-code-indexed-fixed-full-group-horizon-capacity-no-update-v2'


def stress(stage, position, *, allow_mechanics=False):
    if (type(stage) is not NativeSuffixBatchedHistoryStageV5
            or stage.mode != 'capacity' and not (allow_mechanics is True and stage.mode == 'mechanics')
            or type(position) is not int or position not in {p for e in stage.entries for p in e}
            or stage.failed or stage.sources.get(str(Path(__file__).resolve())) != file_hash(Path(__file__).resolve())):
        raise ValueError('An actual selected native capacity source or explicit own-small fixture is required')
    stage.intact()
    info = stage.data.information(position)
    if not info['native_execution_eligible']:
        raise ValueError('Synthetic native group stress requires a genuinely eligible view; CE-only views retain their anchor separately')
    model, device = stage.model, stage.device
    flags = [(m, m.training) for m in model.modules()]
    checkpointing = model.policy.gradient_checkpointing
    before = dict(parameters=signature(model), Adam=value_digest(stage.optimizer.state_dict()),
        RNG=value_digest(capture_rng(device)), completed=stage.completed, metrics=value_digest(stage.metrics))
    horizon = stage.options['CPU_mechanics_horizon'] or info['full_native_action_budget']
    group, vocab = stage.options['group_size'], model.config.vocab_size
    if vocab <= 4 or horizon < 1: raise ValueError('A full native action horizon and non-control vocabulary are required')
    actions = [[4 + (i + row) % (vocab - 4) for i in range(horizon)] for row in range(group)]
    result = None
    try:
        with stage.data.episode(position) as episode, episode.oracle.checked_source_batch(), torch.autocast(device.type, enabled=False):
            prepared = episode.rollout_view()
            prefix = prepared['native_rollout_prefix_ids']
            expected_KV = (2 * group * model.config.num_layers * model.config.num_kv_heads
                * (model.config.hidden_size // model.config.num_heads) * (len(prefix) + horizon - 1) * 4)
            if len(prefix) + horizon > model.config.max_text_length or expected_KV > stage.options['maximum_cache_bytes']:
                raise ValueError('Every original prefix/full horizon/full group must fit the declared cache without shortening')
            model.eval()
            with torch.no_grad():
                state = model.policy.prefill(torch.tensor([prefix] * group, dtype=torch.long, device=device))
                cached, actual_peak = [[] for _ in range(group)], 0
                for step in range(horizon):
                    logits = state['logits'][:, -1].float()
                    if logits.shape != (group, vocab) or not bool(torch.isfinite(logits).all()):
                        raise ValueError('All full-group fixed-action histories need finite full-vocabulary logits')
                    actual_peak = max(actual_peak, state['cache'].bytes)
                    choices = torch.tensor([a[step] for a in actions], dtype=torch.long, device=device)
                    values = logits.log_softmax(-1).gather(1, choices[:, None]).flatten().tolist()
                    for row, value in zip(cached, values): row.append(value)
                    if step + 1 < horizon:
                        state = model.policy.decode(choices[:, None], state['cache'])
                if actual_peak != expected_KV:
                    raise ValueError('The actual live cache must physically reach every full-batch/full-history element')
                del state, logits, choices, values
            # Preserve the real training memory pattern: cached sampling ends,
            # then whole original CE, then sequential policy/reference passes.
            model.train(); model.policy.gradient_checkpointing = True
            batch = episode.batch(device); labels = batch.pop('labels')
            logits = model(**batch)['logits']
            loss, anchor = reference_anchor(logits, labels, info['original_response_EOS_targets'], weight=stage.options['anchor_weight'])
            loss.backward(inputs=stage.parameters)
            del batch, labels, logits, loss
            maximum_difference = 0.
            fixed_losses = []
            for emitted, logps in zip(actions, cached):
                ids = torch.tensor([prefix + emitted], dtype=torch.long, device=device)
                selected = torch.tensor(emitted, dtype=torch.long, device=device)[:, None]
                logits = model(ids)['logits'][0, len(prefix)-1:-1].float()
                logp = logits.log_softmax(-1).gather(1, selected).flatten()
                with torch.no_grad():
                    ref = stage.reference(ids)['logits'][0, len(prefix)-1:-1].float()
                    ref_logp = ref.log_softmax(-1).gather(1, selected).flatten()
                    difference = float((logp.detach() - torch.tensor(logps, device=device)).abs().max())
                    maximum_difference = max(maximum_difference, difference)
                    if (not bool(torch.isfinite(logp).all()) or not bool(torch.isfinite(ref_logp).all())
                            or difference > stage.options['cache_absolute_bound']):
                        raise ValueError('Every fixed action needs finite reference scores and matching cache/full probabilities')
                synthetic = (-logp.mean() + stage.options['KL_weight'] * (logp - ref_logp).square().mean()) / group
                synthetic.backward(inputs=stage.parameters)
                fixed_losses.append(float(synthetic.detach()))
                del ids, selected, logits, logp, ref, ref_logp, synthetic
            if (any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in stage.parameters)
                    or any(p.grad is not None for p in stage.fixed)):
                raise ValueError('Every actual selected actor gradient must allocate finitely without updating any fixed head')
            result = dict(format=FORMAT, complete=True, original_source=copy.deepcopy(info), mode=stage.mode,
                actual_native_group_size=group, full_original_action_horizon=info['full_native_action_budget'],
                actual_fixed_non_EOS_actions_per_history=horizon, actual_fixed_actions=actions,
                actual_original_prefix_tokens=len(prefix), actual_complete_original_CE_targets=anchor['local_original_response_EOS_targets'],
                actual_prefill_batch_sizes=[group], actual_decode_batch_sizes=[group] * (horizon - 1),
                actual_peak_logical_KV_bytes=actual_peak, exact_full_group_horizon_logical_KV_bytes=expected_KV,
                actual_maximum_cached_full_logp_difference=maximum_difference,
                actual_finite_actor_gradient_tensors=len(stage.parameters), actual_synthetic_losses=fixed_losses,
                no_optimizer_step=True, fixed_actions_not_sampled_trajectories=True, no_execution_reward_claimed=True,
                source_cursor_Adam_weights_and_all_private_RNG_unchanged=True,
                full_size_GPU_horizon_allocation_passed=stage.mode == 'capacity',
                full_source_state_readback_and_owned_training_admission_still_required=True,
                production_training_admitted=False, new_model_quality_gain=False, goal_achieved=False)
    finally:
        stage.optimizer.zero_grad(set_to_none=True)
        for module, training in flags: module.training = training
        model.policy.gradient_checkpointing = checkpointing
    after = dict(parameters=signature(model), Adam=value_digest(stage.optimizer.state_dict()),
        RNG=value_digest(capture_rng(device)), completed=stage.completed, metrics=value_digest(stage.metrics))
    if before != after: raise ValueError('Synthetic stress changed a parameter, Adam state, source cursor or private RNG')
    stage.intact()
    return result
