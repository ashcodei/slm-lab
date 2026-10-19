"""Connect batched private-seed sampling to the existing strict reasoning RL.

Every sampled action is rescored under the actual gradient computation. Keep
the unchanged full-answer CE anchor and frozen own-reference KL in every group.
This kernel performs no Adam step, persistence, GPUlease or capacity admission.
"""
import math

import torch

from .batched_own_native_rollouts_v2 import sample, BehaviorComparisonError, probability_observation
from .execution_group_controlled_v2 import advantages
from .frozen_own_actor_reference_v1 import FrozenOwnActorReference, signature
from .public_reasoning_group_gradients_v1 import selected, warmup_gradients, clean_failed_gradients
from .single_update_code_policy_v8 import objective

FORMAT = 'own-batched-public-reasoning-fresh-group-gradient-v2'


@clean_failed_gradients
def group_gradients(model, data, position, reference, *, profile, seeds, purpose,
                    maximum_cache_bytes, KL_weight=.01, anchor_weight=1.,
                    cache_absolute_bound=.001, CPU_mechanics_horizon=None):
    parameters, fixed = selected(model, data)
    device = next(model.parameters()).device
    if (type(reference) is not FrozenOwnActorReference or profile not in ['anchor_CE', 'anchor_RL']
            or purpose not in ['CPU-mechanics', 'GPU-runtime']
            or not math.isfinite(KL_weight) or KL_weight < 0
            or not math.isfinite(anchor_weight) or anchor_weight <= 0
            or not math.isfinite(cache_absolute_bound) or not 0 < cache_absolute_bound <= .001
            or any(p.grad is not None for p in parameters)
            or any(p.dtype != torch.float32 for p in model.parameters())
            or device.type == 'cuda' and torch.backends.cuda.matmul.allow_tf32):
        raise ValueError('Exact fresh own-policy FP32group, frozen own reference and finite anchored objective required')
    reference.assert_bound(model)
    info = data.information(position)
    ids, labels = data.sequence(position)
    prefix = ids[:info['original_prompt_positions']]
    horizon = info['full_training_action_budget']
    if CPU_mechanics_horizon is not None:
        if (purpose != 'CPU-mechanics' or device.type != 'cpu' or type(CPU_mechanics_horizon) is not int
                or not 1 <= CPU_mechanics_horizon <= horizon):
            raise ValueError('A short mechanics horizon is explicit CPU-only evidence, never a production source budget')
        horizon = CPU_mechanics_horizon
    fixed_before = signature(fixed)
    model.eval()
    sampling = sample(model, prefix, seeds=seeds, horizon=horizon, purpose=purpose,
        maximum_cache_bytes=maximum_cache_bytes, cache_full_absolute_bound=cache_absolute_bound)
    sampled_signature = model.policy.signature()
    trajectories = []
    for item in sampling['trajectories']:
        verdict = data.score(position, item['actual_generated_token_ids'], horizon=horizon)
        trajectories.append(dict(item, actual_strict_source_verifier=verdict))
    rewards = [t['actual_strict_source_verifier']['reward'] for t in trajectories]
    relative = advantages(rewards)
    action_count = sum(t['actual_native_action_count'] for t in trajectories)
    totals = dict(policy_surrogate=0., reference_KL=0.)
    model.train()
    model.policy.gradient_checkpointing = True
    with torch.autocast(device.type, enabled=False):
        for i, item in enumerate(trajectories):
            if model.policy.signature() != sampled_signature:
                raise ValueError('The actual sampled policy changed before its single gradient update')
            actions = item['actual_generated_token_ids']
            native = torch.tensor([prefix + actions], dtype=torch.long, device=device)
            logits = model(native)['logits'][0, len(prefix)-1:-1]
            with torch.no_grad():
                original = reference(native)['logits'][0, len(prefix)-1:-1]
            actual = logits.detach().float().log_softmax(-1).gather(1, torch.tensor(actions, device=device)[:, None]).flatten()
            finite = bool(torch.isfinite(logits).all()) and bool(torch.isfinite(actual).all())
            difference = (float((actual - torch.tensor(item['actual_cached_behavior_logps'], device=device)).abs().max())
                if finite else None)
            item['actual_gradient_forward_behavior_logps'] = probability_observation(actual)
            item['actual_gradient_cached_max_absolute_difference'] = difference
            if not finite or difference > cache_absolute_bound:
                failure = dict(sampling, trajectories=trajectories, behavior_rescore_passed=False,
                    cached_full_max_absolute_difference=difference,
                    nonfinite_gradient_forward_behavior_observed=not finite,
                    gradient_behavior_mismatch_before_any_optimizer_commit=True)
                raise BehaviorComparisonError(failure)
            loss, metric = objective(logits, original, actions, item['actual_cached_behavior_logps'], relative[i], action_count,
                policy_scale=int(profile == 'anchor_RL'), KL_weight=KL_weight, freshly_sampled_single_update=True)
            loss.backward(inputs=parameters)
            for key, value in metric.items():
                totals[key] += value
            del loss, logits, original
        anchor = warmup_gradients(model, data, [position], amp=lambda: torch.autocast(device.type, enabled=False), weight=anchor_weight)
    if (signature(fixed) != fixed_before or any(p.grad is not None for p in fixed)
            or any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in parameters)):
        raise ValueError('Only all selected actor gradients may accumulate; every original fixed value must stay exact')
    reference.assert_bound(model)
    return dict(format=FORMAT, profile=profile, complete_original_source=info, original_complete_prompt_ids=prefix,
        complete_batched_sampling_observation=sampling, trajectories=trajectories,
        actual_group_size=len(seeds), actual_sampled_action_tokens=action_count,
        actual_strict_rewards=rewards, actual_group_relative_advantages=relative,
        all_equal_reward_groups_have_zero_policy_advantages=len(set(rewards)) == 1,
        unconditional_original_answer_CE_anchor=anchor, anchor_weight=anchor_weight, KL_weight=KL_weight,
        frozen_own_reference_used_in_every_group=True, reference_binding=reference.binding(),
        resident_training_inventory=reference.resident_inventory(model),
        actual_policy_private_generator_seeds=list(seeds), full_original_source_action_budget=info['full_training_action_budget'],
        actual_delivered_action_horizon=horizon, CPU_mechanics_horizon=CPU_mechanics_horizon,
        cached_full_probability_bound=cache_absolute_bound, **totals,
        fresh_group_receives_at_most_one_later_optimizer_update=True,
        kernel_performs_optimizer_updates=0, full_size_GPUcapacity_private_recovery_and_quality_not_established=True,
        human_preferences=False, external_neural_teacher=False, goal_achieved=False)
