"""Fresh batched suffix policy gradients with bounded public execution rewards.

Run independent CPUscoring tasks while the complete unconditional original CE
gradient is computed. Every trajectory, including failures and abstentions,
remains. This performs no optimizer step, resource admission or promotion.
"""
from concurrent.futures import ThreadPoolExecutor
import copy
import math
import time

import torch

from .anchored_code_objective_v1 import reference_anchor
from .batched_own_native_rollouts_v2 import sample, probability_observation
from .codenet_checkpoint_v1 import capture_rng
from .execution_group_controlled_v2 import advantages
from .frozen_own_actor_reference_v1 import FrozenOwnActorReference, signature
from .native_code_execution_oracle_v2 import safe_observation
from .own_dense_actor_v1 import OwnDenseActor
from .parallel_function_repair_stage_v1 import value_digest
from .public_reasoning_group_gradients_v1 import clean_failed_gradients
from .qualified_native_token_suffix_episode_v1 import QualifiedNativeTokenSuffixEpisode
from .single_update_code_policy_v8 import objective
from .source_execution_threads_v1 import initialize, observation

FORMAT = 'own-batched-exact-native-code-suffix-fresh-group-gradients-v1'


class CodeGroupFailure(ValueError):
    def __init__(self, report):
        self.report = safe_observation(report)
        super().__init__('Native code gradient group rejected before any optimizer step: ' + str(report.get('phase')))


def verdict_semantics(result):
    """Compare actual code outcomes, retaining timings separately in full rows."""
    keys = ['reward', 'status', 'actual_stopped_on_EOS', 'actual_generated_suffix_ids',
        'actual_supplied_public_reference_ids', 'complete_actual_response', 'actual_delivered_action_budget']
    value = {k: result.get(k) for k in keys}
    raw = result.get('actual_original_oracle_observation')
    if raw is not None:
        value['actual_oracle'] = {k: raw[k] for k in ['reward', 'status', 'informative', 'supported',
            'informative_cases_passed', 'informative_cases', 'all_finite_original_cases_passed',
            'full_reference_stream_passed', 'all_original_published_examples_passed',
            'all_augmented_reference_observations_passed', 'added_probe_passes', 'scored_observations',
            'scored_passes', 'scored_examples_passed', 'scored_examples'] if k in raw}
        actual = raw.get('actual_execution', raw.get('raw_candidate_execution'))
        if actual is not None:
            value['actual_execution'] = {k: actual[k] for k in ['source_id', 'actual_program_sha256',
                'original_program_sha256', 'program_sha256', 'original_tests_sha256', 'mode', 'total_cases',
                'passed_cases', 'syntax_valid', 'all_cases_observed', 'case_passes', 'status',
                'original_example_passes', 'complete'] if k in actual}
            rows = actual.get('raw_results')
            if isinstance(rows, dict):
                value['actual_execution']['returned_values'] = [r['value'] for r in rows['results']]
            elif isinstance(rows, list):
                value['actual_execution']['actual_original_commands_and_probe_results'] = copy.deepcopy(rows)
    return value


def group_semantics(record):
    return dict(episode=record['episode_binding'], profile=record['profile'], seeds=record['actual_private_seeds'],
        horizon=record['actual_delivered_action_horizon'], anchor=record['unconditional_original_CE_anchor'],
        rewards=record['actual_rewards'], advantages=record['actual_advantages'],
        trajectories=[dict(actions=r['actual_generated_token_ids'], cached=r['actual_cached_behavior_logps'],
            full=r['actual_full_forward_behavior_logps'], gradient=r['actual_gradient_forward_behavior_logps'],
            actual_execution_semantics=verdict_semantics(r['actual_native_execution'])) for r in record['trajectories']])


@clean_failed_gradients
def group_gradients(model, episode, reference, *, profile, seeds, purpose, maximum_cache_bytes,
                    KL_weight=.01, anchor_weight=1., cache_absolute_bound=.001,
                    execution_workers=2, CPU_mechanics_horizon=None):
    if (type(model) is not OwnDenseActor or type(episode) is not QualifiedNativeTokenSuffixEpisode
            or type(reference) is not FrozenOwnActorReference or profile not in ['anchor_CE', 'anchor_execution']
            or purpose not in ['CPU-mechanics', 'GPU-runtime'] or type(execution_workers) is not int or not 1 <= execution_workers <= 2
            or any(not math.isfinite(v) for v in [KL_weight, anchor_weight, cache_absolute_bound])
            or KL_weight < 0 or anchor_weight <= 0 or not 0 < cache_absolute_bound <= .001):
        raise ValueError('Exact own actor, source-qualified native episode and finite anchored objective are required')
    model.assert_shared()
    reference.assert_bound(model)
    parameters = [p for _, p in model.named_selected(('actor',))]
    fixed = [p for p in model.parameters() if id(p) not in {id(q) for q in parameters}]
    device = next(model.parameters()).device
    if (any(p.grad is not None or p.dtype != torch.float32 or p.device != device for p in model.parameters())
            or device.type == 'cuda' and torch.backends.cuda.matmul.allow_tf32
            or purpose == 'CPU-mechanics' and (device.type != 'cpu' or model.counts()['actual_unique'] >= 10**6 or torch.cuda.is_initialized())
            or purpose == 'GPU-runtime' and (episode.scope != 'complete-source-catalogue' or CPU_mechanics_horizon is not None)):
        raise ValueError('Clear FP32gradients and explicit small-CPU or complete-source actual-GPU semantics required')
    prepared = episode.rollout_view()
    prefix = prepared['native_rollout_prefix_ids']
    full_horizon = prepared['full_native_action_budget']
    horizon = full_horizon
    if CPU_mechanics_horizon is not None:
        if purpose != 'CPU-mechanics' or type(CPU_mechanics_horizon) is not int or not 1 <= CPU_mechanics_horizon <= full_horizon:
            raise ValueError('Short mechanics horizons never change a production source budget')
        horizon = CPU_mechanics_horizon
    before_fixed, before_RNG = signature(fixed), value_digest(capture_rng(device))
    parent_context = observation()
    record = dict(format=FORMAT, episode_binding=episode.binding(), profile=profile, actual_private_seeds=list(seeds),
        actual_delivered_action_horizon=horizon, full_original_native_horizon=full_horizon,
        CPU_mechanics_horizon=CPU_mechanics_horizon, trajectories=[])
    try:
        with episode.oracle.checked_source_batch():
            episode.check()
            model.eval()
            sampling = sample(model, prefix, seeds=seeds, horizon=horizon, purpose=purpose,
                maximum_cache_bytes=maximum_cache_bytes, cache_full_absolute_bound=cache_absolute_bound)
            record['complete_batched_sampling_observation'] = sampling
            sampled_signature = model.policy.signature()
            trajectories = copy.deepcopy(sampling['trajectories'])
            record['trajectories'] = trajectories
            def execute(item):
                began = time.monotonic()
                result = episode.score(item['actual_generated_token_ids'], horizon=horizon)
                return dict(actual_native_execution=result, actual_execution_thread=observation(),
                    actual_execution_started=began, actual_execution_finished=time.monotonic())
            anchor_error = None
            with ThreadPoolExecutor(max_workers=execution_workers, initializer=initialize) as pool:
                futures = [pool.submit(execute, t) for t in trajectories]
                model.train()
                model.policy.gradient_checkpointing = True
                anchor_started = time.monotonic()
                try:
                    batch = episode.batch(device)
                    labels = batch.pop('labels')
                    with torch.autocast(device.type, enabled=False):
                        logits = model(**batch)['logits']
                        loss, anchor = reference_anchor(logits, labels,
                            prepared['original_response_EOS_targets'], weight=anchor_weight)
                        loss.backward(inputs=parameters)
                    record['unconditional_original_CE_anchor'] = anchor
                    del loss, logits, batch
                except Exception as exc:
                    anchor_error = repr(exc)
                anchor_finished = time.monotonic()
                execution_errors = []
                for item, future in zip(trajectories, futures):
                    try: item.update(future.result())
                    except Exception as exc:
                        item['actual_native_execution'] = dict(reward=None, status='preserved_execution_exception', error=repr(exc))
                        execution_errors.append(repr(exc))
                if anchor_error or execution_errors:
                    raise CodeGroupFailure(dict(record, phase='original_CE_or_actual_execution', anchor_error=anchor_error,
                        actual_execution_errors=execution_errors, all_sampled_attempts_retained=True))
            rewards = [t['actual_native_execution']['reward'] for t in trajectories]
            if any(r is not None and (type(r) not in [int, float] or not math.isfinite(r) or not 0 <= r <= 1) for r in rewards):
                raise CodeGroupFailure(dict(record, phase='invalid_actual_reward'))
            informative = all(r is not None for r in rewards)
            relative = advantages(rewards) if informative else [0.] * len(rewards)
            record.update(actual_rewards=rewards, actual_advantages=relative, policy_rewards_informative=informative,
                policy_gradient_abstained_due_to_missing_verdict=not informative,
                actual_CE_interval=[anchor_started, anchor_finished],
                actual_CE_and_execution_task_intervals_overlap=any(t['actual_execution_started'] < anchor_finished
                    and t['actual_execution_finished'] > anchor_started for t in trajectories))
            action_count = sum(len(t['actual_generated_token_ids']) for t in trajectories)
            totals = dict(policy_surrogate=0., reference_KL=0.)
            with torch.autocast(device.type, enabled=False):
                for i, item in enumerate(trajectories):
                    if model.policy.signature() != sampled_signature:
                        raise CodeGroupFailure(dict(record, phase='sampled_policy_changed_before_its_only_gradient_update'))
                    actions = item['actual_generated_token_ids']
                    native = torch.tensor([prefix + actions], dtype=torch.long, device=device)
                    logits = model(native)['logits'][0, len(prefix)-1:-1]
                    with torch.no_grad(): original = reference(native)['logits'][0, len(prefix)-1:-1]
                    full = logits.detach().float().log_softmax(-1).gather(1, torch.tensor(actions, device=device)[:, None]).flatten()
                    finite = bool(torch.isfinite(logits).all()) and bool(torch.isfinite(full).all())
                    difference = float((full - torch.tensor(item['actual_cached_behavior_logps'], device=device)).abs().max()) if finite else None
                    item['actual_gradient_forward_behavior_logps'] = probability_observation(full)
                    item['actual_gradient_cached_max_absolute_difference'] = difference
                    if not finite or difference > cache_absolute_bound:
                        raise CodeGroupFailure(dict(record, phase='nonfinite_or_mismatched_gradient_behavior',
                            nonfinite_gradient_forward_behavior_observed=not finite))
                    loss, metrics = objective(logits, original, actions, item['actual_cached_behavior_logps'], relative[i],
                        action_count, policy_scale=int(profile == 'anchor_execution'), KL_weight=KL_weight,
                        freshly_sampled_single_update=True)
                    loss.backward(inputs=parameters)
                    for key, value in metrics.items(): totals[key] += value
                    del loss, logits, original
            episode.check()
        if (signature(fixed) != before_fixed or any(p.grad is not None for p in fixed)
                or any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in parameters)
                or value_digest(capture_rng(device)) != before_RNG or observation() != parent_context):
            raise CodeGroupFailure(dict(record, phase='fixed_values_gradients_private_RNG_or_parent_scheduler_changed'))
        reference.assert_bound(model)
        record.update(**totals, actual_group_size=len(seeds), actual_sampled_action_tokens=action_count,
            KL_weight=KL_weight, anchor_weight=anchor_weight, cache_full_absolute_bound=cache_absolute_bound,
            fixed_base_and_unused_critic_unchanged=True, all_original_reference_CE_targets_unconditional=True,
            actual_private_global_RNG_unchanged=True, original_parent_execution_context_unchanged=True,
            execution_workers=execution_workers, own_reference_binding=reference.binding(),
            resident_training_inventory=reference.resident_inventory(model),
            kernel_performs_optimizer_updates=0, full_size_GPUcapacity_recovery_and_quality_not_proved=True,
            conditional_public_hint_not_unassisted_code_score=True, human_preferences=False,
            external_neural_teacher=False, production_training_admitted=False, goal_achieved=False)
        return record
    except CodeGroupFailure: raise
    except Exception as exc:
        raise CodeGroupFailure(dict(record, phase='unhandled_source_or_gradient_failure', actual_error=repr(exc))) from exc
