"""Fresh full-vocabulary own-policy attempts with batched native KV decoding.

Each trajectory owns its generator and stops at its first real EOS or horizon.
Finished rows leave the KV batch; no padded actions, repairs, rejection samples
or replacement attempts enter the records. This sampler performs no update and
does not grant full-size GPU capacity or training admission.
"""
from dataclasses import replace
import hashlib

import torch

from .own_dense_actor_v1 import OwnDenseActor, DensePolicyView
from .own_dense_growth_v1 import DenseCache


class BehaviorComparisonError(ValueError):
    def __init__(self, report):
        self.report = report
        super().__init__('Actual batched-cache/full-forward behavior exceeds the declared absolute bound: '
            + str(report['cached_full_max_absolute_difference']))


def state_sha(state):
    return hashlib.sha256(state.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def select_cache(policy, cache, rows):
    """Select actual still-active histories; preserve the original native owner."""
    if (type(policy) is not DensePolicyView or type(cache) is not DenseCache
            or cache.owner != id(policy) or cache.parameter_signature != policy.signature()
            or not isinstance(rows, list) or not rows or len(rows) != len(set(rows))
            or any(type(n) is not int or n < 0 for n in rows)
            or len(cache.layers) != policy.config.num_layers):
        raise ValueError('Exact unchanged own-policy cache and distinct actual active history rows required')
    first = cache.layers[0][0]
    expected = (first.shape[0], policy.config.num_kv_heads, cache.stored_tokens,
        policy.config.hidden_size // policy.config.num_heads)
    if (first.ndim != 4 or max(rows) >= first.shape[0]
            or any(tuple(t.shape) != expected or t.device != first.device or t.dtype != first.dtype
                or t.requires_grad for pair in cache.layers for t in pair)):
        raise ValueError('Every actual layer must preserve the complete batch/head/history geometry')
    index = torch.tensor(rows, dtype=torch.long, device=first.device)
    return replace(cache, layers=tuple(tuple(t.index_select(0, index) for t in pair) for pair in cache.layers))


def signature(model):
    return tuple((id(p), p._version, p.data_ptr(), p.requires_grad, p.dtype, p.device) for p in model.parameters())


@torch.no_grad()
def sample(model, prefix_ids, *, seeds, horizon, purpose, maximum_cache_bytes,
           cache_full_absolute_bound=.001):
    if type(model) is not OwnDenseActor:
        raise ValueError('Exactly one counted own scratch-origin actor is required')
    model.assert_shared()
    parameters = list(model.parameters())
    device = parameters[0].device
    if (purpose not in ['CPU-mechanics', 'GPU-runtime'] or model.training or model.policy.training
            or not isinstance(prefix_ids, list) or not prefix_ids or prefix_ids[0] != 1
            or any(type(t) is not int or not 0 <= t < model.config.vocab_size for t in prefix_ids)
            or not isinstance(seeds, list) or not 2 <= len(seeds) <= 8 or len(seeds) != len(set(seeds))
            or any(type(s) is not int or not 0 <= s < 2**63 for s in seeds)
            or type(horizon) is not int or horizon < 1 or len(prefix_ids) + horizon > model.config.max_text_length
            or type(maximum_cache_bytes) is not int or maximum_cache_bytes < 1
            or type(cache_full_absolute_bound) not in [int, float] or not 0 < cache_full_absolute_bound <= .001
            or torch.is_autocast_enabled(device.type)
            or any(p.dtype != torch.float32 or p.device != device or p.grad is not None for p in parameters)):
        raise ValueError('Explicit complete native FP32 eval-policy group, private seeds, horizon and bounded cache required')
    if purpose == 'CPU-mechanics':
        if device.type != 'cpu' or torch.cuda.is_initialized():
            raise ValueError('Small mechanics must remain CUDA-hidden and CPU-only')
    elif (device.type != 'cuda' or device.index is None or not torch.cuda.is_initialized()
            or torch.backends.cuda.matmul.allow_tf32 or model.counts()['actual_unique'] != 999706442
            or model.base.counts()['total_unique'] != 984399690 or model.config.vocab_size != 32768):
        raise ValueError('GPUruntime requires the actual existing full-size FP32 own actor with TF32 disabled')
    predicted = (2 * len(seeds) * model.config.num_layers * model.config.num_kv_heads
        * (model.config.hidden_size // model.config.num_heads) * (len(prefix_ids) + horizon - 1) * 4)
    if predicted > maximum_cache_bytes:
        raise ValueError('The full declared logical KVcache must fit without shrinking a source or action horizon')
    initial_signature = signature(model)
    initial_flags = tuple(m.training for m in model.modules())
    initial_inventory = tuple(model.state_dict())
    CPU_rng = torch.get_rng_state().clone()
    GPU_rng = torch.cuda.get_rng_state(device).clone() if device.type == 'cuda' else None
    generators = [torch.Generator(device=device).manual_seed(seed) for seed in seeds]
    trajectories = [dict(sample_position=i, private_seed=seed, original_prompt_ids=list(prefix_ids),
        requested_native_action_budget=horizon, actual_generated_token_ids=[], actual_cached_behavior_logps=[],
        initial_private_generator_sha256=state_sha(g.get_state())) for i, (seed, g) in enumerate(zip(seeds, generators))]
    active = list(range(len(seeds)))
    prefill_sizes, decode_sizes, cache_sizes = [], [], []
    state = model.policy.prefill(torch.tensor([prefix_ids] * len(seeds), dtype=torch.long, device=device))
    prefill_sizes.append(len(active))
    for step in range(horizon):
        if state['cache'].bytes > maximum_cache_bytes:
            raise ValueError('Actual logical KVstorage exceeded its declared bound')
        cache_sizes.append(state['cache'].bytes)
        logits = state['logits'][:, -1].float()
        if tuple(logits.shape) != (len(active), model.config.vocab_size) or not bool(torch.isfinite(logits).all()):
            raise ValueError('All active histories require actual finite full-vocabulary logits')
        probabilities = logits.softmax(-1)
        logps = logits.log_softmax(-1)
        choices = torch.stack([torch.multinomial(probabilities[row], 1, generator=generators[position])[0]
            for row, position in enumerate(active)])
        chosen_ids = choices.cpu().tolist()
        selected_logps = logps.gather(1, choices[:, None]).flatten().cpu().tolist()
        remaining_rows, remaining_positions, remaining_ids = [], [], []
        for row, (position, token, logp) in enumerate(zip(active, chosen_ids, selected_logps)):
            trajectories[position]['actual_generated_token_ids'].append(token)
            trajectories[position]['actual_cached_behavior_logps'].append(logp)
            if token != 2 and step + 1 < horizon:
                remaining_rows.append(row)
                remaining_positions.append(position)
                remaining_ids.append(token)
        if not remaining_rows:
            break
        cache = (state['cache'] if len(remaining_rows) == len(active) else
            select_cache(model.policy, state['cache'], remaining_rows))
        active = remaining_positions
        state = model.policy.decode(torch.tensor(remaining_ids, dtype=torch.long, device=device)[:, None], cache)
        decode_sizes.append(len(active))
    del state
    maximum_difference = 0.
    for item, generator in zip(trajectories, generators):
        actions = item['actual_generated_token_ids']
        if (not actions or len(actions) > horizon or 2 in actions[:-1]
                or len(actions) < horizon and actions[-1] != 2):
            raise ValueError('Each exact original sample must stop at its first emitted EOS or declared horizon')
        ids = torch.tensor([prefix_ids + actions], dtype=torch.long, device=device)
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids, dtype=torch.bool))['logits'][0, len(prefix_ids)-1:-1].float()
        actual = logits.log_softmax(-1).gather(1, torch.tensor(actions, device=device)[:, None]).flatten()
        difference = float((actual - torch.tensor(item['actual_cached_behavior_logps'], device=device)).abs().max())
        maximum_difference = max(maximum_difference, difference)
        item.update(actual_full_forward_behavior_logps=actual.cpu().tolist(),
            cached_full_max_absolute_difference=difference, actual_stopped_on_EOS=actions[-1] == 2,
            actual_native_action_count=len(actions), final_private_generator_sha256=state_sha(generator.get_state()),
            actual_full_vocabulary_sampling=True, temperature=1., no_token_filter_repair_or_replacement_sample=True)
    if (signature(model) != initial_signature or tuple(m.training for m in model.modules()) != initial_flags
            or tuple(model.state_dict()) != initial_inventory or any(p.grad is not None for p in parameters)
            or not torch.equal(torch.get_rng_state(), CPU_rng)
            or GPU_rng is not None and not torch.equal(torch.cuda.get_rng_state(device), GPU_rng)):
        raise ValueError('Sampling must preserve every source weight/flag/inventory and original global RNG state')
    report = dict(complete=True, purpose=purpose, actual_device_type=device.type, actual_device_index=device.index,
        actual_actor_parameters=model.counts()['actual_unique'], actual_native_parameters=model.base.counts()['total_unique'],
        actual_vocabulary=model.config.vocab_size, original_prompt_ids=list(prefix_ids), trajectories=trajectories,
        actual_group_size=len(seeds), actual_generated_actions=sum(len(t['actual_generated_token_ids']) for t in trajectories),
        actual_prefill_batch_sizes=prefill_sizes, actual_decode_batch_sizes=decode_sizes,
        actual_peak_logical_KVcache_bytes=max(cache_sizes), conservative_full_horizon_logical_KVcache_bytes=predicted,
        cached_full_max_absolute_difference=maximum_difference, cache_full_absolute_bound=cache_full_absolute_bound,
        all_actual_actions_and_probabilities_preserved=True, finished_histories_never_resampled_or_padded=True,
        original_model_parameters_modes_inventory_and_global_RNG_unchanged=True,
        behavior_rescore_passed=maximum_difference <= cache_full_absolute_bound,
        full_size_GPUcapacity_throughput_recovery_and_RLquality_not_proved_by_this_sampler=True,
        model_updates=0, training_admitted=False, goal_achieved=False)
    if maximum_difference > cache_full_absolute_bound:
        raise BehaviorComparisonError(report)
    return report
