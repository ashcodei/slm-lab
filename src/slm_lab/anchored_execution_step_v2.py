"""Fresh anchored execution with a frozen snapshot of our own starting actor.

This kernel does not own production scheduling, source epochs, checkpoints or
GPU admission. A production caller must bind those and complete the new full
context/action-horizon four-GPU capacity/recovery checks first. CPU mechanics
may explicitly shorten generation, never the original CE sequence or target.
"""
import math

import torch
from torch import distributed as dist

PARENT_KERNEL_SHA256 = '6ac2c1f63c0f11e1671813185eb781b6bdf0bf933bd36845a2fba916131d3aa2'

from .anchored_code_objective_v1 import reference_anchor
from .execution_group_controlled_v2 import advantages
from .expanded_execution_corpus_v1 import ExpandedExecutionCorpus
from .own_dense_actor_v1 import OwnDenseActor
from .frozen_own_actor_reference_v1 import FrozenOwnActorReference
from .single_update_code_policy_v8 import objective


def update(model, optimizer, corpus, position, *, own_reference, method, local_group_size=1,
           rank=0, world=1, control_group=None, gradient_group=None,
           cache_absolute_bound, KL_weight=.01, anchor_weight=1., CPU_mechanics_horizon=None):
    if (type(model) is not OwnDenseActor or type(corpus) is not ExpandedExecutionCorpus
            or method not in ['anchor_CE', 'anchor_execution']
            or type(local_group_size) is not int or not 1 <= local_group_size <= 4
            or type(world) is not int or not 1 <= world <= 4 or type(rank) is not int or not 0 <= rank < world
            or not math.isfinite(cache_absolute_bound) or not 0 < cache_absolute_bound <= .001
            or not math.isfinite(KL_weight) or KL_weight < 0 or not math.isfinite(anchor_weight) or anchor_weight <= 0):
        raise ValueError('Explicit counted own actor, fresh global group and finite anchored objectives required')
    if type(own_reference) is not FrozenOwnActorReference:
        raise ValueError('Frozen snapshot of the actual own starting policy required')
    own_reference.assert_bound(model)
    device = next(model.parameters()).device
    if (any(p.dtype != torch.float32 for p in model.parameters())
            or device.type == 'cuda' and torch.backends.cuda.matmul.allow_tf32):
        raise ValueError('This sampler/backward kernel requires FP32 with TF32 disabled')
    if world > 1 and (not dist.is_initialized() or dist.get_world_size(control_group) != world
                      or dist.get_world_size(gradient_group) != world
                      or dist.get_backend(control_group) != 'gloo' or dist.get_backend(gradient_group) != 'gloo'):
        raise ValueError('Exact Gloo control and host-gradient replica groups required')
    parameters = [p for _, p in model.named_selected(('actor',))]
    if [id(p) for group in optimizer.param_groups for p in group['params']] != [id(p) for p in parameters]:
        raise ValueError('Only all selected actor parameters may enter this optimizer')
    fixed = [p for p in model.parameters() if id(p) not in {id(p) for p in parameters}]
    def signature(values): return [(id(p), p._version, p.data_ptr(), p.requires_grad) for p in values]
    fixed_signature = signature(fixed)
    if any(p.grad is not None for p in fixed):
        raise ValueError('Fixed own reference, auxiliary heads and critic must start without gradients')
    row = corpus.row(position); ids_, labels_ = corpus.sequence(position)
    prefix = ids_[:row['prefix_positions']]
    horizon = row['full_training_action_budget']
    if CPU_mechanics_horizon is not None:
        if device.type != 'cpu' or type(CPU_mechanics_horizon) is not int or not 1 <= CPU_mechanics_horizon <= horizon:
            raise ValueError('Short mechanics horizon is CPU-only and cannot alter the native source budget')
        horizon = CPU_mechanics_horizon
    if len(prefix) + horizon > model.config.max_text_length:
        raise ValueError('Complete native prefix and entire declared action budget must fit')

    def gather(value):
        if world == 1: return [value]
        values = [None] * world
        dist.all_gather_object(values, value, group=control_group)
        return values

    def coordinated(error, phase):
        errors = gather(error)
        if any(x is not None for x in errors):
            raise ValueError(f'Coordinated {phase} rejection before optimizer commit: {errors}')

    specification = dict(position=position, source_id=row['id'], method=method, world=world,
        local_group_size=local_group_size, original_targets=row['response_EOS_targets'],
        action_horizon=horizon, KL_weight=KL_weight, anchor_weight=anchor_weight,
        cache_absolute_bound=cache_absolute_bound)
    if any(item != specification for item in gather(specification)):
        raise ValueError('Every replica must use the same source and complete normalization profile')
    sampled_signature = model.policy.signature(); local = []; error = None
    try:
        model.eval()
        with torch.no_grad(), torch.autocast(device.type, enabled=False):
            initial = model.policy.prefill(torch.tensor([prefix], device=device))
            for _ in range(local_group_size):
                state = initial; actions = []; behavior = []
                for i in range(horizon):
                    logits = state['logits'][0, -1].float()
                    token = int(torch.multinomial(logits.softmax(-1), 1))
                    actions.append(token); behavior.append(float(logits.log_softmax(-1)[token]))
                    if token == 2 or i + 1 == horizon: break
                    state = model.policy.decode(torch.tensor([[token]], device=device), state['cache'])
                response = corpus.tokenizer.decode(actions)
                local.append(dict(response_ids=actions, actual_cached_behavior_logps=behavior, own_response=response,
                    actual_execution_reward=corpus.score(position, response, stopped_on_eos=actions[-1] == 2),
                    temperature=1., full_vocabulary_sampling=True))
            del initial, state
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    coordinated(error, 'sampling/execution')
    trajectories = [item for replica in gather(local) for item in replica]
    rewards = [item['actual_execution_reward']['reward'] for item in trajectories]
    relative = advantages(rewards)
    global_actions = sum(len(item['response_ids']) for item in trajectories)
    local_metrics = dict(policy_surrogate=0., reference_KL=0., original_reference_CE_sum=0.)
    differences = []; model.train(); model.policy.gradient_checkpointing = True
    optimizer.zero_grad(set_to_none=True)
    try:
        error = None
        try:
            if model.policy.signature() != sampled_signature:
                raise ValueError('Actual sampling policy changed before its sole optimizer update')
            with torch.autocast(device.type, enabled=False):
                for index, item in enumerate(local):
                    ids = torch.tensor([prefix + item['response_ids']], device=device)
                    logits = model(ids)['logits'][0, len(prefix)-1:-1]
                    with torch.no_grad(): reference = own_reference(ids)['logits'][0, len(prefix)-1:-1]
                    full = logits.detach().float().log_softmax(-1).gather(1, torch.tensor(item['response_ids'], device=device)[:, None]).flatten()
                    cached = torch.tensor(item['actual_cached_behavior_logps'], device=device)
                    maximum = float((full - cached).abs().max()); differences.append(maximum)
                    if maximum > cache_absolute_bound:
                        raise ValueError(f'Cached/full actual behavior exceeds explicit guard: {maximum}')
                    item['actual_full_behavior_logps'] = full.cpu().tolist()
                    loss, metrics = objective(logits, reference, item['response_ids'], item['actual_cached_behavior_logps'],
                        relative[rank * local_group_size + index], global_actions,
                        policy_scale=int(method == 'anchor_execution'), KL_weight=KL_weight, freshly_sampled_single_update=True)
                    loss.backward(inputs=parameters)
                    for key, value in metrics.items(): local_metrics[key] += value
                    del loss, logits, reference
                # Always executed, including positive and varying reward groups.
                batch = corpus.batch(position, device)
                labels = batch.pop('labels')
                logits = model(**batch)['logits']
                loss, metric = reference_anchor(logits, labels, world * row['response_EOS_targets'], weight=anchor_weight)
                loss.backward(inputs=parameters)
                local_metrics['original_reference_CE_sum'] = metric['reference_CE_sum'] / world
                del loss, logits
            if (any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in parameters)
                    or any(p.grad is not None for p in fixed) or signature(fixed) != fixed_signature):
                raise ValueError('All selected gradients and every fixed parameter must preserve their contract')
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
        coordinated(error, 'forward/backward')
        if world > 1:
            flat = torch.cat([p.grad.reshape(-1) for p in parameters]); host = flat.cpu()
            dist.all_reduce(host, op=dist.ReduceOp.SUM, group=gradient_group)
            flat.copy_(host); offset = 0
            for p in parameters:
                p.grad.copy_(flat[offset:offset+p.numel()].view_as(p)); offset += p.numel()
            del flat, host
        norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True))
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
    except BaseException:
        optimizer.zero_grad(set_to_none=True)
        raise
    own_reference.assert_bound(model)
    metrics = gather(local_metrics)
    # Full local records now include actual full-forward behavior probabilities.
    trajectories = [item for replica in gather(local) for item in replica]
    return dict(**specification, complete_prompt_ids=prefix, trajectories=trajectories,
        global_rewards=rewards, global_advantages=relative, global_sampled_action_tokens=global_actions,
        original_reference_CE_sum=math.fsum(m['original_reference_CE_sum'] for m in metrics),
        original_reference_token_mean_CE=math.fsum(m['original_reference_CE_sum'] for m in metrics) / row['response_EOS_targets'],
        policy_surrogate=math.fsum(m['policy_surrogate'] for m in metrics),
        reference_KL=math.fsum(m['reference_KL'] for m in metrics),
        original_reference_used_regardless_of_rewards=True, fresh_group_optimizer_updates=1,
        source_reference_targets=row['response_EOS_targets'], gradient_norm_before_clip=norm,
        cached_full_max_absolute_differences=[value for replica in gather(differences) for value in replica],
        CPU_mechanics_horizon=CPU_mechanics_horizon, full_GPU_profile_capacity_not_proved_by_this_kernel=True,
        own_reference_is_frozen_starting_policy=True, reference_binding=own_reference.binding(),
        resident_training_inventory=own_reference.resident_inventory(model),
        human_preferences=False, genuine_human_RLHF=False, candidate_promoted=False, goal_achieved=False)
