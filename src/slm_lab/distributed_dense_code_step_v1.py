"""Whole-source dense SFT with four-way data parallel gradient summation.

Every rank retains the same ordinary FP32 Adam and complete own model. The
global response/EOS denominator is applied before backward; gradients are
summed, never averaged a second time. Empty ranks contribute actual zeros.
"""
import math

import torch
from torch import distributed as dist

from .codenet_checkpoint_v1 import capture_rng
from .complete_response_stopping_loss_v1 import metadata, complete_response_CE_sums
from .frozen_own_actor_reference_v1 import signature
from .own_dense_checkpoint_v1 import selected
from .parallel_function_repair_stage_v1 import value_digest
from .public_reasoned_code_train_data_v1 import PublicReasonedCodeSFTData
from .recovered_reasoned_code_data_v1 import RecoveredReasonedCodeSFTData


def gather(value, group=None):
    values = [None] * dist.get_world_size()
    dist.all_gather_object(values, value, group=group)
    return values


def agree(error, group=None):
    errors = gather(error, group)
    if any(e is not None for e in errors):
        raise ValueError('Collective dense update rejected: ' + repr(errors))


def update(model, optimizer, data, positions, *, checked=lambda: None,
           object_group=None, gradient_clip=1., allow_mechanics=False):
    if not dist.is_initialized():
        raise ValueError('An actual initialized replica population is required')
    rank, world = dist.get_rank(), dist.get_world_size()
    error = None
    try:
        if (type(data) not in (PublicReasonedCodeSFTData, RecoveredReasonedCodeSFTData)
                or data.role != 'train' or data.views != 'both'
                or data.human_preferences is not False or data.own_model_on_policy is not False
                or data.use_as_RL_reward_oracle_forbidden is not True
                or not isinstance(positions, list) or not positions
                or len(positions) != len(set(positions))
                or not math.isfinite(gradient_clip) or gradient_clip <= 0):
            raise ValueError('Complete distinct admitted train sources and finite clip required')
        named = selected(model, optimizer)
        parameters = [p for _, p in named]
        device = parameters[0].device
        small = allow_mechanics is True and device.type == 'cpu' and model.counts()['total_unique'] < 10**7
        if not small and (world != 4 or device.type != 'cuda'
                or model.counts()['total_unique'] != 984399690
                or sum(p.numel() for p in parameters) != 931244544 or len(parameters) != 317
                or model.config.max_text_length != 4096
                or not torch.are_deterministic_algorithms_enabled()
                or torch.backends.cuda.matmul.allow_tf32):
            raise ValueError('Four deterministic actual full-size CUDA replicas required')
        if any(p.dtype != torch.float32 or p.grad is not None for p in model.parameters()):
            raise ValueError('FP32 model and clear gradient boundary required')
        checked()
        data.guard.check()
        info = [data.information(p) for p in positions]
        meta = metadata(info, 'plain_tokens')
        clocks = {float(s['step']) for s in optimizer.state.values()}
        if optimizer.state and (len(optimizer.state) != len(parameters) or len(clocks) != 1):
            raise ValueError('Every common full-language Adam clock is required')
        previous = int(next(iter(clocks))) if clocks else 0
    except Exception as exc:
        error = repr(exc)
    agree(error, object_group)
    descriptors = gather((positions, meta, previous), object_group)
    if any(d != descriptors[0] for d in descriptors):
        raise ValueError('All ranks must agree on complete source order, denominator and clock')
    chosen = {id(p) for p in parameters}
    fixed = [p for p in model.parameters() if id(p) not in chosen]
    before, rng = signature(fixed), value_digest(capture_rng(device))
    model.train()
    model.network.gradient_checkpointing = True
    micros = []
    error = None
    try:
        for local in range(rank, len(positions), world):
            checked()
            position, source = positions[local], info[local]
            batch = data.batch([position], device)
            ids, labels = data.sequence(position)
            if (batch['input_ids'][0].tolist() != ids or batch['labels'][0].tolist() != labels
                    or not bool(batch['attention_mask'].all()) or len(ids) != source['complete_context_tokens']):
                raise ValueError('Every original whole input/response/EOS must reach backward')
            with torch.autocast(device.type, dtype=torch.bfloat16):
                logits = model(batch['input_ids'], attention_mask=batch['attention_mask'], pixel_values=None)['logits']
            if logits.dtype != torch.bfloat16:
                raise ValueError('Actual BF16 forward is required')
            with torch.autocast(device.type, enabled=False):
                ce = complete_response_CE_sums(logits.float(), batch['labels'], batch['attention_mask'], profile='plain_tokens')
                loss = ce['weighted_CE_sums'].sum() / meta['actual_global_weighted_targets']
            loss.backward(inputs=parameters)
            micros.append(dict(position=position, complete_context_tokens=len(ids),
                original_CE_sum=float(ce['original_CE_sums'].detach()), EOS_CE_sum=float(ce['EOS_CE_sums'].detach()),
                forward_dtype=str(logits.dtype), loss_dtype=str(loss.dtype)))
            del logits, loss, ce, batch
        if signature(fixed) != before or any(p.grad is not None for p in fixed):
            raise ValueError('Fixed values must retain their exact boundary')
        for p in parameters:
            if p.grad is None:
                if rank < len(positions):
                    raise ValueError('A nonempty rank omitted a language gradient')
                p.grad = torch.zeros_like(p)
            if not bool(torch.isfinite(p.grad).all()):
                raise ValueError('Every local dense gradient must be finite')
        checked()
    except Exception as exc:
        error = repr(exc)
    agree(error, object_group)
    for p in parameters:
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
    norm = float(torch.nn.utils.clip_grad_norm_(parameters, gradient_clip, error_if_nonfinite=True))
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    error = None
    if (signature(fixed) != before or value_digest(capture_rng(device)) != rng
            or any(float(optimizer.state[p]['step']) != previous + 1 for p in parameters)):
        error = 'Fixed values/private RNG/common ordinary Adam clock failed'
    agree(error, object_group)
    all_micros = [m for part in gather(micros, object_group) for m in part]
    order = {p: i for i, p in enumerate(positions)}
    all_micros.sort(key=lambda row: order[row['position']])
    if [m['position'] for m in all_micros] != positions:
        raise ValueError('Every global whole source must occur exactly once')
    ce_sum = math.fsum(m['original_CE_sum'] for m in all_micros)
    eos_sum = math.fsum(m['EOS_CE_sum'] for m in all_micros)
    return dict(actual_positions=list(positions), source_metadata=meta, actual_microbatches=all_micros,
        original_token_mean_NLL=ce_sum/meta['response_EOS_targets'], EOS_token_mean_NLL=eos_sum/meta['genuine_EOS_targets'],
        weighted_token_mean_CE=ce_sum/meta['actual_global_weighted_targets'], gradient_norm_before_clip=norm,
        selected_Adam_states=len(optimizer.state), actual_Adam_step=previous+1,
        selected_full_language_values=sum(p.numel() for p in parameters), selected_full_language_tensors=len(parameters),
        protocol='distributed_full_dense_BF16_forward_FP32_CE_summed_gradients_Adam_v1', world_size=world,
        rank_source_counts=[len(positions[r::world]) for r in range(world)],
        private_RNG_unchanged=True, fixed_vision_critic_and_auxiliary_parameters_unchanged=True,
        complete_original_sources_and_EOS_preserved=True, full_epoch_or_model_quality_proved=False)
