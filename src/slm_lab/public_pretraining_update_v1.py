"""Reference-free raw/retention CE, with decoder-frozen image/word updates."""
import math

import torch
from torch import distributed as dist

from .fast_joint_stage_v1 import backward_batch, parameter_groups, learning_factor, reduce_gradients
from .protected_joint_stage_v1 import backward_protected

FORMAT = 'own-reference-free-public-pretraining-and-protected-retention-v1'


def backward_public(model, batch, *, kind, global_targets, global_context_targets, global_words=0, amp=None):
    if kind not in {'text','image','word'}:raise ValueError('Explicit native source modality required')
    if kind == 'text':
        if batch.get('pixel_values') is not None:raise ValueError('Raw/text replay cannot hide image inputs')
        result = backward_batch(model, batch, global_targets=global_targets, global_words=0, amp=amp)
        return dict(CE_sum=result['CE_sum'], CTC_word_sum=0.)
    result = backward_protected(model, batch, kind=kind, global_targets=global_targets,
        global_context_targets=global_context_targets, global_words=global_words, reference=None, amp=amp)
    if result['reference_parameters'] != 0 or result['full_context_reference_KL_sum'] != 0:
        raise ValueError('Public pretraining must never construct or query a reference policy')
    return dict(CE_sum=result['CE_sum'], CTC_word_sum=result['CTC_word_sum'])


def update(model, optimizer, data, schedule, step, options, *, device='cpu', rank=0, world=1, amp=None, prepared=None):
    if (data.role != 'train' or data.purpose != 'optimization' or options['ctc_weight'] != 1
            or options['word_native_CE_weight'] != 1 or type(world) is not int or not 0 <= rank < world):
        raise ValueError('Complete admitted train-only public-pretraining profile required')
    name, positions = schedule.at(step);local = positions[rank::world];kind = data.kinds[name]
    targets = sum(data.targets[name][p] for p in positions)
    contexts = sum(data.lengths[name][p]-1 for p in positions)
    words = len(positions) if kind == 'word' else 0
    factor = learning_factor(step, len(schedule), options['warmup_fraction'], options['cosine_floor'])
    for group in optimizer.param_groups:group['lr'] = options['learning_rates'][group['name']]*factor
    model.train();model.network.gradient_checkpointing = True;optimizer.zero_grad(set_to_none=True)
    groups = parameter_groups(model)
    parameters = [p for rows in groups.values() for _,p in rows];selected = {id(p) for p in parameters}
    totals = dict(CE_sum=0., CTC_word_sum=0., response_EOS_targets=0., words=0.)
    length = max(data.lengths[name][p] for p in local) if local else 1
    prefix = model.config.image_token_count(options['image_size']) if kind in {'image','word'} else 0
    width = max(1, min(options['maximum_microbatch'], options['maximum_microbatch_tokens']//(length+prefix)))
    if prepared is not None and len(prepared) != math.ceil(len(local)/width):
        raise ValueError('Every complete original prefetched local microbatch required')
    wanted = ['language'] if kind == 'text' else ['vision']+(['word'] if kind == 'word' else [])
    try:
        for i, begin in enumerate(range(0, len(local), width)):
            batch = data.batch(name, local[begin:begin+width], device) if prepared is None else prepared[i]
            if prepared is not None:
                batch = {k:(v.to(device, non_blocking=True) if isinstance(v,torch.Tensor) else v) for k,v in batch.items()}
            record = backward_public(model, batch, kind=kind, global_targets=targets,
                global_context_targets=contexts, global_words=words, amp=amp)
            totals['CE_sum'] += record['CE_sum'];totals['CTC_word_sum'] += record['CTC_word_sum']
            totals['response_EOS_targets'] += int((batch['labels'][:,1:] != -100).sum())
            totals['words'] += len(batch.get('texts', []))
        if totals['response_EOS_targets'] != sum(data.targets[name][p] for p in local):
            raise ValueError('Every original local native target and EOS must reach backward')
        active = reduce_gradients(parameters, device, world, options['gradient_bucket_values'])
        if (any(p.grad is not None for n,rows in groups.items() if n not in wanted for _,p in rows)
                or any(p.grad is not None for p in model.parameters() if id(p) not in selected)
                or any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for n in wanted for _,p in groups[n])):
            raise ValueError('Every active full-model group and no inactive head needs finite gradients')
        norm = float(torch.nn.utils.clip_grad_norm_(parameters, options['gradient_clip'], error_if_nonfinite=True))
        optimizer.step();optimizer.zero_grad(set_to_none=True)
    except BaseException:
        optimizer.zero_grad(set_to_none=True)
        raise
    metrics = torch.tensor(list(totals.values()), dtype=torch.float64, device=device)
    if world > 1:dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    combined = dict(zip(totals, metrics.cpu().tolist()))
    return dict(step=step+1, objective_format=FORMAT, family=name, global_original_positions=positions,
        global_examples=len(positions), complete_context_tokens=sum(data.lengths[name][p] for p in positions),
        **combined, token_mean_CE=combined['CE_sum']/targets,
        mean_byte_normalized_CTC=combined['CTC_word_sum']/words if words else None,
        actual_active_groups=wanted, resident_training_only_reference_parameters=0, reference_forward_queried=False,
        gradient_norm_before_clip=norm, active_parameter_tensors=active,
        learning_rates={g['name']:g['lr'] for g in optimizer.param_groups})
