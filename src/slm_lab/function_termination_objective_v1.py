"""Complete function CE plus an explicit auxiliary on its one real EOS target.

The auxiliary reweights an already supervised true source EOS. It creates no
new target and never truncates or repairs a program. Weight-zero is the matched
whole-code CE control. Useful generated stopping/correctness remains unproved.
"""
import math

import torch
from torch.nn import functional as F

from .fast_joint_stage_v1 import parameter_groups


def backward(model,batch,*,global_targets,global_examples,EOS_weight,amp=None):
    from contextlib import nullcontext
    if (global_targets < 1 or global_examples < 1 or not math.isfinite(EOS_weight) or not 0 <= EOS_weight <= 1
            or batch.get('pixel_values') is not None):
        raise ValueError('Explicit full function target/example counts and bounded true-EOS auxiliary required')
    ids,labels,attention = (batch[k] for k in ['input_ids','labels','attention_mask'])
    if (ids.ndim != 2 or not len(ids) or ids.shape != labels.shape or ids.shape != attention.shape
            or attention.dtype != torch.bool):
        raise ValueError('Exact right-padded native function tensor shapes and masks required')
    lengths = attention.sum(1)
    target = labels[:,1:];mask = target != -100
    rows = torch.arange(len(ids),device=ids.device);last = lengths-2
    if (ids.shape != labels.shape or attention.shape != ids.shape or bool((lengths < 3).any())
            or bool((attention[:,1:] & ~attention[:,:-1]).any())
            or not bool((ids[rows,lengths-1] == 2).all()) or not bool((target[rows,last] == 2).all())
            or not bool(((target == 2).sum(1) == 1).all()) or not bool(mask.any(1).all())
            or bool((mask & ~attention[:,1:]).any()) or not bool((ids[:,0] == 1).all())
            or not bool((labels[:,0] == -100).all())
            or not torch.equal(target[mask],ids[:,1:][mask])):
        raise ValueError('Every complete function must retain exactly its original final EOS and native response mask')
    first = mask.to(torch.int64).argmax(1)
    positions = torch.arange(target.shape[1],device=target.device)[None,:]
    if not torch.equal(mask,(positions >= first[:,None]) & (positions <= last[:,None])):
        raise ValueError('The original whole function response must remain one complete contiguous target span')
    groups = parameter_groups(model)
    language = [p for _,p in groups['language']]
    inactive = [p for n,group in groups.items() if n != 'language' for _,p in group]
    if any(p.grad is not None for p in inactive):raise ValueError('No stale image/word gradients may enter function learning')
    with (amp or nullcontext)():
        logits = model(input_ids=ids,attention_mask=attention)['logits'][:,:-1].float()
    CE = F.cross_entropy(logits[mask],target[mask],reduction='sum')
    EOS_CE = F.cross_entropy(logits[rows,last],target[rows,last],reduction='sum')
    objective = CE/global_targets+EOS_weight*EOS_CE/global_examples
    if not bool(torch.isfinite(objective)):raise ValueError('The actual whole-function objective must remain finite')
    objective.backward(inputs=language)
    if any(p.grad is not None for p in inactive):raise ValueError('Function learning changed an inactive modality')
    return dict(complete_function_CE_sum=float(CE.detach()),true_final_EOS_CE_sum=float(EOS_CE.detach()),
        complete_response_EOS_targets=int(mask.sum()),complete_examples=len(ids),EOS_auxiliary_weight=EOS_weight,
        existing_real_EOS_target_reweighted_not_new_target=True,reference_neural_forwards=0,
        generated_code_stopping_or_correctness_gain_not_established=True)
