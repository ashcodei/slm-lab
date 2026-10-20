"""Whole-response CE controls with one genuine terminal EOS per source.

Weighting is an experiment, not a claim of improved stopping or correctness.
No response token is removed, shortened, duplicated, or replaced by a reward.
The caller divides each local sum by the same global weighted target count.
"""
import copy

import torch
from torch.nn import functional as F

PROFILES = {'plain_tokens': 1., 'eos_weight4': 4.}


def eos_weight(profile):
    if profile not in PROFILES or PROFILES != {'plain_tokens': 1., 'eos_weight4': 4.}:
        raise ValueError('Exactly the fixed plain-CE or EOS-weight4 comparison required')
    return PROFILES[profile]


def metadata(information, profile):
    weight = eos_weight(profile)
    if not isinstance(information, list) or not information:
        raise ValueError('A nonempty global batch of complete original source information is required')
    for info in information:
        if (info['source_role'] != 'train' or info['view'] not in ('code_only', 'reasoning')
                or info['supervision'] != 'public_synthetic_weak_SFT_only'
                or info['use_as_RL_reward_oracle_forbidden'] is not True
                or any(type(info[k]) is not int for k in
                    ('position', 'complete_context_tokens', 'original_prompt_positions', 'response_EOS_targets'))
                or not 1 <= info['original_prompt_positions'] < info['complete_context_tokens'] <= 4096
                or info['response_EOS_targets'] != info['complete_context_tokens']-info['original_prompt_positions']):
            raise ValueError('Full admitted train code/reasoning contexts and targets must remain exact')
    if len({info['position'] for info in information}) != len(information):
        raise ValueError('No source may be rounded or repeated within an original global batch')
    targets = sum(info['response_EOS_targets'] for info in information)
    count = len(information)
    return dict(source_presentations=count, complete_context_tokens=sum(info['complete_context_tokens'] for info in information),
        response_EOS_targets=targets, genuine_EOS_targets=count, non_EOS_response_targets=targets-count,
        EOS_weight=weight, actual_global_weighted_targets=targets+(weight-1)*count,
        complete_sources=copy.deepcopy(information))


def complete_response_CE_sums(logits, labels, attention_mask, *, profile):
    """Return differentiable per-source sums; never normalize per microbatch."""
    weight = eos_weight(profile)
    if (logits.ndim != 3 or logits.dtype != torch.float32 or logits.shape[0] < 1
            or logits.shape[1] < 2 or logits.shape[2] < 5 or labels.shape != logits.shape[:2]
            or labels.dtype != torch.long or attention_mask.shape != labels.shape
            or attention_mask.dtype != torch.bool or labels.device != logits.device
            or attention_mask.device != logits.device):
        raise ValueError('Actual FP32 logits and matching complete long labels/bool masks required')
    positions = torch.arange(labels.shape[1], device=labels.device)[None, :]
    lengths = attention_mask.sum(-1)
    target_mask = labels.ne(-100)
    counts = target_mask.sum(-1)
    starts = lengths-counts
    if (bool((counts < 1).any()) or bool((starts < 1).any())
            or not torch.equal(attention_mask, positions < lengths[:, None])
            or not torch.equal(target_mask, (positions >= starts[:, None]) & attention_mask)):
        raise ValueError('Every full prompt must be masked and the whole response plus EOS contiguous; padding stays ignored')
    terminal = positions == lengths[:, None]-1
    if (not torch.equal(labels.eq(2), terminal)
            or bool(((labels < 4) & target_mask & ~terminal).any())
            or bool(((labels >= logits.shape[-1]) & target_mask).any())):
        raise ValueError('Exactly one genuine last EOS and every original non-special response token required')
    losses = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1),
        ignore_index=-100, reduction='none').reshape(labels.shape[0], -1)
    ordinary = losses.sum(-1)
    stopping = (losses * terminal[:, 1:]).sum(-1)
    weighted = ordinary+(weight-1)*stopping
    if not bool(torch.isfinite(weighted).all()):
        raise ValueError('Every complete original response/EOS CE must be finite')
    return dict(original_CE_sums=ordinary, EOS_CE_sums=stopping, weighted_CE_sums=weighted,
        original_target_counts=counts, genuine_EOS_counts=torch.ones_like(counts),
        weighted_target_counts=counts+(weight-1))
