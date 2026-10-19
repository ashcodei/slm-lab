"""Always-on original-code supervision for a fresh execution-policy update.

Use once per replica per source, regardless of sampled rewards. Other parts of
the update can backward each sampled trajectory separately to bound memory.
Every contribution uses its complete global target/action denominator before
gradient SUM; no second worker average should be applied.
"""
import math

import torch
from torch.nn import functional as F


def reference_anchor(logits, labels, global_targets, *, weight=1.):
    if (logits.ndim != 3 or labels.shape != logits.shape[:2] or logits.shape[1] < 2
            or labels.dtype != torch.long or type(global_targets) is not int
            or not math.isfinite(weight) or weight <= 0):
        raise ValueError('Complete original causal logits/labels and positive reference anchor required')
    targets = labels[:, 1:].reshape(-1)
    local = int((targets != -100).sum())
    if (local < 1 or global_targets < local
            or bool(((targets != -100) & ((targets < 0) | (targets >= logits.shape[-1]))).any())):
        raise ValueError('Every valid original response/EOS target needs its complete global denominator')
    summed = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                             targets, ignore_index=-100, reduction='sum')
    loss = weight * summed / global_targets
    if not bool(torch.isfinite(loss)):
        raise ValueError('Original-code anchor must be finite before backward')
    return loss, dict(local_original_response_EOS_targets=local, global_original_response_EOS_targets=global_targets,
        reference_CE_sum=float(summed.detach()), reference_CE_weight=weight,
        original_reference_supervision_is_unconditional_on_execution_rewards=True)
