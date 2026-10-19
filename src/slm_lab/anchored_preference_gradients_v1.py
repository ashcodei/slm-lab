"""Supervised-positive plus DPO gradients with one response graph at a time."""
import math

import torch

from .objectives import dpo_loss, response_logps


def score(model, batch):
    result = model(**{key: value for key, value in batch.items() if key != 'labels'})
    return response_logps(result['logits'], batch['input_ids'], batch['labels'])


def gradients(model, chosen, rejected, reference_chosen, reference_rejected, amp,
              beta=.1, preference_weight=.1):
    """Accumulate the exact chain-rule gradient of NLL + weight * DPO.

    This requires deterministic forwards (no dropout or mutable running state).
    The caller owns gradient clearing, clipping and optimizer updates. Reference
    log probabilities must be frozen scores of these exact token/mask batches.
    """
    if (not math.isfinite(beta) or beta <= 0 or not math.isfinite(preference_weight)
            or preference_weight < 0):
        raise ValueError('Finite positive beta and nonnegative preference weight required')
    if any(batch['input_ids'].shape[0] != 1 for batch in [chosen, rejected]):
        raise ValueError('This isolated preference update requires one complete pair')
    counts = [(batch['labels'][:, 1:] != -100).sum().item() for batch in [chosen, rejected]]
    if min(counts) < 1:
        raise ValueError('Both complete responses require loss targets')
    for reference in [reference_chosen, reference_rejected]:
        if reference.shape != (1,) or reference.requires_grad or not torch.isfinite(reference).all():
            raise ValueError('Each exact frozen reference score must be one finite detached value')
    with torch.no_grad(), amp():
        good = score(model, chosen)
        bad = score(model, rejected)
    good = good.detach().requires_grad_(True)
    bad = bad.detach().requires_grad_(True)
    nll = -good.mean() / counts[0]
    preference = dpo_loss(good, bad, reference_chosen, reference_rejected, beta)
    objective = nll + preference_weight * preference
    if not torch.isfinite(objective):
        raise ValueError('Nonfinite anchored preference objective; no gradients applied')
    good_weight, bad_weight = torch.autograd.grad(objective, [good, bad])
    with amp():
        current_good = score(model, chosen)
    (current_good * good_weight.detach()).sum().backward()
    del current_good
    with amp():
        current_bad = score(model, rejected)
    (current_bad * bad_weight.detach()).sum().backward()
    margin = (good - bad) - (reference_chosen - reference_rejected)
    return {'loss': float(objective.detach()), 'chosen_nll': float(nll.detach()),
            'preference_loss': float(preference.detach()), 'relative_reference_margin': float(margin.detach()),
            'chosen_log_probability': float(good.detach()), 'rejected_log_probability': float(bad.detach()),
            'chosen_target_tokens': counts[0], 'rejected_target_tokens': counts[1]}
