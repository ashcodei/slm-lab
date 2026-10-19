"""Fresh single-update group policy gradient with the same own-reference KL.

One newly sampled group receives exactly one optimizer update. In exact
arithmetic this has the same first derivative as the clipped-ratio objective
at ratio1. It avoids giving FP32 cached/full rounding a false importance ratio.
Actual cached probabilities remain recorded and independently checked by the
caller; this is not an off-policy replay or multi-epoch PPO objective.
"""
import math

import torch


def objective(logits, reference, actions, behavior, advantage, global_actions, *,
              policy_scale, KL_weight, freshly_sampled_single_update):
    if (freshly_sampled_single_update is not True or logits.ndim != 2
            or logits.shape != reference.shape or len(actions) != len(logits)
            or len(behavior) != len(actions) or not actions
            or type(global_actions) is not int or global_actions < len(actions)
            or not math.isfinite(advantage) or type(policy_scale) is not int or policy_scale not in (0,1)
            or not math.isfinite(KL_weight) or KL_weight < 0
            or any(type(a) is not int or not 0 <= a < logits.shape[-1] for a in actions)
            or any(not math.isfinite(p) or p > 0 for p in behavior)):
        raise ValueError('One fresh bounded on-policy group/update with complete global action denominator required')
    logp = logits.float().log_softmax(-1)
    logq = reference.detach().float().log_softmax(-1)
    target = torch.tensor(actions,dtype=torch.long,device=logits.device)
    selected = logp.gather(1,target[:,None]).squeeze(1)
    policy = -advantage*selected.sum()/global_actions
    KL = (logq.exp()*(logq-logp)).sum()/global_actions
    loss = policy_scale*policy+KL_weight*KL
    if not bool(torch.isfinite(loss)):
        raise ValueError('Fresh-group policy and own-reference KL must remain finite')
    return loss,dict(policy_surrogate=float(policy.detach()),reference_KL=float(KL.detach()))
