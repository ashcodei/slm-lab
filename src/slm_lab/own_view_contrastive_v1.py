"""Experimental contrastive decoding between two views of one own backbone.

Prior work: https://aclanthology.org/2023.acl-long.687/
This uses log p(code) - strength * log p(general), restricted to tokens
whose code probability is at least plausibility times its maximum.
The same-size shared views differ from that paper's large/small model pair.
It adds inference compute and two caches, but no model parameters/training.
"""
import math

import torch

from .own_code_delta_v1 import OwnCodeDelta


def scores(expert_logits, amateur_logits, *, strength, plausibility):
    if (type(strength) not in {int, float} or not math.isfinite(strength) or not 0 <= strength <= 1
            or type(plausibility) not in {int, float} or not math.isfinite(plausibility) or not 0 < plausibility <= 1):
        raise ValueError('Finite strength in [0,1] and plausibility in (0,1] required')
    if (expert_logits.ndim != 2 or expert_logits.shape != amateur_logits.shape
            or expert_logits.device != amateur_logits.device
            or not bool(torch.isfinite(expert_logits).all()) or not bool(torch.isfinite(amateur_logits).all())):
        raise ValueError('Matching finite [batch,vocabulary] logits on the same device required')
    expert = expert_logits.float().log_softmax(-1)
    amateur = amateur_logits.float().log_softmax(-1)
    plausible = expert >= expert.max(-1, keepdim=True).values + math.log(plausibility)
    return (expert - strength * amateur).masked_fill(~plausible, -torch.inf)


@torch.no_grad()
def generate(model, input_ids, pixel_values=None, *, strength, plausibility=.1,
             max_new_tokens=512, eos_token_id=2, checked=lambda: None):
    if type(model) is not OwnCodeDelta or model.training or model.code.training or model.base.training:
        raise ValueError('Exact own conditional-code model in evaluation mode required')
    # Validate options before any prefill or live state access.
    scores(torch.zeros(1, 2), torch.zeros(1, 2), strength=strength, plausibility=plausibility)
    if (input_ids.ndim != 2 or input_ids.shape[1] < 1 or input_ids.shape[0] < 1
            or type(max_new_tokens) is not int or max_new_tokens < 1
            or input_ids.shape[1] + max_new_tokens > model.config.max_text_length
            or type(eos_token_id) is not int or not 0 <= eos_token_id < model.config.vocab_size):
        raise ValueError('Whole original prompt and positive generation budget must fit original context')
    model.assert_partition(); checked()
    expert = model.code.prefill(input_ids, pixel_values)
    amateur = model.base.prefill(input_ids, pixel_values) if strength else None
    tokens = input_ids
    done = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
    trace = []
    for step in range(max_new_tokens):
        checked()
        code_logits = expert['logits'][:, -1].float()
        if strength:
            general_logits = amateur['logits'][:, -1].float()
            selected = scores(code_logits, general_logits, strength=strength, plausibility=plausibility).argmax(-1, keepdim=True)
        else:
            if not bool(torch.isfinite(code_logits).all()):
                raise ValueError('Finite code logits required')
            selected = code_logits.argmax(-1, keepdim=True)
        selected = torch.where(done[:, None], eos_token_id, selected)
        active = ~done
        trace.append(dict(token_ids=selected[:, 0].tolist(), active=active.tolist(),
            code_log_probability=code_logits.log_softmax(-1).gather(-1, selected)[:, 0].tolist(),
            general_log_probability=(general_logits.log_softmax(-1).gather(-1, selected)[:, 0].tolist() if strength else None)))
        tokens = torch.cat((tokens, selected), dim=1)
        done |= selected[:, 0] == eos_token_id
        if bool(done.all()) or step + 1 == max_new_tokens:
            break
        expert = model.code.decode(selected, expert['cache'])
        if strength:
            amateur = model.base.decode(selected, amateur['cache'])
    checked()
    return dict(tokens=tokens, trace=trace, additional_neural_parameters=0,
        inference_views=2 if strength else 1, generation_budget=max_new_tokens,
        stopped_on_all_EOS=bool(done.all()), strength=strength, plausibility=plausibility)
