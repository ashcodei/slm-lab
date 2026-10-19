"""Diagnostic token metrics, with explicit response boundaries and denominators."""
from contextlib import nullcontext

import torch
from torch.nn import functional as F


def token_metrics(logits, ids, target_start):
    if logits.ndim!=3 or ids.ndim!=2 or ids.shape[0]!=1 or logits.shape[:2]!=ids.shape:
        raise ValueError('One complete aligned sequence required')
    if not 1<=target_start<ids.shape[1]:
        raise ValueError('Explicit first supervised response token required')
    target=ids[0,target_start:]
    prediction=logits[0,target_start-1:-1].float()
    logps=prediction.log_softmax(-1)
    nll=-logps.gather(1,target[:,None]).squeeze(1)
    if not torch.isfinite(nll).all():raise ValueError('Finite target likelihood required')
    correct=prediction.argmax(-1).eq(target)
    def summary(start,end):
        end=min(end,len(target));start=min(start,end);count=end-start
        return dict(tokens=count,nll_sum=float(nll[start:end].sum()),correct=int(correct[start:end].sum()))
    return dict(**summary(0,len(target)),first_token_nll=float(nll[0]),
        first_token_correct=bool(correct[0]),first_token_gold_id=int(target[0]),
        first_token_prediction_id=int(prediction[0].argmax()),
        first_8=summary(0,8),positions_9_to_32=summary(8,32),
        positions_33_to_128=summary(32,128),after_128=summary(128,len(target)))


def control_sequence(record, other, limit, target_budget=128):
    if record['probe_id']==other['probe_id'] or record['language']!=other['language']:
        raise ValueError('Distinct same-language control required')
    prefix=other['actual_training_ids'][:other['actual_training_target_start']]
    response=record['actual_training_ids'][record['actual_training_target_start']:][:target_budget]
    if len(prefix)+len(response)>limit:
        raise ValueError('Control whole prompt and declared target window do not fit; never clip the prompt')
    return prefix+response,len(prefix)


def prefix_metrics(generated, target, budget):
    if len(generated)>budget or not target:
        raise ValueError('Bounded generated response and nonempty gold target required')
    compared=min(len(target),budget)
    correct=sum(i<len(generated) and generated[i]==target[i] for i in range(compared))
    common=0
    for x,y in zip(generated,target):
        if x!=y:break
        common+=1
    return dict(generated_tokens=len(generated),compared_gold_prefix_tokens=compared,
        position_matched_tokens=correct,common_exact_prefix_tokens=common,
        exact_bounded_gold_prefix=len(generated)>=compared and generated[:compared]==target[:compared],
        full_response_fits_generation_budget=len(target)<=budget,
        exact_full_training_response_tokens=generated==target if len(target)<=budget else None)


@torch.no_grad()
def padding_cache_diagnostic(model, ids, target_start, amp=nullcontext, padding=7):
    if ids.shape[0]!=1 or target_start<1 or ids.shape[1]+padding>model.config.max_text_length:
        raise ValueError('One full source context with explicit padding capacity required')
    prefix=ids[:,:target_start]
    padded=F.pad(ids,(0,padding),value=0)
    mask=torch.arange(padded.shape[1],device=ids.device)[None,:]<ids.shape[1]
    with amp():
        full=model(ids)['logits'][0]
        masked=model(padded,attention_mask=mask)['logits'][0,:ids.shape[1]]
        cached=model.prefill(prefix)['logits'][0,-1]
    position=target_start-1
    return dict(padding=padding,prompt_tokens=target_start,full_context_tokens=ids.shape[1],
        padded_vs_unpadded_max_abs=float((masked.float()-full.float()).abs().max()),
        padded_vs_unpadded_argmax_agreement=int((masked.argmax(-1)==full.argmax(-1)).sum()),
        padded_vs_unpadded_positions=ids.shape[1],
        first_response_full_vs_cached_max_abs=float((full[position].float()-cached.float()).abs().max()),
        first_response_argmax_agrees=bool(full[position].argmax()==cached.argmax()),
        full_forward_first_response_prediction=int(full[position].argmax()),
        cached_first_response_prediction=int(cached.argmax()))
