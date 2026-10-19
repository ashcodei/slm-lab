"""Candidate anti-interference objective; not yet a production recipe.

Image/word batches train vision/readout through a fixed language decoder.
Text batches train language with full native-context KL to our own frozen
initial policy. The extra reference is training-only and separately counted.
Actual capacity, source-complete scheduling and quality comparison are pending.
"""
from contextlib import nullcontext
import math

import torch
from torch.nn import functional as F

from .fast_joint_stage_v1 import parameter_groups
from .own_dense_growth_v1 import OwnDenseAssistant
from .shared_word_stage_v1 import ctc_word_sum


def backward_protected(model,batch,*,kind,global_targets,global_context_targets,
                       reference=None,reference_KL_weight=.1,global_words=0,amp=None):
    if (kind not in ('text','image','word') or global_targets<=0 or global_context_targets<=0
            or not math.isfinite(reference_KL_weight) or reference_KL_weight<=0
            or (kind=='word' and global_words<=0)):
        raise ValueError('Explicit whole-source target/context/word counts and positive own-reference strength required')
    groups=parameter_groups(model);active_names=['language'] if kind=='text' else ['vision']+(['word'] if kind=='word' else [])
    active=[p for name in active_names for _,p in groups[name]];active_ids={id(p) for p in active}
    inactive=[p for name,rows in groups.items() if name not in active_names for _,p in rows]
    if any(p.grad is not None for p in inactive):raise ValueError('No stale cross-family gradients may enter protected learning')
    if kind=='text' and (type(reference) is not OwnDenseAssistant or reference.training
                         or any(p.requires_grad for p in reference.parameters()) or reference.counts()!=model.counts()):
        raise ValueError('A separately counted frozen eval-mode copy of our own exact policy required for text anchoring')
    flags=[(p,p.requires_grad) for p in model.parameters()];amp=amp or nullcontext
    for p,flag in flags:p.requires_grad_(id(p) in active_ids)
    result=dict(CE_sum=0.,full_context_reference_KL_sum=0.,CTC_word_sum=0.,active_groups=active_names,
                reference_parameters=reference.counts()['total_unique'] if kind=='text' else 0,
                training_only_reference_not_exported=True,production_recipe_or_quality_established=False)
    try:
        if kind=='word':
            with amp():word_logits=model.word_logits(batch['word_pixels'])
            ctc,_=ctc_word_sum(word_logits,batch['texts']);(ctc/global_words).backward(inputs=active)
            result['CTC_word_sum']=float(ctc.detach());del ctc,word_logits
        inputs={k:batch[k] for k in ('input_ids','attention_mask','pixel_values') if k in batch}
        with amp():logits=model(**inputs)['logits']
        target=batch['labels'][:,1:];mask=target!=-100
        ce=F.cross_entropy(logits[:,:-1].float()[mask],target[mask],reduction='sum');loss=ce/global_targets
        if kind=='text':
            with torch.no_grad(),amp():base=reference(**inputs)['logits']
            context=batch.get('attention_mask',torch.ones_like(batch['input_ids'],dtype=torch.bool))[:,1:].bool()
            logp=logits[:,:-1].float()[context].log_softmax(-1)
            logq=base[:,:-1].float()[context].log_softmax(-1)
            KL=(logq.exp()*(logq-logp)).sum();loss=loss+reference_KL_weight*KL/global_context_targets
            result['full_context_reference_KL_sum']=float(KL.detach())
        if not bool(torch.isfinite(loss)):raise ValueError('Protected actual source objective must be finite')
        loss.backward(inputs=active);result['CE_sum']=float(ce.detach())
        if any(p.grad is not None for p in inactive):raise ValueError('Protected family leaked gradients into an inactive group')
    finally:
        for p,flag in flags:p.requires_grad_(flag)
    return result
