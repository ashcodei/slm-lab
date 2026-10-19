"""Teacher-free source-pair matching experiment; not correctness or human RLHF.

The paired code is original public source. A mismatched question is another
same-language question from a different source problem. This is a source
association label, not proof that the code cannot solve that other problem.
"""
from bisect import bisect_right
from collections import defaultdict
import math
import random

import numpy as np
import torch


def pair_map(records, seed):
    """Address every original record; mark unpairable rows rather than drop them."""
    if type(seed) is not int or not records:
        raise ValueError('Nonempty exact source population and integer seed required')
    roles={r['role'] for r in records}
    if len(roles)!=1 or not roles <= {'train','validation'}:
        raise ValueError('One sealed train or validation role required')
    if len({r['id'] for r in records})!=len(records):
        raise ValueError('Unique original source identities required')
    groups=defaultdict(list)
    for i,r in enumerate(records):
        if (not r['group_id'] or not r['language'] or not r['body_sha256'] or not r['prompt_sha256']
                or type(r['prefix_tokens']) is not int or type(r['target_tokens']) is not int
                or r['prefix_tokens']<1 or r['target_tokens']<1
                or r['prefix_tokens']+r['target_tokens']>4096):
            raise ValueError('Every original complete source framing/population required')
        groups[r['language']].append(i)
    pools={lang:sorted(ids,key=lambda i:(records[i]['prefix_tokens'],records[i]['id'])) for lang,ids in groups.items()}
    lengths={lang:[records[i]['prefix_tokens'] for i in ids] for lang,ids in pools.items()}
    result=np.full(len(records),-1,dtype='<i4')
    for i,r in enumerate(records):
        ids=pools[r['language']]
        eligible=bisect_right(lengths[r['language']],4096-r['target_tokens'])
        if not eligible:
            continue
        def different(j):
            other=records[j]
            return (other['group_id']!=r['group_id'] and other['body_sha256']!=r['body_sha256']
                and other['prompt_sha256']!=r['prompt_sha256'])
        rng=random.Random(seed+i)
        candidates=[ids[rng.randrange(eligible)] for _ in range(32)]
        chosen=next((j for j in candidates if different(j)),None)
        if chosen is None:
            chosen=next((j for j in ids[:eligible] if different(j)),None)
        if chosen is not None:
            result[i]=chosen
    return result


def validate_pair_map(records, pairs):
    if pairs.dtype!=np.dtype('<i4') or pairs.shape!=(len(records),):
        raise ValueError('Exact full-population original pair index required')
    for i,value in enumerate(pairs):
        j=int(value)
        if j==-1:
            continue
        if not 0<=j<len(records):
            raise ValueError('Counterfactual source role index outside population')
        a,b=records[i],records[j]
        if (a['role']!=b['role'] or a['language']!=b['language'] or a['group_id']==b['group_id']
                or a['body_sha256']==b['body_sha256'] or a['prompt_sha256']==b['prompt_sha256']
                or b['prefix_tokens']+a['target_tokens']>4096):
            raise ValueError('Distinct same-role/language complete original prompt pairing required')


def mismatch_sequence(positive, negative):
    """Replace only the complete original question; preserve every code/EOS label."""
    ids,labels=positive
    other,other_labels=negative
    if len(ids)!=len(labels) or len(other)!=len(other_labels):
        raise ValueError('Complete original sequence and mask lengths required')
    start=next((i for i,v in enumerate(labels) if v!=-100),len(ids))
    other_start=next((i for i,v in enumerate(other_labels) if v!=-100),len(other))
    if (not 1<=start<len(ids) or not 1<=other_start<len(other)
            or ids[0]!=1 or other[0]!=1 or ids[-1]!=2 or other[-1]!=2
            or labels!=[-100]*start+ids[start:] or other_labels!=[-100]*other_start+other[other_start:]):
        raise ValueError('Complete own-BPE BOS/question/original code/EOS framing required')
    merged=other[:other_start]+ids[start:]
    if len(merged)>4096:
        raise ValueError('Whole mismatched prompt and original code must fit without clipping')
    return merged,[-100]*other_start+ids[start:]


def _validate_examples(examples, training):
    if not examples:
        raise ValueError('Nonempty original examples required')
    targets=[]
    for e in examples:
        if e['role'] not in ({'train'} if training else {'train','validation'}):
            raise ValueError('Held roles cannot enter training gradients')
        ids,labels=e['positive']
        if len(ids)!=len(labels) or ids[0]!=1 or ids[-1]!=2 or len(ids)>4096:
            raise ValueError('Full original causal programme framing required')
        start=next((i for i,y in enumerate(labels) if y!=-100),len(labels))
        if not 1<=start<len(ids) or labels!=[-100]*start+ids[start:]:
            raise ValueError('Complete contiguous original response/EOS mask required')
        n=len(ids)-start
        if e['negative'] is not None:
            other,other_labels=e['negative']
            other_start=next((i for i,y in enumerate(other_labels) if y!=-100),len(other_labels))
            if (not 1<=other_start<len(other) or len(other)!=len(other_labels) or len(other)>4096
                    or other[0]!=1 or other_labels!=[-100]*other_start+other[other_start:]
                    or other[other_start:]!=ids[start:]):
                raise ValueError('Mismatched input must retain every original code target/EOS')
        targets.append(n)
    return targets


def _loss(model, sequence, device, amp):
    ids,labels=sequence
    with amp():
        loss=model(input_ids=torch.tensor([ids],device=device),labels=torch.tensor([labels],device=device))['loss']
    if loss.ndim!=0 or not bool(torch.isfinite(loss)):
        raise ValueError('Finite full original sequence mean NLL required')
    return loss


def scores(model, examples, device, amp):
    _validate_examples(examples,False)
    with torch.no_grad():
        return [dict(matched=float(_loss(model,e['positive'],device,amp)),
            mismatched=None if e['negative'] is None else float(_loss(model,e['negative'],device,amp))) for e in examples]


def matching_gradients(model, examples, *, alpha, beta, margin, device, amp):
    """Exact streaming gradient of CE + weighted softplus(beta*(CE-wrongNLL)+m).

    Detached coefficients are the analytic derivative at the current policy.
    Extra forward passes avoid retaining multiple full4096-token graphs. Both
    correct and mismatched inputs keep every original code/EOS target. This
    requires deterministic forwards; dropout layers are rejected.
    """
    targets=_validate_examples(examples,True)
    if (any(type(v) not in {float,int} or not math.isfinite(v) for v in [alpha,beta,margin])
            or not 0<=alpha<=1 or not 0<beta<=100 or margin<0
            or any(isinstance(m,torch.nn.Dropout) and m.p for m in model.modules())):
        raise ValueError('Finite explicit source-matching objective and deterministic forward required')
    measured=scores(model,examples,device,amp)
    total=sum(targets)
    ce=regularizer=0.
    coefficients=[]
    for e,n,observed in zip(examples,targets,measured):
        weight=n/total
        ce+=weight*observed['matched']
        coefficient=0.
        if observed['mismatched'] is not None:
            x=beta*(observed['matched']-observed['mismatched'])+margin
            softplus=max(x,0.)+math.log1p(math.exp(-abs(x)))
            sigmoid=1/(1+math.exp(-x)) if x>=0 else math.exp(x)/(1+math.exp(x))
            regularizer+=weight*softplus
            coefficient=alpha*beta*sigmoid
        (_loss(model,e['positive'],device,amp)*(weight*(1+coefficient))).backward()
        if coefficient:
            (_loss(model,e['negative'],device,amp)*(-weight*coefficient)).backward()
        coefficients.append(coefficient)
    return dict(positive_source_target_tokens=total,CE=ce,source_pair_softplus=regularizer,
        total_objective=ce+alpha*regularizer,alpha=alpha,beta=beta,margin=margin,
        paired_examples=sum(e['negative'] is not None for e in examples),examples=len(examples),
        matched_and_mismatched_NLL=measured,negative_gradient_coefficients=coefficients,
        correctness_or_human_preference_labels=False)
