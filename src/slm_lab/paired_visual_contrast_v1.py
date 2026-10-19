"""Whole-answer CE plus a bidirectional image contrast on original paired VQA.

Every pair keeps both proper-image CE targets. Only opposite yes/no answers
with >=9/10 exact human votes on both distinct photos/pixels add contrast.
Identical question/answer tokens are compared across the two images. This is
an unproven source-derived grounding objective, not human preference RLHF,
DPO, a reproduction of prior work, or a demonstrated model improvement.
"""
from dataclasses import asdict
import hashlib
import math

import torch
from torch import distributed as dist
from torch import nn

from .code_residual_collective_checkpoint_v2 import _digest,_same,_validated,fingerprint


def pixel_digest(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def eligible(example):
    return (set(example['answers'])=={'yes','no'} and example['answer_types']==['yes/no','yes/no']
        and all(n>=9 for n in example['human_votes']) and len(set(example['photo_ids']))==2
        and len(set(example['pixel_sha256']))==2)


def validate(examples,config,contract_sha256):
    if not examples or not _digest(contract_sha256):raise ValueError('Complete nonempty externally bound paired source required')
    result=[]
    for e in examples:
        if (e.get('split')!='train' or e.get('source')!='original_paired_VQA_human_answers'
                or e.get('fully_finalized_source') is not True or e.get('source_contract_sha256')!=contract_sha256
                or e.get('human_preferences') is not False or e.get('external_model_teacher_used') is not False):
            raise ValueError('Only finalized original train-role human-answer sources may supply gradients')
        if not isinstance(e.get('id'),str) or not e['id'] or not isinstance(e.get('question'),str) or not e['question']:
            raise ValueError('Complete original paired question/source identities required')
        if (any(not isinstance(e.get(k),list) or len(e[k])!=2 for k in ['answers','answer_types','human_votes','photo_ids','pixel_sha256','sequences','images'])
                or any(not isinstance(v,str) or not v for k in ['answers','answer_types','photo_ids'] for v in e[k])
                or e['answers'][0]==e['answers'][1] or any(type(n) is not int or not 6<=n<=10 for n in e['human_votes'])):
            raise ValueError('Both original distinct answers/photos and exact publisher human votes required')
        counts=[];prefixes=[];bodies=[]
        for ids,labels in e['sequences']:
            if (not isinstance(ids,list) or not isinstance(labels,list) or len(ids)!=len(labels)
                    or not 2<=len(ids)<=config.max_text_length or ids[0]!=1 or ids[-1]!=2
                    or any(type(v) is not int or not 0<=v<config.vocab_size for v in ids)
                    or any(type(v) is not int for v in labels)):
                raise ValueError('Full own-BPE question/answer/EOS must fit without clipping')
            start=next((i for i,v in enumerate(labels) if v!=-100),len(ids))
            if not 1<=start<len(ids) or labels!=[-100]*start+ids[start:]:raise ValueError('Complete contiguous answer/EOS mask required')
            counts.append(len(ids)-start);prefixes.append(ids[:start]);bodies.append(ids[start:])
        if prefixes[0]!=prefixes[1] or bodies[0]==bodies[1]:raise ValueError('Identical full question and distinct original answers required')
        for image,digest in zip(e['images'],e['pixel_sha256']):
            if (not isinstance(image,torch.Tensor) or image.ndim!=3 or image.shape[0]!=3 or image.shape[1]!=image.shape[2]
                    or image.dtype not in {torch.float32,torch.float64} or not bool(torch.isfinite(image).all())
                    or not _digest(digest) or pixel_digest(image)!=digest):raise ValueError('Exact finite original decoded image pixels required')
            config.image_token_count(image.shape[1])
        if e['images'][0].shape!=e['images'][1].shape or e['images'][0].dtype!=e['images'][1].dtype:
            raise ValueError('Image contrast changes pixels while preserving resolution, dtype and prefix positions')
        result.append(dict(targets=counts,contrast_eligible=eligible(e)))
    return result


def gradients(model,examples,*,contract_sha256,alpha,beta,margin,world,device,amp):
    """Accumulate exact streamed gradients; caller averages workers once and steps.

    CE is normalized over all proper response/EOS targets globally. Contrast is
    normalized over two directions per eligible pair exposure globally. Every
    original pair, including lower-consensus/nonbinary ones, keeps proper CE.
    No optimizer step, zero_grad, source admission or automatic routing occurs.
    """
    if type(world) is not int or world not in {1,4} or (world==1 and dist.is_initialized()) or (world==4 and
            (not dist.is_initialized() or dist.get_world_size()!=4)):
        raise ValueError('Explicit standalone or actual four-worker gradient layout required')
    def checked(fn):return _validated(fn) if world==4 else fn()
    def preflight():
        if (any(type(x) not in {int,float} or not math.isfinite(x) for x in [alpha,beta,margin])
                or not 0<=alpha<=1 or not 0<beta<=100 or not 0<=margin<=10
                or any(isinstance(m,nn.Dropout) and m.p for m in model.modules())):
            raise ValueError('Finite explicit paired objective and deterministic policy required')
        return validate(examples,model.config,contract_sha256)
    layout=checked(preflight)
    if world==4:_same(fingerprint(dict(contract=contract_sha256,alpha=alpha,beta=beta,margin=margin,config=asdict(model.config))))
    totals=torch.tensor([sum(sum(r['targets']) for r in layout),sum(r['contrast_eligible'] for r in layout)],device=device,dtype=torch.long)
    if world==4:dist.all_reduce(totals)
    tokens,selected=map(int,totals)
    def loss(e,answer,image):
        ids,labels=e['sequences'][answer]
        with amp():value=model(input_ids=torch.tensor([ids],device=device),labels=torch.tensor([labels],device=device),
            pixel_values=e['images'][image].to(device).unsqueeze(0))['loss']
        if value.ndim!=0 or not bool(torch.isfinite(value)):raise ValueError('Finite complete original image/answer NLL required')
        return value
    def measure():
        rows=[]
        with torch.no_grad():
            for e,r in zip(examples,layout):
                values=[float(loss(e,0,0)),float(loss(e,1,1))]
                wrong=[float(loss(e,0,1)),float(loss(e,1,0))] if r['contrast_eligible'] else None
                rows.append(dict(id=e['id'],targets=r['targets'],contrast_eligible=r['contrast_eligible'],proper_nll=values,other_image_nll=wrong))
        return rows
    rows=checked(measure)
    ce=sum(sum(n*v for n,v in zip(r['targets'],r['proper_nll'])) for r in rows)
    contrast=sum(sum(max(x,0.)+math.log1p(math.exp(-abs(x))) for x in [margin+beta*(a-b) for a,b in zip(r['proper_nll'],r['other_image_nll'])])
        for r in rows if r['contrast_eligible'])
    values=torch.tensor([ce,contrast],device=device,dtype=torch.float64)
    if world==4:dist.all_reduce(values)
    def backward():
        for e,r in zip(examples,rows):
            for answer,n in enumerate(r['targets']):
                coefficient=0.
                if r['contrast_eligible']:
                    x=margin+beta*(r['proper_nll'][answer]-r['other_image_nll'][answer])
                    sig=1/(1+math.exp(-x)) if x>=0 else math.exp(x)/(1+math.exp(x))
                    coefficient=alpha*beta*sig*world/(2*selected)
                (loss(e,answer,answer)*(world*n/tokens+coefficient)).backward()
                if coefficient:(loss(e,answer,1-answer)*(-coefficient)).backward()
    checked(backward)
    pieces=[None]*world
    if world==4:dist.all_gather_object(pieces,rows)
    else:pieces=[rows]
    all_rows=[dict(rank=i,local_exposure=j,**r) for i,part in enumerate(pieces) for j,r in enumerate(part)]
    ce=float(values[0])/tokens;contrast=alpha*float(values[1])/(2*selected) if selected else 0.
    return dict(global_proper_target_tokens=tokens,global_eligible_pair_exposures=selected,global_pair_exposures=len(all_rows),
        token_mean_proper_CE=ce,weighted_image_contrast=contrast,summed_objective=ce+contrast,
        alpha=alpha,beta=beta,margin=margin,records=all_rows,all_original_proper_CE_examples_retained=True,
        human_preferences=False,model_capability_claim=False)
