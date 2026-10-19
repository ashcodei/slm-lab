"""Full-weight joint CE/byte-CTC with complete distributed family updates.

Each global family batch has one exact target normalizer across all workers.
Inactive vision/word leaves keep grad=None, so neither Adam nor decay changes
them on text-only updates. FP32 bucket sums avoid per-tensor NCCL overhead.
"""
from contextlib import nullcontext
import math

import torch
from torch import distributed as dist
from torch.nn import functional as F

from .own_dense_growth_v1 import OwnDenseAssistant
from .shared_word_stage_v1 import ctc_word_sum

FORMAT='own-full-weight-fast-joint-CE-CTC-v1'


def parameter_groups(model):
    if type(model) is not OwnDenseAssistant:raise ValueError('Actual own counted dense text/image model required')
    groups=dict(language=[],vision=[],word=[])
    for name,p in model.named_parameters():
        if name.startswith('network.vision.'):groups['vision'].append((name,p))
        elif name.startswith('network.') and not name.startswith('network.value_head.'):groups['language'].append((name,p))
        elif name.startswith('heads.word.'):groups['word'].append((name,p))
    return groups


def make_optimizer(model,options,device='cpu'):
    groups=parameter_groups(model)
    selected={id(p) for group in groups.values() for _,p in group}
    for p in model.parameters():p.requires_grad_(id(p) in selected)
    return torch.optim.AdamW([dict(params=[p for _,p in group],name=name,lr=options['learning_rates'][name]) for name,group in groups.items()],
        betas=tuple(options['betas']),weight_decay=options['weight_decay'],foreach=False,fused=torch.device(device).type=='cuda')


def backward_batch(model,batch,*,global_targets,global_words,word=False,amp=None,ctc_weight=1.,native_weight=1.):
    if batch is None:return dict(CE_sum=0.,CTC_word_sum=0.,response_EOS_targets=0,words=0)
    if (global_targets<=0 or (word and global_words<=0) or not all(math.isfinite(x) and x>0 for x in (ctc_weight,native_weight))):
        raise ValueError('Exact complete global target/word denominators and positive objective weights required')
    amp=amp or nullcontext;parameters=[p for group in parameter_groups(model).values() for _,p in group]
    ctc_total=0.
    if word:
        with amp():logits=model.word_logits(batch['word_pixels'])
        ctc,_=ctc_word_sum(logits,batch['texts'])
        (ctc*ctc_weight/global_words).backward(inputs=parameters)
        ctc_total=float(ctc.detach());del ctc,logits
    inputs={k:batch[k] for k in ('input_ids','attention_mask','pixel_values') if k in batch}
    with amp():logits=model(**inputs)['logits']
    target=batch['labels'][:,1:];mask=target!=-100;n=int(mask.sum())
    if not n:raise ValueError('Every complete original local batch must retain real next/responseEOS targets')
    ce=F.cross_entropy(logits[:,:-1].float()[mask],target[mask],reduction='sum')
    if not bool(torch.isfinite(ce)):raise ValueError('Finite actual complete-source CE required')
    (ce*(native_weight if word else 1.)/global_targets).backward(inputs=parameters)
    return dict(CE_sum=float(ce.detach()),CTC_word_sum=ctc_total,response_EOS_targets=n,words=len(batch.get('texts',[])))


def reduce_gradients(parameters,device,world,bucket_values=16*1024**2):
    """Sum already globally normalized gradients, including last-partial ranks."""
    if world==1:return sum(p.grad is not None for p in parameters)
    if not dist.is_initialized() or dist.get_world_size()!=world:raise ValueError('Actual initialized distributed worker count required')
    present=torch.tensor([int(p.grad is not None) for p in parameters],dtype=torch.int32,device=device)
    dist.all_reduce(present,op=dist.ReduceOp.MAX);active=present.cpu().tolist();bucket=[];values=0
    def flush():
        if not bucket:return
        flat=torch.cat([p.grad.reshape(-1) for p in bucket]);dist.all_reduce(flat,op=dist.ReduceOp.SUM);offset=0
        for p in bucket:p.grad.copy_(flat[offset:offset+p.numel()].view_as(p));offset+=p.numel()
        bucket.clear()
    for p,on in zip(parameters,active):
        if not on:continue
        if p.grad is None:p.grad=torch.zeros_like(p)
        if bucket and values+p.numel()>bucket_values:flush();values=0
        bucket.append(p);values+=p.numel()
        if values>=bucket_values:flush();values=0
    flush();return sum(active)


def learning_factor(step,total,warmup_fraction=.02,floor=.1):
    if type(step) is not int or not 0<=step<total or not 0<warmup_fraction<1 or not 0<floor<=1:
        raise ValueError('Explicit bounded whole-source warmup/cosine schedule required')
    warmup=max(1,math.ceil(total*warmup_fraction))
    if step<warmup:return (step+1)/warmup
    return floor+(1-floor)*.5*(1+math.cos(math.pi*(step-warmup)/max(1,total-warmup-1)))


def update(model,optimizer,data,schedule,step,options,*,device='cpu',rank=0,world=1,amp=None,prepared=None):
    if data.role!='train' or data.purpose!='optimization':raise ValueError('Held data cannot supply optimization or capacity gradients')
    name,positions=schedule.at(step);local=positions[rank::world];kind=data.kinds[name]
    targets=sum(data.targets[name][p] for p in positions);words=len(positions) if kind=='word' else 0
    factor=learning_factor(step,len(schedule),options['warmup_fraction'],options['cosine_floor'])
    for group in optimizer.param_groups:group['lr']=options['learning_rates'][group['name']]*factor
    model.train();model.network.gradient_checkpointing=True;optimizer.zero_grad(set_to_none=True)
    parameters=[p for group in parameter_groups(model).values() for _,p in group];selected={id(p) for p in parameters}
    totals=dict(CE_sum=0.,CTC_word_sum=0.,response_EOS_targets=0,words=0)
    # Tokens include the actual image prefix. The cap selects a smaller real
    # microbatch, never a shorter example or synthetic allocation stand-in.
    length=max(data.lengths[name][p] for p in local) if local else 1
    prefix=model.config.image_token_count(options['image_size']) if kind in ('word','image') else 0
    width=max(1,min(options['maximum_microbatch'],options['maximum_microbatch_tokens']//(length+prefix)))
    if prepared is not None and len(prepared)!=math.ceil(len(local)/width):raise ValueError('Exact prefetched original microbatch count required')
    try:
        for i,start in enumerate(range(0,len(local),width)):
            batch=data.batch(name,local[start:start+width],device) if prepared is None else prepared[i]
            if prepared is not None:batch={k:(v.to(device,non_blocking=True) if isinstance(v,torch.Tensor) else v) for k,v in batch.items()}
            record=backward_batch(model,batch,global_targets=targets,global_words=words,word=kind=='word',amp=amp,
                ctc_weight=options['ctc_weight'],native_weight=options['word_native_CE_weight'])
            for key in totals:totals[key]+=record[key]
        if totals['response_EOS_targets']!=sum(data.targets[name][p] for p in local):raise ValueError('Every original local target must reach backward')
        active=reduce_gradients(parameters,device,world,options['gradient_bucket_values'])
        if (not active or any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in parameters)
                or any(p.grad is not None for p in model.parameters() if id(p) not in selected)):
            raise ValueError('Only counted language/vision/word parameters may receive finite gradients')
        norm=float(torch.nn.utils.clip_grad_norm_(parameters,options['gradient_clip'],error_if_nonfinite=True))
        optimizer.step();optimizer.zero_grad(set_to_none=True)
    except BaseException:optimizer.zero_grad(set_to_none=True);raise
    metrics=torch.tensor([totals[k] for k in totals],dtype=torch.float64,device=device)
    if world>1:dist.all_reduce(metrics,op=dist.ReduceOp.SUM)
    combined=dict(zip(totals,metrics.cpu().tolist()))
    return dict(step=step+1,family=name,global_original_positions=positions,global_examples=len(positions),
        complete_context_tokens=sum(data.lengths[name][p] for p in positions),**combined,
        token_mean_CE=combined['CE_sum']/targets,mean_byte_normalized_CTC=combined['CTC_word_sum']/words if words else None,
        gradient_norm_before_clip=norm,active_parameter_tensors=active,learning_rates={g['name']:g['lr'] for g in optimizer.param_groups})
