"""Whole-source global optimizer steps for the protected joint objective."""
import math

import torch
from torch import distributed as dist

from .fast_joint_stage_v1 import parameter_groups,learning_factor,reduce_gradients
from .protected_joint_stage_v1 import backward_protected


def update(model,optimizer,data,schedule,step,options,*,reference,device='cpu',rank=0,world=1,amp=None,prepared=None):
    if (data.role!='train' or data.purpose!='optimization' or options['ctc_weight']!=1 or options['word_native_CE_weight']!=1
            or not 0<options['reference_KL_weight']<=1):raise ValueError('Train-only complete-source protected objective required')
    name,positions=schedule.at(step);local=positions[rank::world];kind=data.kinds[name]
    targets=sum(data.targets[name][p] for p in positions);contexts=sum(data.lengths[name][p]-1 for p in positions)
    words=len(positions) if kind=='word' else 0
    factor=learning_factor(step,len(schedule),options['warmup_fraction'],options['cosine_floor'])
    for group in optimizer.param_groups:group['lr']=options['learning_rates'][group['name']]*factor
    model.train();model.network.gradient_checkpointing=True;optimizer.zero_grad(set_to_none=True)
    groups=parameter_groups(model);parameters=[p for rows in groups.values() for _,p in rows];selected={id(p) for p in parameters}
    totals=dict(CE_sum=0.,CTC_word_sum=0.,response_EOS_targets=0.,words=0.,full_context_reference_KL_sum=0.)
    length=max(data.lengths[name][p] for p in local) if local else 1
    prefix=model.config.image_token_count(options['image_size']) if kind in ('word','image') else 0
    width=max(1,min(options['maximum_microbatch'],options['maximum_microbatch_tokens']//(length+prefix)))
    if prepared is not None and len(prepared)!=math.ceil(len(local)/width):raise ValueError('Exact complete prepared source batches required')
    try:
        for i,start in enumerate(range(0,len(local),width)):
            batch=data.batch(name,local[start:start+width],device) if prepared is None else prepared[i]
            if prepared is not None:batch={k:(v.to(device,non_blocking=True) if isinstance(v,torch.Tensor) else v) for k,v in batch.items()}
            metric=backward_protected(model,batch,kind=kind,global_targets=targets,global_context_targets=contexts,
                global_words=words,reference=reference,reference_KL_weight=options['reference_KL_weight'],amp=amp)
            for key in ['CE_sum','CTC_word_sum','full_context_reference_KL_sum']:totals[key]+=metric[key]
            totals['response_EOS_targets']+=int((batch['labels'][:,1:]!=-100).sum());totals['words']+=len(batch.get('texts',[]))
        if totals['response_EOS_targets']!=sum(data.targets[name][p] for p in local):raise ValueError('Every complete original target must reach backward')
        active=reduce_gradients(parameters,device,world,options['gradient_bucket_values'])
        wanted=['language'] if kind=='text' else ['vision']+(['word'] if kind=='word' else [])
        if (any(p.grad is not None for n,rows in groups.items() if n not in wanted for _,p in rows)
                or any(p.grad is not None for p in reference.parameters())
                or any(p.grad is not None for p in model.parameters() if id(p) not in selected)
                or any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for n in wanted for _,p in groups[n])):
            raise ValueError('Only the active family groups may receive complete finite gradients')
        norm=float(torch.nn.utils.clip_grad_norm_(parameters,options['gradient_clip'],error_if_nonfinite=True))
        optimizer.step();optimizer.zero_grad(set_to_none=True)
    except BaseException:optimizer.zero_grad(set_to_none=True);raise
    metrics=torch.tensor(list(totals.values()),dtype=torch.float64,device=device)
    if world>1:dist.all_reduce(metrics,op=dist.ReduceOp.SUM)
    combined=dict(zip(totals,metrics.cpu().tolist()))
    return dict(step=step+1,family=name,global_original_positions=positions,global_examples=len(positions),
        complete_context_tokens=sum(data.lengths[name][p] for p in positions),**combined,
        token_mean_CE=combined['CE_sum']/targets,mean_byte_normalized_CTC=combined['CTC_word_sum']/words if words else None,
        full_context_reference_KL_mean=combined['full_context_reference_KL_sum']/contexts if kind=='text' else None,
        actual_active_groups=wanted,reference_KL_weight=options['reference_KL_weight'],
        resident_training_only_reference_parameters=reference.counts()['total_unique'],
        reference_forward_queried=kind=='text',gradient_norm_before_clip=norm,active_parameter_tensors=active,
        learning_rates={g['name']:g['lr'] for g in optimizer.param_groups})
