"""Complete function-repair CE over disjoint replica shards and host SUM.

The objective uses the global original target/weight denominator before any
backward. Empty tail ranks perform no forward and contribute zero gradients.
This is a kernel: owned four-L40 capacity/recovery and scheduling remain gates.
"""
from contextlib import nullcontext
import math

import torch
from torch import distributed as dist

from .code_target_regions_v1 import weighted_CE
from .function_repair_curriculum_v1 import FunctionRepairCurriculum
from .full_function_repair_stage_v1 import PROFILES,CodeWeights,metadata
from .own_dense_actor_v1 import OwnDenseActor
from .own_dense_actor_checkpoint_v1 import selected

FORMAT='own-parallel-complete-function-repair-global-target-CE-v1'


def signature(parameters):return [(id(p),p._version,p.data_ptr(),p.requires_grad) for p in parameters]


def local_backward(model,data,positions,profile,options,weights,*,rank,world,amp=None):
    """Backpropagate one real rank's complete sources, already global-normalized."""
    if (type(model) is not OwnDenseActor or type(data) is not FunctionRepairCurriculum
            or data.role!='train' or data.purpose!='optimization' or profile not in PROFILES
            or type(world) is not int or not 1<=world<=4 or type(rank) is not int or not 0<=rank<world):
        raise ValueError('Exact own actor, admitted optimization-only train sources and explicit replica required')
    info=metadata(data,positions,profile,weights);arm,weighted=PROFILES[profile];slots=positions[rank::world]
    width=max(1,min(options['maximum_microbatch'],options['maximum_microbatch_tokens']//max(data.lengths[p] for p in positions)))
    device=next(model.parameters()).device;parameters=[p for _,p in model.named_selected(('actor',))]
    model.train();model.policy.gradient_checkpointing=True;amp=amp or nullcontext
    weighted_sum=flat_sum=0.;targets=contexts=microbatches=0;precision=[]
    for begin in range(0,len(slots),width):
        local_slots=slots[begin:begin+width]
        batch=data.batch(arm,local_slots,device=device,for_optimization=True)
        target_weights=torch.zeros_like(batch['labels'],dtype=torch.float32)
        for i,slot in enumerate(local_slots):
            mask=batch['labels'][i]!=-100
            target_weights[i,mask]=torch.from_numpy(weights.for_slot(data,slot).copy()).to(device) if weighted else 1.
        with amp():
            logits=model(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'])['logits']
            precision.append(dict(actual_logits_dtype=str(logits.dtype),autocast_enabled=torch.is_autocast_enabled(device.type),
                actual_autocast_dtype=str(torch.get_autocast_dtype(device.type)) if torch.is_autocast_enabled(device.type) else None))
        numerator,_=weighted_CE(logits,batch['labels'],target_weights)
        if not bool(torch.isfinite(numerator)):raise ValueError('Every complete original source objective must remain finite')
        (numerator/info['actual_global_weighted_targets']).backward(inputs=parameters)
        weighted_sum+=float(numerator.detach())
        with torch.no_grad():
            flat,_=weighted_CE(logits.detach(),batch['labels'],(batch['labels']!=-100).float());flat_sum+=float(flat)
        targets+=int((batch['labels'][:,1:]!=-100).sum());contexts+=int(batch['attention_mask'].sum());microbatches+=1
        del logits,numerator,flat,batch,target_weights
    expected_targets=sum(sum(v!=-100 for v in data.native(arm,p)[1]) for p in slots)
    if targets!=expected_targets:raise ValueError('No original positive target or true EOS may be lost on a replica')
    return dict(rank=rank,actual_original_slots=slots,actual_response_EOS_targets=targets,complete_context_tokens=contexts,
        weighted_CE_sum=weighted_sum,original_CE_sum=flat_sum,actual_microbatches=microbatches,
        actual_precision_per_forward=precision,no_forward_for_empty_tail=not slots)


def update(model,optimizer,data,positions,profile,options,weights=None,*,rank=0,world=1,
           control_group=None,gradient_group=None,amp=None):
    if (type(world) is not int or not 1<=world<=4 or type(rank) is not int or not 0<=rank<world
            or world>1 and (not dist.is_initialized() or dist.get_world_size(control_group)!=world
                or dist.get_world_size(gradient_group)!=world or dist.get_rank(control_group)!=rank
                or dist.get_backend(control_group)!='gloo' or dist.get_backend(gradient_group)!='gloo')):
        raise ValueError('Actual matching Gloo replica/control/host-gradient groups required')
    def gather(value):
        if world==1:return [value]
        result=[None]*world;dist.all_gather_object(result,value,group=control_group);return result
    def coordinated(error,phase):
        errors=gather(error)
        if any(e is not None for e in errors):raise ValueError('Coordinated '+phase+' rejection before optimizer commit: '+repr(errors))
    error=None
    try:
        named=selected(model,optimizer,('actor',));parameters=[p for _,p in named];identities={id(p) for p in parameters}
        fixed=[p for p in model.parameters() if id(p) not in identities];before=signature(fixed)
        info=metadata(data,positions,profile,weights);device=next(model.parameters()).device
        if (type(data) is not FunctionRepairCurriculum or data.role!='train' or data.purpose!='optimization'
                or any(p.dtype!=torch.float32 or p.device!=device for p in model.parameters())
                or any(p.grad is not None for p in fixed)
                or device.type=='cuda' and torch.backends.cuda.matmul.allow_tf32
                or optimizer.param_groups[0]['lr']!=options['learning_rate']):
            raise ValueError('Complete own FP32 replicas, selected actor optimizer and admitted train-only sources required')
        specification=dict(info=info,world=world,data_schedule_identity=data.identity([positions]),
            microbatch=options['maximum_microbatch'],microbatch_tokens=options['maximum_microbatch_tokens'],
            gradient_clip=options['gradient_clip'],learning_rate=options['learning_rate'],device_type=device.type,
            Adam={k:v for k,v in optimizer.param_groups[0].items() if k!='params'})
    except Exception as exc:error=type(exc).__name__+': '+str(exc)
    coordinated(error,'preflight')
    if any(item!=specification for item in gather(specification)):
        raise ValueError('Every replica must use the same complete sources, objective, normalization and optimizer settings')
    optimizer.zero_grad(set_to_none=True)
    try:
        error=None
        try:
            local=local_backward(model,data,positions,profile,options,weights,rank=rank,world=world,amp=amp)
            if (signature(fixed)!=before or any(p.grad is not None for p in fixed)
                    or any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in parameters) and local['actual_original_slots']):
                raise ValueError('Every nonempty replica needs only selected finite gradients and unchanged frozen parameters')
            if not local['actual_original_slots']:
                if any(p.grad is not None for p in parameters):raise ValueError('An empty tail rank must not manufacture a source gradient')
                for p in parameters:p.grad=torch.zeros_like(p)
        except Exception as exc:error=type(exc).__name__+': '+str(exc)
        coordinated(error,'complete-source forward/backward')
        locals_=gather(local)
        if ([v['rank'] for v in locals_]!=list(range(world))
                or [v['actual_original_slots'] for v in locals_]!=[positions[r::world] for r in range(world)]
                or sum(v['actual_response_EOS_targets'] for v in locals_)!=info['response_EOS_targets']
                or sum(v['complete_context_tokens'] for v in locals_)!=info['complete_context_tokens']):
            raise ValueError('Every and only original complete source/target must be covered across actual rank shards')
        if world>1:
            flat=torch.cat([p.grad.reshape(-1) for p in parameters]);host=flat.cpu()
            dist.all_reduce(host,op=dist.ReduceOp.SUM,group=gradient_group);flat.copy_(host);offset=0
            for p in parameters:p.grad.copy_(flat[offset:offset+p.numel()].view_as(p));offset+=p.numel()
            del host,flat
        if any(not bool(torch.isfinite(p.grad).all()) for p in parameters):raise ValueError('Actual globally summed selected gradients must be finite')
        norm=float(torch.nn.utils.clip_grad_norm_(parameters,options['gradient_clip'],error_if_nonfinite=True))
        optimizer.step();optimizer.zero_grad(set_to_none=True)
    except BaseException:
        optimizer.zero_grad(set_to_none=True);raise
    if before!=signature(fixed) or any(p.grad is not None for p in fixed):raise ValueError('No base/vision/word/unused critic value may change')
    weighted_sum=math.fsum(v['weighted_CE_sum'] for v in locals_);flat_sum=math.fsum(v['original_CE_sum'] for v in locals_)
    return dict(format=FORMAT,**info,actual_world_size=world,actual_device_type=device.type,
        actual_rank_source_shards=[v['actual_original_slots'] for v in locals_],actual_rank_observations=locals_,
        gradient_transport='Gloo_host_SUM_already_global_normalized' if world>1 else 'serial_global_normalized',
        weighted_CE_sum=weighted_sum,original_CE_sum=flat_sum,
        weighted_token_mean_NLL=weighted_sum/info['actual_global_weighted_targets'],
        original_token_mean_NLL=flat_sum/info['response_EOS_targets'],
        gradient_norm_before_clip=norm,learning_rate=options['learning_rate'],
        every_original_target_and_true_EOS_retained=True,neural_teacher_or_reference_queried=False,
        empty_tail_ranks_never_forward_or_duplicate_examples=True,actual_four_L40_capacity_not_proved_by_CPU_mechanics=True)
