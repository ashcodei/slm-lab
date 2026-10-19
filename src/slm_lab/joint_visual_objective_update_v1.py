"""One optimizer step for the three complete visual training objectives.

This is a mathematical integration primitive, not a training controller or
data admission. Its caller must bind admitted original sources, the complete
schedule, ownership and private recovery before any production use.
"""
import hashlib
import json
import math

import torch
from torch import distributed as dist

from .fast_joint_stage_v1 import reduce_gradients
from .global_native_visual_stage_kernel_v1 import accumulate_global_native_visual
from .global_photo_gradient_cache_stage_v2 import accumulate_global_photo_cached
from .global_word_microbatch_stage_v2 import accumulate_global_word_microbatched
from .own_actor_vision_prefix_split_backward_v1 import OwnActorVisionPrefixSplitBackward

FAMILIES={'images:additional_photo_captions':'photo','words':'word',
    'images:additional_real_words':'native','images:additional_commons':'native',
    'images:original_visual_reasoning':'native','images:human_photo_reading':'native',
    'images:own_context_photo_reading':'native','replay:generic_visual':'native','replay:image_reading':'native'}


def named_groups(view):
    if type(view) is not OwnActorVisionPrefixSplitBackward:raise ValueError('Exactly one original own visual policy view required')
    view.assert_shared()
    return dict(vision=[('vision.'+n,p) for n,p in view.actor.base.network.vision.named_parameters()],
        pointing=[('pointing.'+n,p) for n,p in view.actor.base.heads['pointing'].named_parameters()],
        word=[('word.'+n,p) for n,p in view.actor.base.heads['word'].named_parameters()])


def make_optimizer(view,*,learning_rates,betas=(.9,.95),eps=1e-5,weight_decay=.01):
    groups=named_groups(view)
    if (set(learning_rates)!=set(groups) or any(type(v) not in [int,float] or not math.isfinite(v) or v<=0 for v in learning_rates.values())
            or type(eps) not in [int,float] or not math.isfinite(eps) or eps<=0
            or type(weight_decay) not in [int,float] or not math.isfinite(weight_decay) or weight_decay<0
            or len(betas)!=2 or any(type(v) not in [int,float] or not math.isfinite(v) or not 0<=v<1 for v in betas)):
        raise ValueError('Complete finite ordinary grouped AdamW settings required')
    # Never change any existing parameter's trainability metadata.
    return torch.optim.AdamW([dict(params=[p for _,p in rows],name=name,lr=learning_rates[name]) for name,rows in groups.items()],
        betas=tuple(betas),eps=eps,weight_decay=weight_decay,foreach=False,fused=False)


def optimizer_groups(view,optimizer):
    expected=named_groups(view)
    if type(optimizer) is not torch.optim.AdamW or len(optimizer.param_groups)!=3:
        raise ValueError('One exact three-group ordinary AdamW optimizer required')
    for group,(name,rows) in zip(optimizer.param_groups,expected.items()):
        if (group.get('name')!=name or [id(p) for p in group['params']]!=[id(p) for _,p in rows]
                or group.get('foreach') is not False or group.get('fused') not in [False,None]
                or any(group.get(k,False) for k in ['amsgrad','maximize','capturable','differentiable'])):
            raise ValueError('Exact original vision/pointing/word Adam parameter order and execution required')
    return expected


def optimizer_clocks(view,optimizer):
    result={}
    for name,rows in optimizer_groups(view,optimizer).items():
        clocks=[float(optimizer.state[p]['step']) if p in optimizer.state else 0. for _,p in rows]
        if any(not math.isfinite(v) or v<0 or v!=int(v) for v in clocks) or len(set(clocks))!=1:
            raise ValueError('Every leaf in a family group must have one finite nonnegative integral Adam clock')
        result[name]=int(clocks[0])
    return result


def update(view,optimizer,prepared,source,options,*,control_group=None):
    initialized=dist.is_initialized()
    if initialized and dist.get_backend(control_group)!='gloo':raise ValueError('Actual explicit Gloo control group required')
    if not initialized and control_group is not None:raise ValueError('A serial integration call cannot use an uninitialized group')
    world=dist.get_world_size(control_group) if initialized else 1;rank=dist.get_rank(control_group) if initialized else 0
    def gather(value):
        if world==1:return [value]
        result=[None]*world;dist.all_gather_object(result,value,group=control_group);return result
    error=specification=None
    try:
        groups=optimizer_groups(view,optimizer);device=next(view.parameters()).device
        if (not 1<=world<=4 or initialized and dist.get_world_size()!=world or any(p.grad is not None for p in view.parameters())
                or any(p.device!=device or p.dtype!=torch.float32 for p in view.parameters())
                or source['family'] not in FAMILIES or set(options)!={'precision','maximum_examples','maximum_padded_tokens',
                    'matching_weight','temperature','CTC_weight','native_word_CE_weight','gradient_clip','gradient_bucket_values'}
                or options['precision'] not in ['fp32','bf16']
                or any(type(options[k]) not in [int,float] or not math.isfinite(options[k]) or options[k]<=0
                    for k in ['matching_weight','temperature','CTC_weight','native_word_CE_weight','gradient_clip'])
                or type(options['gradient_bucket_values']) is not int or options['gradient_bucket_values']<1):
            raise ValueError('One clear-gradient own FP32 replica and complete finite explicit objective settings required')
        kind=FAMILIES[source['family']]
        keys={'family','global_positions','global_context_lengths','global_response_EOS_targets'}
        keys|={'pixel_keys','captions'} if kind=='photo' else {'global_word_UTF8_bytes','global_word_UTF8_sha256'} if kind=='word' else set()
        if set(source)!=keys:raise ValueError('Exact complete objective-specific original source specification required')
        positions=source['global_positions'];local=positions[rank::world]
        if local:
            if not isinstance(prepared,dict) or set(prepared)!={'native','word','matching'}:
                raise ValueError('Exactly the actual native/word/matching tensor payload required')
            if kind=='word':
                if prepared['matching'] is not None or prepared['word'] is None:raise ValueError('Whole word inputs and no fabricated matching branch required')
            elif kind=='photo':
                if prepared['word'] is not None or prepared['matching'] is None:raise ValueError('Whole paired matching captions and no fabricated word branch required')
                matching=prepared['matching']
                if matching['pixel_keys']!=source['pixel_keys'][rank::world] or matching['captions']!=source['captions'][rank::world]:
                    raise ValueError('Every matching feature must retain its exact local original pixel/caption association')
            elif prepared['word'] is not None or prepared['matching'] is not None:
                raise ValueError('Generic native CE must not fabricate a different modality branch')
        elif prepared is not None:raise ValueError('An empty rank must not fabricate any source payload')
        named=[item for rows in groups.values() for item in rows];selected=[p for _,p in named];selected_ids={id(p) for p in selected}
        if len(selected_ids)!=len(selected) or any(not p.requires_grad for p in selected):
            raise ValueError('Exactly the existing distinct trainable vision/pointing/word leaves required')
        wanted=['vision','pointing'] if kind=='photo' else ['vision','word'] if kind=='word' else ['vision']
        inactive=[p for name,rows in groups.items() if name not in wanted for _,p in rows]
        fixed=[p for p in view.parameters() if id(p) not in selected_ids]
        unchanged=fixed+inactive
        identity=lambda p:(id(p),p._version,p.data_ptr(),p.requires_grad)
        before=[identity(p) for p in unchanged]
        inactive_states=[(id(p),tuple((k,id(v),v._version) for k,v in optimizer.state.get(p,{}).items())) for p in inactive]
        clocks=optimizer_clocks(view,optimizer)
        settings=[{k:v for k,v in group.items() if k!='params'} for group in optimizer.param_groups]
        specification=hashlib.sha256(json.dumps(dict(source=source,options=options,optimizer=settings),sort_keys=True).encode()).hexdigest()
    except Exception as exc:error=type(exc).__name__+': '+str(exc)
    errors=gather(error)
    if any(value is not None for value in errors):raise ValueError('Coordinated joint visual preflight rejection: '+repr(errors))
    if any(value!=specification for value in gather(specification)):
        raise ValueError('Coordinated joint visual global family/source/settings disagreement')
    view.train();view.actor.base.train();view.actor.base.network.gradient_checkpointing=True;view.native.gradient_checkpointing=True
    dtype=torch.bfloat16 if options['precision']=='bf16' else torch.float32
    amp=lambda:torch.autocast(device.type,dtype=torch.bfloat16,enabled=options['precision']=='bf16')
    arguments={k:v for k,v in source.items() if k!='family'}
    arguments.update(local_positions=local,maximum_examples=options['maximum_examples'],maximum_padded_tokens=options['maximum_padded_tokens'],amp=amp,group=control_group)
    native=prepared['native'] if prepared is not None else None
    # Optimizer boundaries invalidate any enclosing autocast's old weight
    # casts. This affects only this thread's derived casts, never weights.
    torch.clear_autocast_cache()
    try:
        with torch.autocast(device.type,enabled=False):
            if kind=='photo':
                matching=prepared['matching'] if prepared is not None else None
                record=accumulate_global_photo_cached(view,native,matching['text_ids'] if matching else None,
                    matching['text_mask'] if matching else None,matching_weight=options['matching_weight'],temperature=options['temperature'],
                    feature_dtype=dtype,native_logits_dtype=dtype,**arguments)
                ce=record['local_caption_CE_sum'];ctc=0.;observed=record['local_original_response_EOS_targets'];words=0
            elif kind=='word':
                word=prepared['word'] if prepared is not None else None
                batch={**native,'word_pixels':word['word_pixels'],'texts':word['texts']} if prepared is not None else None
                record=accumulate_global_word_microbatched(view,batch,CTC_weight=options['CTC_weight'],
                    native_CE_weight=options['native_word_CE_weight'],native_logits_dtype=dtype,**arguments)
                ce=record['local_native_CE_sum'];ctc=record['local_byte_normalized_CTC_sum'];observed=record['local_response_EOS_targets'];words=record['local_words']
            else:
                record=accumulate_global_native_visual(view,native,native_logits_dtype=dtype,**arguments)
                ce=record['local_native_CE_sum'];ctc=0.;observed=record['local_response_EOS_targets'];words=0
            active=reduce_gradients(selected,device,world,options['gradient_bucket_values'])
            if (active!=sum(len(groups[name]) for name in wanted)
                    or any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for name in wanted for _,p in groups[name])
                    or any(p.grad is not None for p in fixed+inactive)):
                raise ValueError('Only complete active objective groups may have finite globally summed gradients')
            totals=torch.tensor([ce,ctc,observed,words],dtype=torch.float64,device=device)
            if world>1:dist.all_reduce(totals,op=dist.ReduceOp.SUM)
            ce,ctc,observed,words=totals.cpu().tolist()
            if observed!=sum(source['global_response_EOS_targets']) or words!=(len(positions) if kind=='word' else 0):
                raise ValueError('Every original global native EOS target and CTC word must reach one optimizer commit')
            norm=float(torch.nn.utils.clip_grad_norm_(selected,options['gradient_clip'],error_if_nonfinite=True))
            optimizer.step();optimizer.zero_grad(set_to_none=True)
        if (before!=[identity(p) for p in unchanged]
                or inactive_states!=[(id(p),tuple((k,id(v),v._version) for k,v in optimizer.state.get(p,{}).items())) for p in inactive]):
            raise ValueError('Inactive weights, Adam clocks/moments and every fixed parameter must stay unchanged')
        after=optimizer_clocks(view,optimizer)
        if after!={name:clock+int(name in wanted) for name,clock in clocks.items()}:
            raise ValueError('Each active leaf clock advances once; every inactive leaf clock stays fixed')
        return dict(family=source['family'],kind=kind,global_positions=positions,
            global_context_tokens=sum(source['global_context_lengths']),response_EOS_targets=int(observed),
            global_words=int(words),native_CE_sum=ce,byte_normalized_CTC_sum=ctc,active_groups=wanted,
            active_parameter_tensors=active,gradient_norm_before_clip=norm,optimizer_group_clocks=after,
            one_global_gradient_SUM_and_one_optimizer_commit=True,autocast_weight_cache_cleared_at_optimizer_boundaries=True,
            all_inactive_weights_moments_clocks_and_fixed_model_metadata_preserved=True,local_objective_record=record,
            full_source_admission_and_full_size_GPU_capacity_recovery_quality_not_proved=True)
    except BaseException:
        optimizer.zero_grad(set_to_none=True)
        raise
    finally:torch.clear_autocast_cache()
