"""Accumulate complete caption CE and global paired matching on one prefix.

This is a non-optimizer training kernel. The caller binds admitted original
sources and snapshots, SUMs parameter gradients once, clips and commits Adam.
No world average, source duplication or parameter trainability change occurs.
Full-size CUDA capacity/recovery and broad quality admission remain pending.
"""
from contextlib import nullcontext
import hashlib
import json

import torch
from torch import distributed as dist

from .global_visual_matching_gradients_v1 import global_matching_gradients,backward_global_matching
from .own_actor_vision_prefix_reuse_v1 import native_from_visual
from .own_actor_vision_prefix_split_backward_v1 import OwnActorVisionPrefixSplitBackward


def accumulate_global_photo(view,batch,text_ids,text_mask,*,global_positions,local_positions,
                            global_response_EOS_targets,pixel_keys,captions,matching_weight=.1,temperature=.07,
                            feature_dtype=torch.float32,amp=None,group=None):
    distributed = dist.is_initialized()
    if distributed and dist.get_backend(group) != 'gloo':
        raise ValueError('An actual explicit Gloo control/feature group is required')
    world = dist.get_world_size(group) if distributed else 1
    rank = dist.get_rank(group) if distributed else 0
    def gather(value):
        if world == 1:return [value]
        results = [None]*world;dist.all_gather_object(results,value,group=group);return results
    def coordinated(error,phase):
        errors = gather(error)
        if any(e is not None for e in errors):raise ValueError('Coordinated photo '+phase+' rejection: '+repr(errors))
    error = None;specification = None
    try:
        if (type(view) is not OwnActorVisionPrefixSplitBackward or not view.training or not view.native.training
                or feature_dtype not in [torch.float32,torch.bfloat16] or amp is not None and not callable(amp)
                or not isinstance(global_positions,list) or not 2 <= len(global_positions) <= 64
                or len(set(global_positions)) != len(global_positions) or any(type(p) is not int or p < 0 for p in global_positions)
                or local_positions != global_positions[rank::world]
                or len(global_response_EOS_targets) != len(global_positions)
                or any(type(v) is not int or v < 1 for v in global_response_EOS_targets)):
            raise ValueError('Exact trained own view and complete bounded global source/target lists required')
        view.assert_shared()
        if any(p.grad is not None for p in view.parameters()):raise ValueError('Every gradient must start clear')
        selected = [p for _,p in view.named_selected()];selected_ids = {id(p) for p in selected}
        fixed = [p for p in view.parameters() if id(p) not in selected_ids]
        fixed_signature = [(id(p),p._version,p.data_ptr(),p.requires_grad) for p in fixed]
        device = next(view.parameters()).device
        if any(p.dtype != torch.float32 or p.device != device for p in view.parameters()):
            raise ValueError('One consistent original FP32 parameter replica with optional explicit autocast required')
        local_targets = global_response_EOS_targets[rank::world]
        if not local_positions:
            if any(v is not None for v in [batch,text_ids,text_mask]):raise ValueError('Empty ranks must have no fabricated photo/text batch')
        else:
            if (not isinstance(batch,dict) or set(batch) != {'input_ids','attention_mask','labels','pixel_values'}
                    or batch['input_ids'].ndim != 2 or batch['labels'].shape != batch['input_ids'].shape
                    or batch['attention_mask'].shape != batch['input_ids'].shape or batch['attention_mask'].dtype != torch.bool
                    or len(batch['input_ids']) != len(local_positions) or len(batch['pixel_values']) != len(local_positions)
                    or text_ids.ndim != 2 or text_mask.shape != text_ids.shape or text_mask.dtype != torch.bool
                    or len(text_ids) != len(local_positions) or not bool(text_mask.any(1).all())
                    or bool((text_mask[:,1:] & ~text_mask[:,:-1]).any())):
                raise ValueError('Every complete corresponding local photo/native target/matching caption required')
            for ids,labels,mask,wanted in zip(batch['input_ids'],batch['labels'],batch['attention_mask'],local_targets):
                size = int(mask.sum());unmasked = (labels != -100).nonzero().flatten()
                if (size < 2 or not bool(mask[:size].all()) or bool(mask[size:].any()) or int(ids[0]) != 1 or int(ids[size-1]) != 2
                        or not len(unmasked) or int(unmasked[0]) < 1 or int(unmasked[-1]) != size-1
                        or len(unmasked) != wanted or unmasked.tolist() != list(range(int(unmasked[0]),size))
                        or not torch.equal(ids[unmasked],labels[unmasked])):
                    raise ValueError('Whole contiguous original response and genuine final EOS masks must match global counts')
        specification = hashlib.sha256(json.dumps(dict(global_positions=global_positions,
            global_response_EOS_targets=global_response_EOS_targets,feature_dtype=str(feature_dtype),
            pixel_keys=pixel_keys,captions=captions,matching_weight=matching_weight,temperature=temperature),sort_keys=True).encode()).hexdigest()
    except Exception as exc:error = type(exc).__name__+': '+str(exc)
    coordinated(error,'source/native preflight')
    if any(v != specification for v in gather(specification)):
        raise ValueError('Coordinated photo global target/source/settings mismatch')
    amp = amp or nullcontext;caption = None;error = None
    try:
        if not local_positions:
            width = view.actor.base.settings['pointing_rank']
            images = torch.empty(0,width,dtype=feature_dtype,device=device);texts = torch.empty_like(images)
        else:
            content = text_mask & (text_ids != 0) & (text_ids != 1) & (text_ids != 2)
            if not bool(content.any(1).all()):raise ValueError('Whole matching captions need original source content')
            with torch.no_grad(),amp():
                hidden = view.actor.policy(text_ids,attention_mask=text_mask,return_hidden=True,return_logits=False)['hidden']
                text = (hidden.float()*content[:,:,None]).sum(1)/content.sum(1,keepdim=True)
            with amp():
                visual = view.native.vision(batch['pixel_values'],view.native.gradient_checkpointing)
                heads = view.actor.base.heads['pointing']
                images = heads['patch'](visual.mean(1));texts = heads['query'](text.to(visual.dtype))
                caption = native_from_visual(view.native,batch['input_ids'],visual,
                    attention_mask=batch['attention_mask'],labels=batch['labels'])['loss']
            if images.dtype != feature_dtype or texts.dtype != feature_dtype or not bool(torch.isfinite(caption)):
                raise ValueError('Actual feature precision and complete native caption CE must match the declared computation')
    except Exception as exc:error = type(exc).__name__+': '+str(exc)
    coordinated(error,'local forward')
    gradients = global_matching_gradients(images,texts,global_positions=global_positions,local_positions=local_positions,
        pixel_keys=pixel_keys,captions=captions,matching_weight=matching_weight,temperature=temperature,group=group)
    error = None
    try:
        matching = backward_global_matching(images,texts,gradients,parameters=selected,retain_graph=bool(local_positions))
        if local_positions:
            (caption*sum(local_targets)/sum(global_response_EOS_targets)).backward(inputs=selected)
            if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in selected):
                raise ValueError('Every selected photo/matching leaf requires a finite local gradient')
        elif any(p.grad is not None for p in selected):
            raise ValueError('An empty rank must not manufacture any source gradient')
        if (fixed_signature != [(id(p),p._version,p.data_ptr(),p.requires_grad) for p in fixed]
                or any(p.grad is not None for p in fixed)):
            raise ValueError('Every fixed language/actor/critic/other-head value and its metadata must remain unchanged')
    except Exception as exc:error = type(exc).__name__+': '+str(exc)
    coordinated(error,'local backward')
    return dict(rank=rank,world=world,global_positions=global_positions,local_positions=local_positions,
        global_original_response_EOS_targets=sum(global_response_EOS_targets),local_original_response_EOS_targets=sum(local_targets),
        local_caption_CE_sum=float(caption.detach())*sum(local_targets) if caption is not None else 0.,
        global_matching=gradients.record(),actual_image_encoder_forwards=1 if local_positions else 0,
        actual_matching_backward_calls=matching['local_backward_calls'],
        actual_native_caption_backward_calls=1 if local_positions else 0,
        one_parameter_gradient_SUM_still_required=True,optimizer_committed=False,
        complete_original_response_and_EOS_targets_preserved=True,fixed_weights_and_metadata_untouched=True,
        full_size_CUDA_capacity_recovery_and_quality_not_proved=True)
