"""Two-pass feature-gradient caching for complete global photo matching.

Adaptation of Gao et al. (2021), https://aclanthology.org/2021.repl4nlp-1.31/ .
First cache small graph-free features, compute the full global pair loss,
then recompute each bounded chunk and apply its feature derivatives plus
whole native caption CE. No training admission, speed or full-size GPU claim.
"""
from contextlib import nullcontext
import hashlib
import json
import math

import torch
from torch import distributed as dist

from .deterministic_positions_v1 import vision_forward
from .global_native_visual_stage_kernel_v1 import native_microbatches
from .global_visual_matching_gradients_v1 import global_matching_gradients, digest, signature
from .own_actor_vision_alignment_v1 import pair_positives
from .own_actor_vision_prefix_reuse_v1 import native_from_visual
from .own_actor_vision_prefix_split_backward_v1 import OwnActorVisionPrefixSplitBackward


def _features(view, native, caption_ids, caption_mask, amp):
    content = caption_mask & (caption_ids != 1) & (caption_ids != 2) & (caption_ids != 0)
    with torch.no_grad(), amp():
        hidden = view.actor.policy(caption_ids, attention_mask=caption_mask, return_hidden=True, return_logits=False)['hidden']
        text = (hidden.float() * content[:, :, None]).sum(1) / content.sum(1, keepdim=True)
    with amp():
        visual = view.native.vision(native['pixel_values'], view.native.gradient_checkpointing)
        heads = view.actor.base.heads['pointing']
        images = heads['patch'](visual.mean(1))
        texts = heads['query'](text.to(visual.dtype))
    return visual, images, texts


def accumulate_global_photo_cached(view, batch, text_ids, text_mask, *, global_positions, local_positions,
        global_context_lengths, global_response_EOS_targets, pixel_keys, captions, maximum_examples=4,
        maximum_padded_tokens=4096, matching_weight=.1, temperature=.07,
        feature_dtype=torch.float32, native_logits_dtype=torch.float32, amp=None, group=None):
    initialized = dist.is_initialized()
    if initialized and dist.get_backend(group) != 'gloo':
        raise ValueError('Actual explicit Gloo control group required')
    if not initialized and group is not None:
        raise ValueError('An uninitialized control group cannot stand in for a serial call')
    world = dist.get_world_size(group) if initialized else 1
    rank = dist.get_rank(group) if initialized else 0

    def gather(value):
        if world == 1:
            return [value]
        result = [None] * world
        dist.all_gather_object(result, value, group=group)
        return result

    def coordinated(error, phase, clear=False):
        errors = gather(error)
        if any(value is not None for value in errors):
            if clear:
                for parameter in selected:
                    parameter.grad = None
            raise ValueError('Coordinated cached photo ' + phase + ' rejection: ' + repr(errors))

    error = specification = None
    try:
        if (type(view) is not OwnActorVisionPrefixSplitBackward or not view.training or not view.native.training
                or not 1 <= world <= 4 or not isinstance(global_positions, list) or not 2 <= len(global_positions) <= 64
                or len(set(global_positions)) != len(global_positions) or any(type(p) is not int or p < 0 for p in global_positions)
                or local_positions != global_positions[rank::world] or len(global_context_lengths) != len(global_positions)
                or len(global_response_EOS_targets) != len(global_positions)
                or any(type(n) is not int or n < 1 for n in global_response_EOS_targets)
                or any(type(n) not in [int,float] or not math.isfinite(n) or n <= 0 for n in [matching_weight, temperature])
                or feature_dtype not in [torch.float32,torch.bfloat16] or native_logits_dtype not in [torch.float32,torch.bfloat16]
                or amp is not None and not callable(amp)
                or len(pixel_keys) != len(global_positions) or len(captions) != len(global_positions)):
            raise ValueError('Exact own view, complete global source/settings lists and explicit finite objective required')
        positive = pair_positives(pixel_keys, captions)
        if bool(positive.all()):
            raise ValueError('Complete global photo batch must retain observed negatives')
        view.assert_shared()
        prefix = view.actor.config.image_token_count(448)
        native_microbatches(global_context_lengths, prefix_tokens=prefix,
            maximum_examples=maximum_examples, maximum_padded_tokens=maximum_padded_tokens)
        if max(global_context_lengths) > view.actor.config.max_text_length:
            raise ValueError('Every full original context must fit the actual own decoder')
        selected = [p for _,p in view.named_selected()]
        chosen = {id(p) for p in selected}
        if any(p.grad is not None for p in view.parameters()) or any(not p.requires_grad for p in selected):
            raise ValueError('All gradients must start clear and all existing selected leaves must be trainable')
        device = next(view.parameters()).device
        if any(p.device != device or p.dtype != torch.float32 for p in view.parameters()):
            raise ValueError('One own FP32-master replica and optional explicit autocast required')
        if device.type == 'cuda' and (getattr(view.native.vision.forward,'__func__',None) is not vision_forward
                or not torch.are_deterministic_algorithms_enabled()
                or torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32):
            raise ValueError('CUDA requires the existing deterministic position operator and precision controls')
        parameters = list(view.parameters())
        before = [(id(p),p._version,p.data_ptr(),p.requires_grad) for p in parameters]
        fixed = [p for p in parameters if id(p) not in chosen]
        local_lengths = global_context_lengths[rank::world]
        local_targets = global_response_EOS_targets[rank::world]
        chunks = native_microbatches(local_lengths, prefix_tokens=prefix,
            maximum_examples=maximum_examples, maximum_padded_tokens=maximum_padded_tokens)
        if not local_positions:
            if any(value is not None for value in [batch,text_ids,text_mask]):
                raise ValueError('Empty ranks must not fabricate photo/native/matching inputs')
        else:
            if (not isinstance(batch,dict) or set(batch) != {'input_ids','labels','attention_mask','pixel_values'}
                    or batch['input_ids'].ndim != 2 or batch['input_ids'].dtype != torch.long
                    or batch['labels'].shape != batch['input_ids'].shape or batch['labels'].dtype != torch.long
                    or batch['attention_mask'].shape != batch['input_ids'].shape or batch['attention_mask'].dtype != torch.bool
                    or len(batch['input_ids']) != len(local_positions)
                    or tuple(batch['pixel_values'].shape) != (len(local_positions),3,448,448)
                    or batch['pixel_values'].dtype != torch.float32 or not bool(torch.isfinite(batch['pixel_values']).all())
                    or bool((batch['pixel_values'].abs() > 1).any()) or any(v.device != device for v in batch.values())
                    or text_ids.ndim != 2 or text_ids.dtype != torch.long or text_mask.shape != text_ids.shape
                    or text_mask.dtype != torch.bool or len(text_ids) != len(local_positions)
                    or text_ids.device != device or text_mask.device != device):
                raise ValueError('All full native448px and corresponding complete caption-only matching inputs required')
            for ids,labels,mask,caption,cmask,length,wanted in zip(batch['input_ids'],batch['labels'],batch['attention_mask'],
                    text_ids,text_mask,local_lengths,local_targets):
                active = (labels != -100).nonzero().flatten()
                text_length = int(cmask.sum())
                if (int(mask.sum()) != length or not bool(mask[:length].all()) or bool(mask[length:].any())
                        or int(ids[0]) != 1 or int(ids[length-1]) != 2 or bool((ids[length:] != 0).any())
                        or bool(((ids[1:length-1] < 4) | (ids[1:length-1] >= view.actor.config.vocab_size)).any())
                        or not len(active) or int(active[0]) < 1 or len(active) != wanted
                        or active.tolist() != list(range(int(active[0]),length)) or not torch.equal(ids[active],labels[active])
                        or text_length < 3 or not bool(cmask[:text_length].all()) or bool(cmask[text_length:].any())
                        or bool((caption[text_length:] != 0).any()) or caption[:text_length].tolist() != [1]+labels[active].tolist()):
                    raise ValueError('Complete original caption/native response and EOS masks must match exactly')
        specification = hashlib.sha256(json.dumps(dict(global_positions=global_positions,global_context_lengths=global_context_lengths,
            global_response_EOS_targets=global_response_EOS_targets,pixel_keys=pixel_keys,captions=captions,
            maximum_examples=maximum_examples,maximum_padded_tokens=maximum_padded_tokens,matching_weight=matching_weight,
            temperature=temperature,feature_dtype=str(feature_dtype),native_logits_dtype=str(native_logits_dtype)),sort_keys=True).encode()).hexdigest()
    except Exception as exc:
        error = type(exc).__name__+': '+str(exc)
    coordinated(error,'source/native preflight')
    if any(value != specification for value in gather(specification)):
        raise ValueError('Coordinated cached photo global source/normalization/settings mismatch')
    amp = amp or nullcontext

    def chunk_inputs(slots):
        width = max(local_lengths[i] for i in slots)
        text_width = int(text_mask[slots].sum(1).max())
        native = {name:value[slots] if name == 'pixel_values' else value[slots,:width] for name,value in batch.items()}
        return native,text_ids[slots,:text_width],text_mask[slots,:text_width]

    cache_calls = replay_calls = 0
    image_parts = []
    text_parts = []
    error = None
    try:
        for slots in chunks:
            native,ids,mask = chunk_inputs(slots)
            with torch.no_grad(),amp():
                # A first no-grad autocast must not seed an enclosing AMP
                # weight cache with detached casts of the trainable encoder.
                # Preserve its active precision while disabling only caching.
                with torch.autocast(device.type,enabled=torch.is_autocast_enabled(device.type),
                        dtype=torch.get_autocast_dtype(device.type),cache_enabled=False):
                    visual,images,texts = _features(view,native,ids,mask,nullcontext)
            if images.dtype != feature_dtype or texts.dtype != feature_dtype:
                raise ValueError('Cache feature precision differs from the declared full objective')
            image_parts.append(images.detach())
            text_parts.append(texts.detach())
            cache_calls += 1
            del visual,images,texts,native,ids,mask
        width = view.actor.base.settings['pointing_rank']
        cached_images = torch.cat(image_parts).detach().requires_grad_(True) if chunks else torch.empty(0,width,device=device,dtype=feature_dtype)
        cached_texts = torch.cat(text_parts).detach().requires_grad_(True) if chunks else torch.empty_like(cached_images)
        del image_parts,text_parts
        if before != [(id(p),p._version,p.data_ptr(),p.requires_grad) for p in parameters] or any(p.grad is not None for p in parameters):
            raise ValueError('Feature caching must not modify a weight, trainability flag or gradient')
    except Exception as exc:
        error = type(exc).__name__+': '+str(exc)
    coordinated(error,'graph-free feature cache',clear=True)
    gradients = global_matching_gradients(cached_images,cached_texts,global_positions=global_positions,local_positions=local_positions,
        pixel_keys=pixel_keys,captions=captions,matching_weight=matching_weight,temperature=temperature,group=group)
    ce_sum = 0.
    observed = 0
    error = None
    try:
        if (signature(cached_images) != gradients.image_signature or signature(cached_texts) != gradients.text_signature
                or digest(gradients.image_gradients) != gradients.image_gradients_sha256
                or digest(gradients.text_gradients) != gradients.text_gradients_sha256
                or tuple(local_positions) != gradients.local_positions):
            raise ValueError('The exact unchanged cached global derivative plan is required')
        for slots in chunks:
            if before != [(id(p),p._version,p.data_ptr(),p.requires_grad) for p in parameters]:
                raise ValueError('No model weight or trainability metadata may change between the two passes')
            native,ids,mask = chunk_inputs(slots)
            visual,images,texts = _features(view,native,ids,mask,amp)
            if not torch.equal(images,cached_images[slots]) or not torch.equal(texts,cached_texts[slots]):
                raise ValueError('Actual recomputed features differ from the exact cached source features')
            with amp():
                output = native_from_visual(view.native,native['input_ids'],visual,
                    attention_mask=native['attention_mask'],labels=native['labels'])
            if output['logits'].dtype != native_logits_dtype or not bool(torch.isfinite(output['loss'])):
                raise ValueError('Complete actual caption CE and native precision must match the declared graph')
            torch.autograd.backward([images,texts],grad_tensors=[gradients.image_gradients[slots].to(device,images.dtype),
                gradients.text_gradients[slots].to(device,texts.dtype)],inputs=selected,retain_graph=True)
            count = sum(local_targets[i] for i in slots)
            (output['loss']*count/sum(global_response_EOS_targets)).backward(inputs=selected)
            ce_sum += float(output['loss'].detach())*count
            observed += count
            replay_calls += 1
            del output,visual,images,texts,native,ids,mask
        if (observed != sum(local_targets) or before != [(id(p),p._version,p.data_ptr(),p.requires_grad) for p in parameters]
                or any(p.grad is not None for p in fixed)
                or bool(local_positions) and any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in selected)
                or not local_positions and any(p.grad is not None for p in selected)):
            raise ValueError('Every original caption target and finite selected gradient, and no fixed gradient, must survive replay')
    except Exception as exc:
        error = type(exc).__name__+': '+str(exc)
    coordinated(error,'feature replay and whole-caption backward',clear=True)
    return dict(rank=rank,world=world,global_positions=global_positions,local_positions=local_positions,
        global_original_response_EOS_targets=sum(global_response_EOS_targets),local_original_response_EOS_targets=observed,
        local_caption_CE_sum=ce_sum,global_matching=gradients.record(),actual_graph_free_encoder_calls=cache_calls,
        actual_replay_encoder_calls=replay_calls,maximum_live_encoder_chunks=1,
        source_feature_replay_bitexact=True,graph_free_cache_weight_autocast_cache_disabled=True,
        microbatch_source_positions=[[local_positions[i] for i in chunk] for chunk in chunks],
        one_parameter_gradient_SUM_still_required=True,optimizer_committed=False,
        complete_original_global_pairs_and_caption_EOS_preserved=True,fixed_weights_and_trainability_untouched=True,
        full_size_GPU_memory_throughput_capacity_recovery_training_quality_pending=True)
