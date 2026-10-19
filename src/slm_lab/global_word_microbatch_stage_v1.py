"""Bounded whole-word CTC/native-CE accumulation with global normalizers.

Keep all UTF8 bytes,140frames,blank256 and the FP32 word readout. Only
existing vision/word leaves receive gradients. The caller performs one SUM,
clip and optimizer commit; there is no world average or fabricated tail row.
The historical different-batch-shape raw-gradient failure is not waived.
"""
from contextlib import nullcontext
import hashlib
import json
import math
import re

import torch
from torch import distributed as dist

from .deterministic_positions_v1 import vision_forward, patch_grid
from .global_native_visual_stage_kernel_v1 import native_microbatches
from .global_word_stage_kernel_v1 import named_word_selected
from .own_actor_vision_prefix_split_backward_v1 import OwnActorVisionPrefixSplitBackward
from .shared_word_data_v1 import native_bytes
from .shared_word_stage_v1 import ctc_word_sum


def _accumulate(view,batch,*,global_positions,local_positions,global_context_lengths,
        global_response_EOS_targets,global_word_UTF8_bytes,global_word_UTF8_sha256,
        maximum_examples=4,maximum_padded_tokens=4096,CTC_weight=1.,native_CE_weight=1.,
        native_logits_dtype=torch.float32,amp=None,group=None):
    initialized=dist.is_initialized()
    if initialized and dist.get_backend(group)!='gloo':raise ValueError('Actual explicit Gloo control group required')
    if not initialized and group is not None:raise ValueError('An uninitialized control group cannot stand in for a serial call')
    world=dist.get_world_size(group) if initialized else 1;rank=dist.get_rank(group) if initialized else 0
    def gather(value):
        if world==1:return [value]
        result=[None]*world;dist.all_gather_object(result,value,group=group);return result
    def coordinated(error,phase,clear=False):
        errors=gather(error)
        if any(value is not None for value in errors):
            if clear:
                for parameter in selected:parameter.grad=None
            raise ValueError('Coordinated microbatched word '+phase+' rejection: '+repr(errors))
    error=specification=None
    try:
        if (type(view) is not OwnActorVisionPrefixSplitBackward or not view.training or not view.native.training
                or not 1<=world<=4 or not isinstance(global_positions,list) or not 1<=len(global_positions)<=64
                or len(set(global_positions))!=len(global_positions) or any(type(p) is not int or p<0 for p in global_positions)
                or local_positions!=global_positions[rank::world]
                or any(len(values)!=len(global_positions) for values in [global_context_lengths,global_response_EOS_targets,
                    global_word_UTF8_bytes,global_word_UTF8_sha256])
                or any(type(n) is not int or n<1 for n in global_response_EOS_targets)
                or any(type(n) is not int or not 1<=n<=64 for n in global_word_UTF8_bytes)
                or any(not isinstance(value,str) or re.fullmatch(r'[0-9a-f]{64}',value) is None for value in global_word_UTF8_sha256)
                or any(type(n) not in [int,float] or not math.isfinite(n) or n<=0 for n in [CTC_weight,native_CE_weight])
                or native_logits_dtype not in [torch.float32,torch.bfloat16] or amp is not None and not callable(amp)):
            raise ValueError('Complete bounded global source/native/UTF8 identities and explicit finite objective required')
        view.assert_shared();prefix=view.actor.config.image_token_count(448)
        native_microbatches(global_context_lengths,prefix_tokens=prefix,
            maximum_examples=maximum_examples,maximum_padded_tokens=maximum_padded_tokens)
        if max(global_context_lengths)>view.actor.config.max_text_length:
            raise ValueError('The actual native decoder must fit every whole original word context')
        selected=[p for _,p in named_word_selected(view)];chosen={id(p) for p in selected}
        if any(p.grad is not None for p in view.parameters()) or any(not p.requires_grad for p in selected):
            raise ValueError('Every gradient must start clear and all original selected leaves must be trainable')
        device=next(view.parameters()).device
        if any(p.device!=device or p.dtype!=torch.float32 for p in view.parameters()):
            raise ValueError('One own FP32-master replica and optional explicit forward AMP required')
        if device.type=='cuda' and (getattr(view.native.vision.forward,'__func__',None) is not vision_forward
                or getattr(view.actor.base.patch_grid,'__func__',None) is not patch_grid
                or not torch.are_deterministic_algorithms_enabled()
                or torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32):
            raise ValueError('CUDA requires both existing deterministic native/rectangular position paths and precision controls')
        parameters=list(view.parameters());before=[(id(p),p._version,p.data_ptr(),p.requires_grad) for p in parameters]
        fixed=[p for p in parameters if id(p) not in chosen]
        local_lengths=global_context_lengths[rank::world];local_targets=global_response_EOS_targets[rank::world]
        local_bytes=global_word_UTF8_bytes[rank::world];local_hashes=global_word_UTF8_sha256[rank::world]
        chunks=native_microbatches(local_lengths,prefix_tokens=prefix,
            maximum_examples=maximum_examples,maximum_padded_tokens=maximum_padded_tokens)
        if not local_positions:
            if batch is not None:raise ValueError('Empty ranks must not fabricate any native/CTC source input')
        else:
            if (not isinstance(batch,dict) or set(batch)!={'input_ids','labels','attention_mask','pixel_values','word_pixels','texts'}
                    or batch['input_ids'].ndim!=2 or batch['input_ids'].dtype!=torch.long
                    or batch['labels'].shape!=batch['input_ids'].shape or batch['labels'].dtype!=torch.long
                    or batch['attention_mask'].shape!=batch['input_ids'].shape or batch['attention_mask'].dtype!=torch.bool
                    or len(batch['input_ids'])!=len(local_positions) or len(batch['texts'])!=len(local_positions)
                    or tuple(batch['pixel_values'].shape)!=(len(local_positions),3,448,448)
                    or tuple(batch['word_pixels'].shape)!=(len(local_positions),3,64,448)
                    or any(batch[k].device!=device for k in ['input_ids','labels','attention_mask','pixel_values','word_pixels'])
                    or any(batch[k].dtype!=torch.float32 or not bool(torch.isfinite(batch[k]).all())
                        or bool((batch[k].abs()>1).any()) for k in ['pixel_values','word_pixels'])):
                raise ValueError('All corresponding whole native448px/word64x448px inputs and transcripts required')
            for text,wanted,digest in zip(batch['texts'],local_bytes,local_hashes):
                if not isinstance(text,str):raise ValueError('Every original transcript must be text')
                values,required=native_bytes(text)
                if len(values)!=wanted or required>140 or hashlib.sha256(text.encode('utf8')).hexdigest()!=digest:
                    raise ValueError('Every exact whole original UTF8 byte and repeated-byte alignment must remain unchanged')
            for ids,labels,mask,length,wanted in zip(batch['input_ids'],batch['labels'],batch['attention_mask'],local_lengths,local_targets):
                active=(labels!=-100).nonzero().flatten()
                if (int(mask.sum())!=length or not bool(mask[:length].all()) or bool(mask[length:].any())
                        or int(ids[0])!=1 or int(ids[length-1])!=2 or bool((ids[length:]!=0).any())
                        or bool(((ids[1:length-1]<4)|(ids[1:length-1]>=view.actor.config.vocab_size)).any())
                        or not len(active) or int(active[0])<1 or len(active)!=wanted
                        or active.tolist()!=list(range(int(active[0]),length)) or not torch.equal(ids[active],labels[active])):
                    raise ValueError('Every complete original native response and true EOS must match its global budget')
        specification=hashlib.sha256(json.dumps(dict(global_positions=global_positions,global_context_lengths=global_context_lengths,
            global_response_EOS_targets=global_response_EOS_targets,global_word_UTF8_bytes=global_word_UTF8_bytes,
            global_word_UTF8_sha256=global_word_UTF8_sha256,maximum_examples=maximum_examples,maximum_padded_tokens=maximum_padded_tokens,
            CTC_weight=CTC_weight,native_CE_weight=native_CE_weight,native_logits_dtype=str(native_logits_dtype),
            CTC_blank=256,CTC_frames=140,word_logits_dtype='torch.float32'),sort_keys=True).encode()).hexdigest()
    except Exception as exc:error=type(exc).__name__+': '+str(exc)
    coordinated(error,'source/native preflight')
    if any(value!=specification for value in gather(specification)):
        raise ValueError('Coordinated microbatched word global source/normalization/settings mismatch')
    amp=amp or nullcontext;ctc_total=ce_total=0.;byte_count=target_count=context_count=0;records=[];error=None
    try:
        for slots in chunks:
            width=max(local_lengths[i] for i in slots)
            native={name:batch[name][slots,:width] for name in ['input_ids','labels','attention_mask']}
            native['pixel_values']=batch['pixel_values'][slots]
            texts=[batch['texts'][i] for i in slots]
            with amp():logits=view.actor.base.word_logits(batch['word_pixels'][slots])
            if logits.shape!=(len(slots),140,257) or logits.dtype!=torch.float32:
                raise ValueError('Every one of140CTC frames,257classes and original FP32 recurrent logits required')
            with torch.autocast('cpu',enabled=False):ctc,bytes_observed=ctc_word_sum(logits,texts)
            (ctc*CTC_weight/len(global_positions)).backward(inputs=selected)
            ctc_total+=float(ctc.detach());byte_count+=bytes_observed;del logits,ctc
            with amp():logits=view.native(**{k:native[k] for k in ['input_ids','attention_mask','pixel_values']})['logits']
            if logits.dtype!=native_logits_dtype:raise ValueError('Actual native precision differs from the declared forward graph')
            labels=native['labels'][:,1:];mask=labels!=-100
            ce=torch.nn.functional.cross_entropy(logits[:,:-1].float()[mask],labels[mask],reduction='sum')
            if not bool(torch.isfinite(ce)):raise ValueError('Every original native target must retain finite complete CE')
            (ce*native_CE_weight/sum(global_response_EOS_targets)).backward(inputs=selected)
            ce_total+=float(ce.detach());target_count+=int(mask.sum());context_count+=int(native['attention_mask'].sum())
            records.append(dict(original_positions=[local_positions[i] for i in slots],local_slots=slots,
                complete_context_lengths=[local_lengths[i] for i in slots],UTF8_bytes=bytes_observed,
                response_EOS_targets=int(mask.sum()),padded_decoder_tokens=len(slots)*(width+prefix),
                actual_CTC_frames=140,actual_word_logits_dtype='torch.float32'))
            del ce,logits,native
        if (byte_count!=sum(local_bytes) or target_count!=sum(local_targets) or context_count!=sum(local_lengths)
                or before!=[(id(p),p._version,p.data_ptr(),p.requires_grad) for p in parameters]
                or any(p.grad is not None for p in fixed)
                or bool(local_positions) and any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in selected)
                or not local_positions and any(p.grad is not None for p in selected)):
            raise ValueError('Every complete byte/context/EOS and finite vision/word gradient must survive, with all weights fixed before commit')
    except Exception as exc:error=type(exc).__name__+': '+str(exc)
    coordinated(error,'whole word/native backward',clear=True)
    return dict(rank=rank,world=world,global_positions=global_positions,local_positions=local_positions,
        global_words=len(global_positions),global_response_EOS_targets=sum(global_response_EOS_targets),global_UTF8_bytes=sum(global_word_UTF8_bytes),
        local_words=len(local_positions),local_response_EOS_targets=target_count,local_UTF8_bytes=byte_count,local_context_tokens=context_count,
        local_byte_normalized_CTC_sum=ctc_total,local_native_CE_sum=ce_total,CTC_weight=CTC_weight,native_CE_weight=native_CE_weight,
        actual_word_forward_calls=len(records),actual_native_forward_calls=len(records),actual_microbatches=records,
        complete_original_transcripts_and_native_EOS_preserved=True,whole_UTF8_hashes_checked_before_forward=True,
        derivative_and_backward_autocast_explicitly_disabled=True,one_global_parameter_gradient_SUM_still_required=True,
        optimizer_committed=False,all_registered_parameters_and_trainability_metadata_untouched=True,
        previous_full_batch_shape_raw_gradient_gate_still_unwaived=True,full_size_GPU_capacity_recovery_and_quality_pending=True)


def accumulate_global_word_microbatched(view,*args,**kwargs):
    if type(view) is not OwnActorVisionPrefixSplitBackward:return _accumulate(view,*args,**kwargs)
    device=next(view.parameters()).device.type
    if device not in ['cpu','cuda']:return _accumulate(view,*args,**kwargs)
    if kwargs.get('amp') is None and torch.is_autocast_enabled(device):
        dtype=torch.get_autocast_dtype(device);cache=torch.is_autocast_cache_enabled()
        kwargs['amp']=lambda:torch.autocast(device,dtype=dtype,cache_enabled=cache)
    with torch.autocast(device,enabled=False):return _accumulate(view,*args,**kwargs)
