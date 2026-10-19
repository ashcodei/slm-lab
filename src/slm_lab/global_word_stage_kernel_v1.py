"""Whole-word CTC plus native response/EOS CE with global normalization.

CTC keeps bytes0..255/blank256, all140frames and the original FP32 recurrent
readout. Native CE differentiates through the fixed own language policy to
the shared vision encoder. Only vision/word leaves receive gradients. The
caller performs one parameter-gradient SUM, clips and commits its optimizer.
"""
from contextlib import nullcontext
import hashlib
import json
import math

import torch
from torch import distributed as dist

from .own_actor_vision_prefix_split_backward_v1 import OwnActorVisionPrefixSplitBackward
from .shared_word_data_v1 import native_bytes
from .shared_word_stage_v1 import ctc_word_sum


def named_word_selected(view):
    return [('vision.'+n,p) for n,p in view.actor.base.network.vision.named_parameters()]+[
        ('word.'+n,p) for n,p in view.actor.base.heads['word'].named_parameters()]


def accumulate_global_word(view,batch,*,global_positions,local_positions,global_response_EOS_targets,
                           global_word_UTF8_bytes,CTC_weight=1.,native_CE_weight=1.,
                           native_logits_dtype=torch.float32,amp=None,group=None):
    distributed = dist.is_initialized()
    if distributed and dist.get_backend(group) != 'gloo':raise ValueError('Actual explicit Gloo control group required')
    if not distributed and group is not None:raise ValueError('An uninitialized group cannot stand in for a serial word call')
    world = dist.get_world_size(group) if distributed else 1;rank = dist.get_rank(group) if distributed else 0
    def gather(value):
        if world == 1:return [value]
        result = [None]*world;dist.all_gather_object(result,value,group=group);return result
    def coordinated(error,phase):
        errors = gather(error)
        if any(e is not None for e in errors):raise ValueError('Coordinated word '+phase+' rejection: '+repr(errors))
    error = None;specification = None
    try:
        if (type(view) is not OwnActorVisionPrefixSplitBackward or not view.training or not view.native.training
                or not 1 <= world <= 4 or not isinstance(global_positions,list) or not 1 <= len(global_positions) <= 64
                or len(set(global_positions)) != len(global_positions) or any(type(p) is not int or p < 0 for p in global_positions)
                or local_positions != global_positions[rank::world] or len(global_response_EOS_targets) != len(global_positions)
                or len(global_word_UTF8_bytes) != len(global_positions) or any(type(n) is not int or n < 1 for n in global_response_EOS_targets)
                or any(type(n) is not int or not 1 <= n <= 64 for n in global_word_UTF8_bytes)
                or any(type(v) not in (int,float) or not math.isfinite(v) or v <= 0 for v in [CTC_weight,native_CE_weight])
                or native_logits_dtype not in [torch.float32,torch.bfloat16] or amp is not None and not callable(amp)):
            raise ValueError('Complete original word/native target lists and explicit finite global objective required')
        view.assert_shared();selected = [p for _,p in named_word_selected(view)];chosen = {id(p) for p in selected}
        if any(p.grad is not None for p in view.parameters()):raise ValueError('Every gradient must start clear')
        fixed = [p for p in view.parameters() if id(p) not in chosen]
        before = [(id(p),p._version,p.data_ptr(),p.requires_grad) for p in fixed]
        device = next(view.parameters()).device
        if any(p.dtype != torch.float32 or p.device != device for p in view.parameters()):
            raise ValueError('One own FP32-master replica with optional explicit autocast required')
        local_targets = global_response_EOS_targets[rank::world];local_bytes = global_word_UTF8_bytes[rank::world]
        if not local_positions:
            if batch is not None:raise ValueError('An empty tail rank must not fabricate a source batch')
        else:
            if (not isinstance(batch,dict) or set(batch) != {'input_ids','labels','attention_mask','pixel_values','word_pixels','texts'}
                    or batch['input_ids'].ndim != 2 or batch['labels'].shape != batch['input_ids'].shape
                    or batch['attention_mask'].shape != batch['input_ids'].shape or batch['attention_mask'].dtype != torch.bool
                    or len(batch['input_ids']) != len(local_positions) or len(batch['texts']) != len(local_positions)
                    or tuple(batch['pixel_values'].shape) != (len(local_positions),3,448,448)
                    or tuple(batch['word_pixels'].shape) != (len(local_positions),3,64,448)
                    or any(batch[k].device != device for k in ['input_ids','labels','attention_mask','pixel_values','word_pixels'])):
                raise ValueError('Whole original448px native and64x448word inputs with corresponding complete transcripts required')
            for text,wanted in zip(batch['texts'],local_bytes):
                values,required = native_bytes(text)
                if len(values) != wanted or required > 140:raise ValueError('Every original UTF8 byte and repeated-byte alignment must remain complete')
            for ids,labels,mask,wanted in zip(batch['input_ids'],batch['labels'],batch['attention_mask'],local_targets):
                length = int(mask.sum());active = (labels != -100).nonzero().flatten()
                if (not 2 <= length <= view.actor.config.max_text_length or not bool(mask[:length].all()) or bool(mask[length:].any())
                        or int(ids[0]) != 1 or int(ids[length-1]) != 2 or not len(active) or int(active[0]) < 1
                        or active.tolist() != list(range(int(active[0]),length)) or len(active) != wanted
                        or not torch.equal(ids[active],labels[active])):
                    raise ValueError('Every whole native word response and genuine final EOS must retain its global target count')
        specification = hashlib.sha256(json.dumps(dict(global_positions=global_positions,global_response_EOS_targets=global_response_EOS_targets,
            global_word_UTF8_bytes=global_word_UTF8_bytes,CTC_weight=CTC_weight,native_CE_weight=native_CE_weight,
            native_logits_dtype=str(native_logits_dtype),CTC_blank=256,CTC_frames=140),sort_keys=True).encode()).hexdigest()
    except Exception as exc:error = type(exc).__name__+': '+str(exc)
    coordinated(error,'source/native preflight')
    if any(v != specification for v in gather(specification)):raise ValueError('Coordinated word global source/normalization/precision mismatch')
    amp = amp or nullcontext;ctc_total = ce_total = 0.;word_frames = byte_count = native_calls = word_calls = 0;error = None
    try:
        if local_positions:
            with amp():logits = view.actor.base.word_logits(batch['word_pixels'])
            if logits.shape != (len(local_positions),140,257) or logits.dtype != torch.float32:
                raise ValueError('All140actual CTC frames and the original FP32 recurrent readout must be preserved under AMP')
            ctc,byte_count = ctc_word_sum(logits,batch['texts']);word_frames = logits.shape[1];word_calls += 1
            (ctc*CTC_weight/len(global_positions)).backward(inputs=selected)
            ctc_total = float(ctc.detach());del ctc,logits
            inputs = {k:batch[k] for k in ['input_ids','attention_mask','pixel_values']}
            with amp():logits = view.native(**inputs)['logits']
            if logits.dtype != native_logits_dtype:raise ValueError('Actual native word CE precision differs from the declared graph')
            target = batch['labels'][:,1:];mask = target != -100
            ce = torch.nn.functional.cross_entropy(logits[:,:-1].float()[mask],target[mask],reduction='sum')
            if not bool(torch.isfinite(ce)):raise ValueError('Complete actual native word CE must be finite')
            (ce*native_CE_weight/sum(global_response_EOS_targets)).backward(inputs=selected)
            ce_total = float(ce.detach());native_calls += 1
            if byte_count != sum(local_bytes) or any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in selected):
                raise ValueError('Every local complete word byte and every selected vision/word gradient must remain finite')
        elif any(p.grad is not None for p in selected):raise ValueError('An empty rank must not manufacture a word gradient')
        if (before != [(id(p),p._version,p.data_ptr(),p.requires_grad) for p in fixed]
                or any(p.grad is not None for p in fixed)):
            raise ValueError('All fixed language/actor/critic/pointing/other-head values and metadata must remain untouched')
    except Exception as exc:error = type(exc).__name__+': '+str(exc)
    coordinated(error,'actual whole-word backward')
    return dict(rank=rank,world=world,global_positions=global_positions,local_positions=local_positions,
        global_words=len(global_positions),global_response_EOS_targets=sum(global_response_EOS_targets),global_UTF8_bytes=sum(global_word_UTF8_bytes),
        local_words=len(local_positions),local_response_EOS_targets=sum(local_targets),local_UTF8_bytes=byte_count,
        local_byte_normalized_CTC_sum=ctc_total,local_native_CE_sum=ce_total,CTC_weight=CTC_weight,native_CE_weight=native_CE_weight,
        actual_word_forward_calls=word_calls,actual_native_forward_calls=native_calls,actual_CTC_frames=word_frames,
        actual_word_readout_dtype='torch.float32' if local_positions else None,
        actual_native_logits_dtype=str(native_logits_dtype) if local_positions else None,
        one_global_parameter_gradient_SUM_still_required=True,optimizer_committed=False,
        complete_original_transcripts_and_native_EOS_preserved=True,fixed_weights_and_trainability_metadata_untouched=True,
        full_size_CUDA_determinism_capacity_recovery_and_quality_pending=True)
