"""Bounded caption/matching/word/correction gradients for one joint policy.

Caption CE can update the actor as well as vision; matching text features stay
detached, following the existing own matching objective. Word CTC preserves
every UTF8 byte, alongside native word CE. Corrective negatives use explicit
length-normalized supervised ranking, not human DPO or a moving reference.
No optimizer, source admission or full-size GPU capacity is supplied here.
"""
from contextlib import nullcontext, contextmanager
from functools import wraps
import hashlib
import math

import torch
from torch.nn import functional as F

from .global_native_visual_stage_kernel_v1 import native_microbatches
from .global_visual_matching_gradients_v1 import global_matching_gradients
from .joint_batched_human_preferences_v1 import row_logp
from .own_actor_vision_alignment_v1 import pair_positives
from .own_actor_vision_prefix_reuse_v1 import native_from_visual
from .parallel_function_repair_stage_v1 import value_digest
from .shared_joint_actor_vision_v1 import SharedJointActorVisionV1
from .shared_joint_checkpoint_v1 import agree, control
from .shared_word_data_v1 import native_bytes
from .shared_word_stage_v1 import ctc_word_sum

BASE = {'source_role', 'purpose', 'source_sha256', 'positions', 'context_lengths', 'response_EOS_targets',
    'image_size', 'negative_context_lengths', 'negative_response_EOS_targets'}


def explicit_autocast(function):
    @wraps(function)
    def run(view,*args,amp=None,**kwargs):
        device=next(view.parameters()).device.type
        if amp is None and torch.is_autocast_enabled(device):
            dtype=torch.get_autocast_dtype(device)
            amp=lambda:torch.autocast(device,dtype=dtype,cache_enabled=False)
        original=amp or nullcontext
        @contextmanager
        def forward_amp():
            with original():
                with torch.autocast(device,enabled=torch.is_autocast_enabled(device),
                        dtype=torch.get_autocast_dtype(device),cache_enabled=False):
                    yield
        # Trainable cast graphs must not leak between a detached caption
        # anchor, gradient-cache replay, or independently freed microbatches.
        with torch.autocast(device,enabled=False):return function(view,*args,amp=forward_amp,**kwargs)
    return run


def feature_dtype(view,amp):
    device=next(view.parameters()).device.type
    with (amp or nullcontext)():
        return torch.get_autocast_dtype(device) if torch.is_autocast_enabled(device) else torch.float32


def signature(view):
    return [(n, id(p), p._version, p.data_ptr(), p.requires_grad, p.dtype, p.device) for n, p in view.named_parameters()]


def add(view, result, outputs, grad_outputs=None, *, retain_graph=False):
    named = view.named_selected()
    partial = torch.autograd.grad(outputs, [p for _, p in named], grad_outputs=grad_outputs, allow_unused=True,retain_graph=retain_graph)
    for (name, _), value in zip(named, partial):
        if value is not None:
            if not bool(torch.isfinite(value).all()): raise ValueError('Every actual joint task gradient must remain finite')
            result[name] = value if result[name] is None else result[name] + value


def validate_native(batch, lengths, targets, *, device, vocab_size, image_size):
    if (type(batch) is not dict or set(batch) != {'input_ids', 'labels', 'attention_mask', 'pixel_values'}
            or batch['input_ids'].ndim != 2 or batch['input_ids'].dtype != torch.long
            or batch['labels'].shape != batch['input_ids'].shape or batch['labels'].dtype != torch.long
            or batch['attention_mask'].shape != batch['input_ids'].shape or batch['attention_mask'].dtype != torch.bool
            or len(batch['input_ids']) != len(lengths) or any(batch[k].device != device for k in ['input_ids','labels','attention_mask'])):
        raise ValueError('An exact complete native batch on the actual model device is required')
    if image_size is None:
        if batch['pixel_values'] is not None: raise ValueError('A text-only source cannot acquire an image')
    else:
        pixels = batch['pixel_values']
        if (not isinstance(pixels, torch.Tensor) or pixels.device != device or pixels.dtype != torch.float32
                or pixels.shape != (len(lengths),3,image_size,image_size) or not bool(torch.isfinite(pixels).all())
                or bool((pixels.abs()>1).any())): raise ValueError('Every complete original normalized RGB image is required')
    for ids, labels, mask, length, wanted in zip(batch['input_ids'], batch['labels'], batch['attention_mask'], lengths, targets):
        active = (labels != -100).nonzero().flatten()
        if (int(mask.sum()) != length or not bool(mask[:length].all()) or bool(mask[length:].any())
                or int(ids[0]) != 1 or int(ids[length-1]) != 2 or bool((ids[length:] != 0).any())
                or bool(((ids[1:length-1]<4)|(ids[1:length-1]>=vocab_size)).any())
                or len(active) != wanted or not len(active) or int(active[0]) < 1
                or active.tolist() != list(range(int(active[0]),length)) or not torch.equal(ids[active],labels[active])):
            raise ValueError('Every full original prompt/response and true EOS target must remain intact')


def preflight(view, batch, source, *, kind, negatives, word, matching, maximum_examples, maximum_padded_tokens, weights, precision, group):
    rank, world = control(group); error = identity = result = None
    try:
        extra = {'word_height','word_width','UTF8_bytes','UTF8_sha256','CTC_frames'} if kind=='word' else {'pixel_keys','captions'} if kind=='photo' else set()
        if (type(view) is not SharedJointActorVisionV1 or type(source) is not dict or set(source)!=BASE|extra
                or source['source_role']!='train' or source['purpose'] not in ['optimization','capacity']
                or type(source['source_sha256']) is not str or len(source['source_sha256'])!=64
                or any(c not in '0123456789abcdef' for c in source['source_sha256'])
                or type(source['positions']) is not list or not 1<=len(source['positions'])<=64
                or any(type(p) is not int or p<0 for p in source['positions']) or len(set(source['positions']))!=len(source['positions'])
                or any(len(source[k])!=len(source['positions']) for k in ['context_lengths','response_EOS_targets','negative_context_lengths','negative_response_EOS_targets'])
                or any(type(v) not in [int,float] or not math.isfinite(v) or v<=0 for v in weights.values())):
            raise ValueError('The exact joint graph, whole train source, roles and explicit positive objective settings are required')
        view.assert_shared(); device=next(view.parameters()).device
        if any(p.dtype!=torch.float32 or p.device!=device or p.grad is not None for p in view.parameters()):
            raise ValueError('Every FP32 parameter must begin at a clear-gradient boundary on one actual device')
        prefix=view.actor.config.image_token_count(source['image_size'])
        lengths=source['context_lengths']; targets=source['response_EOS_targets']
        if (any(type(n) is not int or not 2<=n<=view.actor.config.max_text_length for n in lengths)
                or any(type(n) is not int or n<1 or n>=l for n,l in zip(targets,lengths))):
            raise ValueError('Every original complete native source must fit the actual decoder')
        negative_lengths=source['negative_context_lengths']; negative_targets=source['negative_response_EOS_targets']
        if any(type(n) is not int or type(t) is not int or (n,t)!=(0,0) and not 1<=t<n<=view.actor.config.max_text_length for n,t in zip(negative_lengths,negative_targets)):
            raise ValueError('Every complete corrective negative must retain its real context/EOS budget')
        if kind!='native' and any(negative_lengths): raise ValueError('Photo/word branches cannot silently consume a different corrective objective')
        effective=[l+(n+prefix if n else 0) for l,n in zip(lengths,negative_lengths)]
        # The existing bounded microbatch primitive caps one source at4096;
        # paired corrections can have two complete decoder contexts.
        if any(l+prefix>maximum_padded_tokens for l in effective): raise ValueError('Complete paired native contexts exceed capacity; clipping is forbidden')
        chunks=native_microbatches(lengths[rank::world],prefix_tokens=prefix,
            maximum_examples=maximum_examples,maximum_padded_tokens=maximum_padded_tokens)
        local=source['positions'][rank::world]
        if local:
            validate_native(batch,lengths[rank::world],targets[rank::world],device=device,vocab_size=view.actor.config.vocab_size,image_size=source['image_size'])
            if type(negatives) is not dict or set(negatives)!={i for i,n in enumerate(negative_lengths[rank::world]) if n}:
                raise ValueError('All and only original local corrective negatives are required')
            for i,negative in negatives.items():
                validate_native(negative,[negative_lengths[rank::world][i]],[negative_targets[rank::world][i]],
                    device=device,vocab_size=view.actor.config.vocab_size,image_size=source['image_size'])
                if not torch.equal(negative['pixel_values'],batch['pixel_values'][i:i+1]):
                    raise ValueError('The corrective positive and original negative must retain the same actual pixels')
                p=int((batch['labels'][i]!=-100).nonzero()[0]);q=int((negative['labels'][0]!=-100).nonzero()[0])
                if p!=q or not torch.equal(batch['input_ids'][i,:p],negative['input_ids'][0,:q]):
                    raise ValueError('A corrective pair must retain its identical complete original prompt')
        elif batch is not None or negatives!={}: raise ValueError('An empty task rank cannot fabricate a source or negative')
        if kind=='word':
            if (any(len(source[k])!=len(source['positions']) for k in ['UTF8_bytes','UTF8_sha256'])
                    or any(type(n) is not int or not 1<=n<=64 for n in source['UTF8_bytes'])
                    or any(type(n) is not int or not view.actor.config.patch_size<=n<=view.actor.config.max_image_size
                        or n%view.actor.config.patch_size for n in [source['word_height'],source['word_width']])
                    or source['CTC_frames']!=source['word_width']//view.actor.config.patch_size*view.actor.base.heads['word'].expansion):
                raise ValueError('Every whole CTC word, frame count and byte identity must remain declared')
            if local:
                if (type(word) is not dict or set(word)!={'word_pixels','texts'} or word['word_pixels'].shape!=(len(local),3,source['word_height'],source['word_width'])
                        or word['word_pixels'].dtype!=torch.float32 or word['word_pixels'].device!=device
                        or not bool(torch.isfinite(word['word_pixels']).all()) or bool((word['word_pixels'].abs()>1).any())
                        or len(word['texts'])!=len(local)):
                    raise ValueError('The complete actual rectangular word pixels and full transcripts are required')
                for text,n,h in zip(word['texts'],source['UTF8_bytes'][rank::world],source['UTF8_sha256'][rank::world]):
                    values,required=native_bytes(text)
                    if len(values)!=n or required>source['CTC_frames'] or hashlib.sha256(text.encode()).hexdigest()!=h:
                        raise ValueError('A complete original UTF8 transcript differs or exceeds CTC capacity')
            elif word is not None: raise ValueError('An empty word rank cannot acquire pixels')
            if matching is not None: raise ValueError('A word task cannot fabricate photo matching')
        elif word is not None: raise ValueError('Only the original word family may have CTC inputs')
        if kind=='photo':
            positives=pair_positives(source['pixel_keys'],source['captions'])
            if len(positives)!=len(source['positions']) or bool(positives.all()):raise ValueError('Full global photo pairs and observed negatives are required')
            if local:
                if (type(matching) is not dict or set(matching)!={'text_ids','text_mask'} or matching['text_ids'].ndim!=2
                        or matching['text_mask'].shape!=matching['text_ids'].shape or matching['text_mask'].dtype!=torch.bool
                        or matching['text_ids'].dtype!=torch.long or matching['text_ids'].device!=device or matching['text_mask'].device!=device
                        or len(matching['text_ids'])!=len(local)):raise ValueError('Actual complete caption-only native inputs are required')
                for ids,mask,labels in zip(matching['text_ids'],matching['text_mask'],batch['labels']):
                    n=int(mask.sum()); active=labels[labels!=-100]
                    if n<3 or not bool(mask[:n].all()) or bool(mask[n:].any()) or ids[:n].tolist()!=[1]+active.tolist() or bool((ids[n:]!=0).any()):
                        raise ValueError('Every matching caption must equal the complete original response with BOS/EOS')
            elif matching is not None:raise ValueError('An empty photo rank cannot fabricate caption features')
        elif matching is not None:raise ValueError('Only original paired photos may have matching inputs')
        identity=value_digest(dict(source=source,kind=kind,maximum_examples=maximum_examples,maximum_padded_tokens=maximum_padded_tokens,weights=weights,precision=str(precision)))
        result=(rank,world,device,prefix,chunks,signature(view))
    except Exception as exc:error=type(exc).__name__+': '+str(exc)
    agree(error,identity,group,world)
    return result


def subset(batch, slots):
    width=int(batch['attention_mask'][slots].sum(1).max())
    return {k:v[slots] if k=='pixel_values' else v[slots,:width] for k,v in batch.items()}


def finish(view, result, before, record, group):
    _,world=control(group);error=None
    if signature(view)!=before or any(p.grad is not None for p in view.parameters()):error='A joint task cannot mutate model values, metadata or accumulated gradients'
    agree(error,None,group,world)
    return result,dict(**record,one_global_parameter_SUM_still_required=world>1,optimizer_updates=0,
        full_size_GPUcapacity_throughput_source_admission_and_quality_unproved=True)


@explicit_autocast
def native_gradients(view,batch,source,*,negatives=None,maximum_examples=4,maximum_padded_tokens=16384,correction_weight=.1,amp=None,group=None):
    negatives={} if negatives is None else negatives
    rank,world,device,prefix,chunks,before=preflight(view,batch,source,kind='native',negatives=negatives,word=None,matching=None,
        maximum_examples=maximum_examples,maximum_padded_tokens=maximum_padded_tokens,weights={'correction':correction_weight},precision=feature_dtype(view,amp),group=group)
    amp=amp or nullcontext;result={n:None for n,_ in view.named_selected()};ce_sum=ranking_sum=0.;observed=negative_observed=0;calls=0
    # Corrective pairs use one source at a time so both original contexts fit
    # the declared bound together. No positive/negative source is dropped.
    if negatives:chunks=[[i] for chunk in chunks for i in chunk]
    error=None
    try:
        for slots in chunks:
            native=subset(batch,slots)
            with amp():
                visual=view.alignment.native.vision(native['pixel_values'],view.alignment.native.gradient_checkpointing)
                output=native_from_visual(view.alignment.native,native['input_ids'],visual,attention_mask=native['attention_mask'])['logits']
            positive,count=row_logp(output,native['labels']);loss=-positive.sum()/sum(source['response_EOS_targets'])
            ce_sum-=float(positive.detach().sum());observed+=int(count.sum())
            for offset,slot in enumerate(slots):
                if slot not in negatives:continue
                negative=negatives[slot]
                with amp():
                    other=native_from_visual(view.alignment.native,negative['input_ids'],visual[offset:offset+1],attention_mask=negative['attention_mask'])['logits']
                rejected,n=row_logp(other,negative['labels'])
                margin=positive[offset]/count[offset]-rejected[0]/n[0]
                ranking=F.softplus(-margin)
                loss=loss+correction_weight*ranking/sum(n>0 for n in source['negative_context_lengths'])
                ranking_sum+=float(ranking.detach());negative_observed+=int(n.sum())
            add(view,result,loss);calls+=1
    except Exception as exc:error=type(exc).__name__+': '+str(exc)
    agree(error,None,group,world)
    return finish(view,result,before,dict(kind='native',rank=rank,world=world,local_CE_sum=ce_sum,local_response_EOS_targets=observed,
        local_correction_ranking_sum=ranking_sum,local_negative_response_EOS_targets=negative_observed,actual_native_microbatches=calls,
        corrective_ranking_is_length_normalized_supervision_not_human_DPO=True),group)


@explicit_autocast
def word_gradients(view,batch,word,source,*,maximum_examples=4,maximum_padded_tokens=16384,CTC_weight=1.,native_CE_weight=1.,amp=None,group=None):
    rank,world,device,prefix,chunks,before=preflight(view,batch,source,kind='word',negatives={},word=word,matching=None,
        maximum_examples=maximum_examples,maximum_padded_tokens=maximum_padded_tokens,weights={'CTC':CTC_weight,'native_CE':native_CE_weight},precision=feature_dtype(view,amp),group=group)
    amp=amp or nullcontext;result={n:None for n,_ in view.named_selected()};ctc_sum=ce_sum=0.;bytes_seen=targets_seen=0;error=None
    try:
        for slots in chunks:
            with amp():logits=view.word_logits(word['word_pixels'][slots])
            if logits.shape!=(len(slots),source['CTC_frames'],257) or logits.dtype!=torch.float32:raise ValueError('All declared original CTC frames and FP32 byte logits must remain')
            with torch.autocast('cpu',enabled=False):ctc,observed=ctc_word_sum(logits,[word['texts'][i] for i in slots])
            add(view,result,ctc*CTC_weight/len(source['positions']));ctc_sum+=float(ctc.detach());bytes_seen+=observed
            native=subset(batch,slots)
            with amp():output=view.native_image(**native)
            count=int((native['labels'][:,1:]!=-100).sum());mean=output['loss']
            add(view,result,mean*native_CE_weight*count/sum(source['response_EOS_targets']))
            ce_sum+=float(mean.detach())*count;targets_seen+=count
    except Exception as exc:error=type(exc).__name__+': '+str(exc)
    agree(error,None,group,world)
    return finish(view,result,before,dict(kind='word',rank=rank,world=world,local_byte_normalized_CTC_sum=ctc_sum,local_native_CE_sum=ce_sum,
        local_UTF8_bytes=bytes_seen,local_response_EOS_targets=targets_seen,actual_word_and_native_microbatches=len(chunks)),group)


@explicit_autocast
def photo_gradients(view,batch,matching,source,*,maximum_examples=4,maximum_padded_tokens=16384,matching_weight=.1,temperature=.07,amp=None,group=None):
    rank,world,device,prefix,chunks,before=preflight(view,batch,source,kind='photo',negatives={},word=None,matching=matching,
        maximum_examples=maximum_examples,maximum_padded_tokens=maximum_padded_tokens,weights={'matching':matching_weight,'temperature':temperature},precision=feature_dtype(view,amp),group=group)
    amp=amp or nullcontext;result={n:None for n,_ in view.named_selected()};images=[];texts=[]
    def features(slots):
        native=subset(batch,slots);ids=matching['text_ids'][slots];mask=matching['text_mask'][slots]
        with torch.no_grad(),amp():
            hidden=view.actor.policy(ids,attention_mask=mask,return_hidden=True,return_logits=False)['hidden']
            content=mask&(ids!=0)&(ids!=1)&(ids!=2);text=(hidden.float()*content[:,:,None]).sum(1)/content.sum(1,keepdim=True)
        with amp():
            visual=view.alignment.native.vision(native['pixel_values'],view.alignment.native.gradient_checkpointing)
            heads=view.actor.base.heads['pointing'];image=heads['patch'](visual.mean(1));text=heads['query'](text.to(visual.dtype))
        return native,visual,image,text
    error=None
    try:
        for slots in chunks:
            with torch.no_grad(),amp():
                with torch.autocast(device.type,enabled=torch.is_autocast_enabled(device.type),dtype=torch.get_autocast_dtype(device.type),cache_enabled=False):
                    _,_,image,text=features(slots)
            images.append(image.detach());texts.append(text.detach())
        width=view.actor.base.settings['pointing_rank']
        cached_images=torch.cat(images).requires_grad_() if images else torch.empty(0,width,device=device,dtype=feature_dtype(view,amp))
        cached_texts=torch.cat(texts).requires_grad_() if texts else torch.empty_like(cached_images)
    except Exception as exc:error=type(exc).__name__+': '+str(exc)
    agree(error,None,group,world)
    derivatives=global_matching_gradients(cached_images,cached_texts,global_positions=source['positions'],local_positions=source['positions'][rank::world],
        pixel_keys=source['pixel_keys'],captions=source['captions'],matching_weight=matching_weight,temperature=temperature,group=group)
    ce_sum=0.;targets_seen=0;error=None
    try:
        for slots in chunks:
            native,visual,image,text=features(slots)
            if not torch.equal(image,cached_images[slots]) or not torch.equal(text,cached_texts[slots]):raise ValueError('Every recomputed complete photo feature must match its unchanged cache')
            with amp():output=native_from_visual(view.alignment.native,native['input_ids'],visual,attention_mask=native['attention_mask'],labels=native['labels'])
            count=int((native['labels'][:,1:]!=-100).sum());mean=output['loss']
            # Preserve branch-wise FP32 parameter accumulation. Combining
            # both upstream derivatives in a shared BF16 visual activation
            # rounds their sum differently from the established two-encoder
            # objective. Reuse the prefix graph, with separate VJPs.
            add(view,result,(image,text),(derivatives.image_gradients[slots].to(device,image.dtype),
                derivatives.text_gradients[slots].to(device,text.dtype)),retain_graph=True)
            # Preserve the established native mean-CE reduction and scaling
            # order. A selected-token sum is algebraically equivalent but
            # changes BF16 backward rounding before the encoder gradients.
            add(view,result,mean*count/sum(source['response_EOS_targets']))
            ce_sum+=float(mean.detach())*count;targets_seen+=count
    except Exception as exc:error=type(exc).__name__+': '+str(exc)
    agree(error,None,group,world)
    return finish(view,result,before,dict(kind='photo',rank=rank,world=world,local_native_CE_sum=ce_sum,local_response_EOS_targets=targets_seen,
        complete_global_matching=derivatives.record(),actual_cache_encoder_calls=len(chunks),actual_replay_encoder_calls=len(chunks),
        maximum_live_encoder_chunks=1,replayed_source_features_bitexact=True),group)
