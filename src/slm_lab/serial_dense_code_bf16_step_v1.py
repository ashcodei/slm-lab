"""Whole-source BF16 SFT of every own dense language matrix/table/norm.

Uses the same admitted public data, response/EOS objective and serial global
normalizer as the adapter study. No vision, critic or auxiliary tensor enters
Adam. This operator grants neither GPU ownership nor training admission.
"""
import math

import torch

from .codenet_checkpoint_v1 import capture_rng
from .complete_response_stopping_loss_v1 import metadata,complete_response_CE_sums
from .frozen_own_actor_reference_v1 import signature
from .own_dense_checkpoint_v1 import selected
from .parallel_function_repair_stage_v1 import value_digest
from .public_reasoned_code_train_data_v1 import PublicReasonedCodeSFTData


def update(model,optimizer,data,positions,profile,*,gradient_clip=1.,checked=lambda:None,allow_mechanics=False):
    if (type(data) is not PublicReasonedCodeSFTData or data.role!='train' or data.views!='both'
            or data.human_preferences is not False or data.own_model_on_policy is not False
            or data.use_as_RL_reward_oracle_forbidden is not True
            or not isinstance(positions,list) or not positions or len(positions)!=len(set(positions))
            or type(gradient_clip) not in [int,float] or not math.isfinite(gradient_clip) or gradient_clip<=0):
        raise ValueError('Distinct complete admitted train views and finite global clip required')
    named=selected(model,optimizer);parameters=[p for _,p in named];device=parameters[0].device
    small=allow_mechanics is True and device.type=='cpu' and model.counts()['total_unique']<10**7
    if not small and (device.type!='cuda' or model.counts()['total_unique']!=984399690
            or sum(p.numel() for p in parameters)!=931244544 or len(parameters)!=317
            or model.config.max_text_length!=4096 or not torch.are_deterministic_algorithms_enabled()
            or torch.backends.cuda.matmul.allow_tf32):
        raise ValueError('The actual deterministic own984M CUDA native or explicit small CPU mechanics is required')
    chosen={id(p) for p in parameters};fixed=[p for p in model.parameters() if id(p) not in chosen]
    if any(p.dtype!=torch.float32 or p.grad is not None for p in model.parameters()):
        raise ValueError('FP32 dense weights/Adam and a clear gradient boundary required')
    data.guard.check();info=[data.information(p) for p in positions];meta=metadata(info,profile)
    previous=0
    if optimizer.state:
        clocks={float(optimizer.state[p]['step']) for p in parameters if p in optimizer.state}
        if len(optimizer.state)!=len(parameters) or len(clocks)!=1:
            raise ValueError('Every dense-language Adam state and common clock is required')
        previous=int(next(iter(clocks)))
    before=signature(fixed);rng=value_digest(capture_rng(device));model.train();model.network.gradient_checkpointing=True
    micros=[]
    for position,source in zip(positions,info):
        checked();batch=data.batch([position],device);ids,labels=data.sequence(position)
        if (batch['input_ids'][0].tolist()!=ids or batch['labels'][0].tolist()!=labels
                or not bool(batch['attention_mask'].all()) or len(ids)!=source['complete_context_tokens']):
            raise ValueError('Every whole original input, response/EOS and real context length is required')
        with torch.autocast(device.type,dtype=torch.bfloat16):
            logits=model(batch['input_ids'],attention_mask=batch['attention_mask'],pixel_values=None)['logits']
        if logits.dtype!=torch.bfloat16:raise ValueError('The declared BF16 forward must actually execute')
        with torch.autocast(device.type,enabled=False):
            ce=complete_response_CE_sums(logits.float(),batch['labels'],batch['attention_mask'],profile=profile)
            loss=ce['weighted_CE_sums'].sum()/meta['actual_global_weighted_targets']
        loss.backward(inputs=parameters)
        micros.append(dict(position=position,complete_context_tokens=len(ids),
            original_CE_sum=float(ce['original_CE_sums'].detach()),EOS_CE_sum=float(ce['EOS_CE_sums'].detach()),
            forward_dtype=str(logits.dtype),loss_dtype=str(loss.dtype)))
        del logits,loss,ce,batch
    checked()
    if (signature(fixed)!=before or any(p.grad is not None for p in fixed)
            or any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in parameters)):
        raise ValueError('All dense language tensors, and no fixed tensor, must have finite complete-batch gradients')
    norm=float(torch.nn.utils.clip_grad_norm_(parameters,gradient_clip,error_if_nonfinite=True))
    optimizer.step();optimizer.zero_grad(set_to_none=True)
    if (signature(fixed)!=before or value_digest(capture_rng(device))!=rng
            or any(float(optimizer.state[p]['step'])!=previous+1 for p in parameters)):
        raise ValueError('Fixed parameters/RNG must remain exact while every full-language Adam clock advances once')
    original=math.fsum(m['original_CE_sum'] for m in micros);eos=math.fsum(m['EOS_CE_sum'] for m in micros)
    return dict(actual_positions=list(positions),source_metadata=meta,actual_microbatches=micros,
        original_token_mean_NLL=original/meta['response_EOS_targets'],EOS_token_mean_NLL=eos/meta['genuine_EOS_targets'],
        weighted_token_mean_CE=(original+(meta['EOS_weight']-1)*eos)/meta['actual_global_weighted_targets'],
        gradient_norm_before_clip=norm,selected_Adam_states=len(optimizer.state),actual_Adam_step=previous+1,
        selected_full_language_values=sum(p.numel() for p in parameters),selected_full_language_tensors=len(parameters),
        protocol='full_dense_language_serial_microbatch1_BF16_forward_FP32_CE_weights_Adam',
        private_RNG_unchanged=True,fixed_vision_critic_and_auxiliary_parameters_unchanged=True,
        complete_original_sources_and_EOS_preserved=True,full_epoch_or_model_quality_proved=False)
