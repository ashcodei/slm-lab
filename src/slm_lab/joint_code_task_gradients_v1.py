"""Complete positively weighted code CE for the joint adapter/vision graph.

Only the actor receives text gradients. Original source masks, weights and
true EOS targets reach one globally weighted denominator before rank SUM.
No synthetic tests are used as rewards; this module performs no Adam step.
"""
from contextlib import nullcontext
import math
import struct
import hashlib

import torch

from .joint_batched_human_preferences_v1 import row_logp
from .joint_visual_task_gradients_v1 import add, explicit_autocast, feature_dtype, finish, signature, validate_native
from .mixed_coding_SFT_data_v1 import MixedCodingSFTData, PROFILES
from .parallel_function_repair_stage_v1 import value_digest
from .shared_joint_actor_vision_v1 import SharedJointActorVisionV1
from .shared_joint_checkpoint_v1 import agree, control


@explicit_autocast
def gradients(view,data,positions,*,profile='balanced_sources',maximum_examples=4,maximum_padded_tokens=16384,amp=None,group=None):
    rank,world=control(group);error=identity=None
    try:
        if (type(view) is not SharedJointActorVisionV1 or type(data) is not MixedCodingSFTData
                or data.role!='train' or data.purpose!='optimization' or data.use_as_RL_reward_oracle_forbidden is not True
                or profile not in PROFILES or type(positions) is not list or not positions or len(set(positions))!=len(positions)
                or any(type(p) is not int or not 0<=p<len(data) for p in positions)
                or type(maximum_examples) is not int or not 1<=maximum_examples<=64
                or type(maximum_padded_tokens) is not int or maximum_padded_tokens<1):
            raise ValueError('The exact joint model, complete original train-only coding sources and declared bounds are required')
        view.assert_shared();device=next(view.parameters()).device;before=signature(view)
        if any(p.device!=device or p.dtype!=torch.float32 or p.grad is not None for p in view.parameters()):
            raise ValueError('The complete actual FP32 joint graph must begin at a clear-gradient boundary')
        info=[data.information(p) for p in positions];weights=[data.weight(profile,p) for p in positions]
        if any(type(w) not in [int,float] or not math.isfinite(w) or w<=0 for w in weights):raise ValueError('All original positive source weights must remain finite')
        for p,row in zip(positions,info):
            ids,labels=data.sequence(p);start=next(i for i,t in enumerate(labels) if t!=-100)
            if (row['source_role']!='train' or ids[0]!=1 or ids[-1]!=2 or not 2<=len(ids)<=view.actor.config.max_text_length
                    or len(ids)>maximum_padded_tokens or labels!=[-100]*start+ids[start:]
                    or len(ids)!=row['complete_context_tokens'] or len(ids)-start!=row['response_EOS_targets']
                    or hashlib.sha256(struct.pack('<'+'H'*len(ids),*ids)).hexdigest()!=row['complete_native_input_sha256']):
                raise ValueError('Every complete original coding context, response mask and EOS must match before forward')
        denominator=math.fsum(row['response_EOS_targets']*w for row,w in zip(info,weights))
        identity=value_digest(dict(source_identity_sha256=data.identity_sha256,positions=positions,information=info,weights=weights,
            original_global_weighted_targets=denominator,profile=profile,maximum_examples=maximum_examples,
            maximum_padded_tokens=maximum_padded_tokens,precision=str(feature_dtype(view,amp))))
    except Exception as exc:error=type(exc).__name__+': '+str(exc)
    agree(error,identity,group,world)
    lengths={p:row['complete_context_tokens'] for p,row in zip(positions,info)}
    chosen={p:row['response_EOS_targets'] for p,row in zip(positions,info)};weight_by_position=dict(zip(positions,weights))
    local=positions[rank::world];chunks=[];current=[]
    for p in sorted(local,key=lambda p:(lengths[p],p)):
        if current and (len(current)==maximum_examples or max(lengths[q] for q in current+[p])*len(current+[p])>maximum_padded_tokens):
            chunks.append(current);current=[]
        current.append(p)
    if current:chunks.append(current)
    result={n:None for n,_ in view.named_selected()};amp=amp or nullcontext;weighted=unweighted=0.;targets=contexts=0;error=None
    try:
        for chunk in chunks:
            batch=data.batch(chunk,device=device)
            validate_native(batch,[lengths[p] for p in chunk],[chosen[p] for p in chunk],device=device,vocab_size=view.actor.config.vocab_size,image_size=None)
            with amp():logits=view.text(**{k:v for k,v in batch.items() if k!='labels'})['logits']
            logp,count=row_logp(logits,batch['labels']);coefficient=logp.new_tensor([weight_by_position[p] for p in chunk])
            ce=-(logp*coefficient).sum();add(view,result,ce/denominator)
            weighted+=float(ce.detach());unweighted-=float(logp.detach().sum());targets+=int(count.sum());contexts+=sum(lengths[p] for p in chunk)
    except Exception as exc:error=type(exc).__name__+': '+str(exc)
    agree(error,None,group,world)
    return finish(view,result,before,dict(kind='code',rank=rank,world=world,profile=profile,global_positions=positions,local_positions=local,
        source_identity_sha256=data.identity_sha256,original_global_weighted_targets=denominator,
        original_global_response_EOS_targets=sum(row['response_EOS_targets'] for row in info),
        local_weighted_CE_sum=weighted,local_unweighted_CE_sum=unweighted,local_response_EOS_targets=targets,
        local_complete_context_tokens=contexts,actual_microbatch_positions=chunks,synthetic_tests_used_as_rewards=False),group)
