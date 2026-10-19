"""Experimental future-order supervision with a shared vocabulary basis.

Inspired by https://arxiv.org/html/2508.19228v2, not a reproduction: a small
low-rank residual head shares our frozen native unembedding instead of adding
the paper's full vocabulary projection. Targets use first future occurrences,
only within the complete supervised response through its original EOS.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .model import RMSNorm
from .own_dense_actor_v1 import OwnDenseActor

FORMAT='own-compact-shared-basis-future-order-auxiliary-v1'


def targets(input_ids,labels,attention_mask,*,window,vocab_size):
    if (type(window) is not int or not 1<=window<=64 or type(vocab_size) is not int or vocab_size<4
            or input_ids.ndim!=2 or input_ids.shape!=labels.shape or input_ids.shape!=attention_mask.shape
            or input_ids.dtype!=torch.long or labels.dtype!=torch.long or attention_mask.dtype!=torch.bool
            or input_ids.shape[1]<2 or any(t.device!=input_ids.device for t in (labels,attention_mask))
            or bool(((input_ids<0)|(input_ids>=vocab_size)).any())
            or bool(((labels!=-100)&(labels!=input_ids)).any())):
        raise ValueError('Complete original native inputs, labels, Boolean padding and a bounded future window required')
    for i in range(input_ids.shape[0]):
        active=torch.nonzero(labels[i]!=-100).flatten();visible=torch.nonzero(attention_mask[i]).flatten()
        if (active.numel()<2 or visible.numel()==0 or int(active[0])<1
                or not torch.equal(active,torch.arange(int(active[0]),int(active[-1])+1,device=active.device))
                or not torch.equal(visible,torch.arange(int(visible[-1])+1,device=visible.device))
                or int(active[-1])!=int(visible[-1]) or int(labels[i,active[-1]])!=2
                or bool((labels[i,active[:-1]]==2).any()) or bool(((labels[i]!=-100)&~attention_mask[i]).any())):
            raise ValueError('Every whole single response, its final EOS and right padding must remain; no truncation or cross-response targets')
    positions=torch.nonzero(labels[:,1:]!=-100)
    if positions.numel()==0:raise ValueError('Every auxiliary position must also have an original next-token target')
    offsets=torch.arange(1,window+1,device=input_ids.device)
    future=positions[:,1,None]+offsets[None,:];inside=future<input_ids.shape[1]
    safe=future.clamp_max(input_ids.shape[1]-1);rows=positions[:,0,None]
    ids=input_ids[rows,safe];valid=inside&(labels[rows,safe]!=-100)
    # A repeated token is ranked by its first *future* occurrence, even when
    # it also equals the token at the current input position.
    first=valid.clone()
    for j in range(1,window):first[:,j]&=~((ids[:,:j]==ids[:,j,None])&valid[:,:j]).any(-1)
    scores=(-offsets.float())[None,:].expand_as(ids).masked_fill(~first,float('-inf'))
    probabilities=scores.softmax(-1)
    if not bool(torch.isfinite(probabilities).all()) or not bool(first[:,0].all()):
        raise ValueError('Every original response/EOS target must receive a finite first-future ranking')
    return positions,ids,probabilities,first


class CompactFutureOrder(nn.Module):
    def __init__(self,actor,*,rank=32,window=16,seed=973331,parameter_limit=10**9):
        super().__init__()
        if (type(actor) is not OwnDenseActor or type(rank) is not int or not 1<=rank<=actor.config.hidden_size
                or type(window) is not int or not 1<=window<=64 or type(seed) is not int or seed<1
                or type(parameter_limit) is not int or not 0<parameter_limit<=10**9):
            raise ValueError('Explicit own counted policy and compact future-order head required')
        actor.assert_shared();hidden=actor.config.hidden_size;extra=2*hidden*rank+hidden
        if actor.counts()['actual_unique']+extra>=parameter_limit:
            raise ValueError('Every native, policy, critic and auxiliary value must remain strictly under the parameter limit')
        device=next(actor.parameters()).device
        if any(p.dtype!=torch.float32 or p.device!=device for p in actor.parameters()):
            raise ValueError('This research objective has an explicit shared FP32 parameter contract')
        self.actor=actor;self.settings=dict(rank=rank,window=window,seed=seed,parameter_limit=parameter_limit)
        devices=[] if device.type=='cpu' else [device.index]
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(seed)
            for d in devices:torch.cuda.default_generators[d].manual_seed(seed)
            self.future=nn.ModuleDict(dict(down=nn.Linear(hidden,rank,bias=False,device=device),
                up=nn.Linear(rank,hidden,bias=False,device=device),norm=RMSNorm(hidden,actor.config.norm_eps).to(device)))
            nn.init.normal_(self.future['down'].weight,std=.02);nn.init.zeros_(self.future['up'].weight)
        self.config=actor.config;self.extra=extra;self.eval();self.counts()

    def counts(self):
        actual=sum(p.numel() for p in self.parameters());extra=sum(p.numel() for p in self.future.parameters())
        if extra!=self.extra or actual!=self.actor.counts()['actual_unique']+extra:
            raise ValueError('Exactly one counted native/policy and only the declared compact head required')
        return dict(policy_native_critic_values=self.actor.counts()['actual_unique'],auxiliary_values=extra,total_unique=actual,
                    additional_vocabulary_or_vision_encoder_values=0)

    def named_selected(self,*,auxiliary):
        rows=[('actor.'+n,p) for n,p in self.actor.named_selected(('actor',))]
        if auxiliary:rows.extend(('future.'+n,p) for n,p in self.future.named_parameters())
        return rows

    def forward(self,*args,**kwargs):return self.actor(*args,**kwargs)
    def generate(self,*args,**kwargs):return self.actor.generate(*args,**kwargs)

    def auxiliary_sum(self,hidden,input_ids,labels,attention_mask,*,chunk_positions=32,recompute_chunks=True):
        if (type(chunk_positions) is not int or not 1<=chunk_positions<=256 or type(recompute_chunks) is not bool
                or hidden.shape!=(*input_ids.shape,self.config.hidden_size)):
            raise ValueError('All original text hidden positions and bounded exact auxiliary chunks required')
        positions,ids,probabilities,first=targets(input_ids,labels,attention_mask,window=self.settings['window'],vocab_size=self.config.vocab_size)
        weight=self.actor.base.network.embedding.weight.detach();loss=hidden.sum()*0
        def part(states,token_ids,weights):
            transformed=self.future['norm'](states+self.future['up'](self.future['down'](states)))
            logits=F.linear(transformed,weight).float()
            return (logits.logsumexp(-1)-(logits.gather(1,token_ids)*weights).sum(-1)).sum()
        for start in range(0,len(positions),chunk_positions):
            pos=positions[start:start+chunk_positions];states=hidden[pos[:,0],pos[:,1]]
            args=(states,ids[start:start+chunk_positions],probabilities[start:start+chunk_positions])
            loss=loss+(checkpoint(part,*args,use_reentrant=False) if recompute_chunks and torch.is_grad_enabled() else part(*args))
        return loss,dict(original_auxiliary_positions=len(positions),unique_future_occurrence_targets=int(first.sum()),
            original_next_token_targets=int((labels[:,1:]!=-100).sum()),window=self.settings['window'],
            first_future_occurrences_only=True,original_EOS_included=True,no_future_cross_response_or_padding=True,
            dense_vocabulary_gold_targets_allocated=False,chunk_positions=chunk_positions,
            recomputed_auxiliary_chunks=recompute_chunks,full_vocab_normalization=True)

    def objective(self,batch,*,auxiliary_weight=.1,chunk_positions=32,recompute_chunks=True):
        if (type(auxiliary_weight) not in (float,int) or not math.isfinite(auxiliary_weight) or not 0<=auxiliary_weight<=1
                or batch.get('pixel_values') is not None or batch.get('labels') is None or batch.get('attention_mask') is None):
            raise ValueError('A declared text-only whole-response CE/auxiliary experiment is required')
        # Validate source boundaries in the CE-only control too.
        targets(batch['input_ids'],batch['labels'],batch['attention_mask'],window=self.settings['window'],vocab_size=self.config.vocab_size)
        with torch.autocast(next(self.parameters()).device.type,enabled=False):
            output=self.actor(**batch,return_hidden=bool(auxiliary_weight))
            count=int((batch['labels'][:,1:]!=-100).sum())
            if auxiliary_weight:
                auxiliary,record=self.auxiliary_sum(output['hidden'],batch['input_ids'],batch['labels'],batch['attention_mask'],
                    chunk_positions=chunk_positions,recompute_chunks=recompute_chunks)
                mean=auxiliary/count;loss=output['loss']+auxiliary_weight*mean
            else:
                mean=output['loss'].detach()*0;loss=output['loss'];record=dict(original_auxiliary_positions=0,original_next_token_targets=count)
        return loss,dict(record,format=FORMAT,original_CE=float(output['loss'].detach()),auxiliary_token_mean=float(mean.detach()),
            auxiliary_weight=float(auxiliary_weight),total_objective=float(loss.detach()),actual_counts=self.counts(),
            extra_auxiliary_compute_not_equal_FLOPs=True,useful_code_or_general_capability_not_proved_by_objective=True)
