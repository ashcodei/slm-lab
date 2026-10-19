"""Matched original-function warm-up before execution RL.

The control repeats the original instruction three times per function. The
experimental arm uses the original, exact completion and verified repair
(completion again where no repair qualifies). Functions and optimizer clocks
match; target-token counts and computational cost need not match. All targets
are exact public train source slices. This is supervised learning, not RLHF.
"""
from collections import Counter
import gzip
import json
import math
from pathlib import Path
import random

import torch
from torch import distributed as dist
from torch.nn import functional as F
from tokenizers import Tokenizer

from .codenet_token_pack_v1 import CompleteCodePack,packed_encoder
from .own_dense_actor_checkpoint_v1 import make_state,restore_state,selected

FORMAT='own-verified-function-views-supervised-warmup-v1'


class VerifiedViewsWarmup:
    def __init__(self,model,optimizer,root,*,method,epochs=4,functions_per_update=8,seed=957713,
                 KL_weight=.01,world=1,rank=0,gradient_group=None,control_group=None):
        self.named=selected(model,optimizer,('actor',));self.parameters=[p for _,p in self.named]
        if (method not in ('original','views') or type(epochs) is not int or epochs<1 or type(seed) is not int or seed<1
                or type(functions_per_update) is not int or functions_per_update<1
                or type(world) is not int or world<1 or functions_per_update<world or type(rank) is not int or not 0<=rank<world
                or not math.isfinite(KL_weight) or KL_weight<0 or any(p.dtype!=torch.float32 for p in model.parameters())):
            raise ValueError('Explicit complete FP32 source/function/update/objective profile required')
        if world>1 and (not dist.is_initialized() or dist.get_world_size(gradient_group)!=world or dist.get_world_size(control_group)!=world):
            raise ValueError('Exact initialized arm replicas required')
        root=Path(root).resolve();folder=root/'data/verified-code-views-v1'
        self.pack=CompleteCodePack(folder,'train','training')
        with gzip.open(folder/'original-rows.jsonl.gz','rt') as f:self.rows=list(map(json.loads,f))
        self.by_function={}
        for i,row in enumerate(self.rows):
            key=row['original_source_position'];self.by_function.setdefault(key,{})[row['view']]=i
        if sorted(self.by_function)!=list(range(len(self.by_function))):raise ValueError('Every contiguous original source position required')
        for modes in self.by_function.values():
            if not {'original_instruction','prefix_completion'}<=modes.keys():raise ValueError('Every complete original and completion required')
        self.model,self.optimizer=model,optimizer;self.method=method;self.epochs=epochs;self.width=functions_per_update
        self.seed,self.KL_weight,self.world,self.rank=seed,KL_weight,world,rank
        self.gradient_group,self.control_group=gradient_group,control_group
        self.steps_per_epoch=math.ceil(len(self.by_function)/self.width);self.planned=self.steps_per_epoch*epochs
        if len(self.by_function)%self.width and len(self.by_function)%self.width<world:raise ValueError('Final unrepeated function group must keep each replica nonempty')
        self.tokenizer=Tokenizer.from_file(str(folder/'tokenizer.json'));self.encoder=packed_encoder(self.pack,self.tokenizer,model.config)
        self.orders=[]
        for epoch in range(epochs):
            order=list(self.by_function);random.Random(seed+epoch).shuffle(order);self.orders.append(order)
        self.fixed=[p for p in model.parameters() if id(p) not in {id(x) for x in self.parameters}]
        self.fixed_signature=self.signature(self.fixed);self.completed=0;self.metrics=[]

    @staticmethod
    def signature(parameters):return [(id(p),p._version,p.data_ptr(),p.requires_grad) for p in parameters]

    def settings(self):return dict(method=self.method,epochs=self.epochs,functions_per_update=self.width,seed=self.seed,
        KL_weight=self.KL_weight,replicas=self.world,replica_rank=self.rank,original_functions=len(self.by_function),
        complete_views=len(self.pack),presentations_per_function=3,planned_updates=self.planned,
        precision='FP32_TF32_disabled',normalization='global_complete_response_EOS_tokens',gradient_clip=1.)

    def positions(self,step):
        if type(step) is not int or not 0<=step<self.planned:raise ValueError('Bounded whole-function update required')
        epoch,within=divmod(step,self.steps_per_epoch);return self.orders[epoch][within*self.width:(within+1)*self.width]

    def views(self,functions):
        out=[]
        for position in functions:
            modes=self.by_function[position];original=modes['original_instruction'];completion=modes['prefix_completion']
            out.extend([original]*3 if self.method=='original' else [original,completion,modes.get('verified_statement_repair',completion)])
        return out

    def gather(self,value):
        if self.world==1:return [value]
        results=[None]*self.world;dist.all_gather_object(results,value,group=self.control_group);return results

    def update(self):
        if self.completed>=self.planned or self.signature(self.fixed)!=self.fixed_signature:raise ValueError('Unchanged frozen base/critic and unconsumed source group required')
        functions=self.positions(self.completed);all_views=self.views(functions);local_views=self.views(functions[self.rank::self.world])
        global_targets=sum(int(self.pack.index[self.pack.positions[p]]['length']-self.pack.index[self.pack.positions[p]]['target_start']) for p in all_views)
        device=next(self.model.parameters()).device;self.model.train();self.model.policy.gradient_checkpointing=True
        self.optimizer.zero_grad(set_to_none=True);CE=KL=0.;targets=context=0
        try:
            with torch.autocast(device_type=device.type,enabled=False):
                for position in local_views:
                    b=self.encoder.batch([dict(packed_code_index=position)],'response',device);inputs={k:v for k,v in b.items() if k!='labels'}
                    logits=self.model(**inputs)['logits'][:,:-1];labels=b['labels'][:,1:];mask=labels!=-100;student=logits[mask].float()
                    ce=F.cross_entropy(student,labels[mask],reduction='sum');kl=ce.detach()*0
                    if self.KL_weight:
                        with torch.no_grad():reference=self.model.base(**inputs)['logits'][:,:-1][mask].float()
                        logp=student.log_softmax(-1);logq=reference.log_softmax(-1);kl=(logq.exp()*(logq-logp)).sum()
                    loss=(ce+self.KL_weight*kl)/global_targets
                    if not bool(torch.isfinite(loss)):raise ValueError('Finite complete public-target objective required')
                    loss.backward(inputs=self.parameters);CE+=float(ce.detach());KL+=float(kl.detach())
                    targets+=int(mask.sum());context+=int(b['attention_mask'].sum())
                    del logits,student,loss,ce,kl,b
                    if self.KL_weight:del reference,logp,logq
            if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in self.parameters) or any(p.grad is not None for p in self.fixed):
                raise ValueError('Every selected actor and no fixed tensor needs finite gradients')
            if self.world>1:
                flat=torch.cat([p.grad.reshape(-1) for p in self.parameters]);dist.all_reduce(flat,group=self.gradient_group);offset=0
                for p in self.parameters:p.grad.copy_(flat[offset:offset+p.numel()].view_as(p));offset+=p.numel()
                del flat
            norm=float(torch.nn.utils.clip_grad_norm_(self.parameters,1.,error_if_nonfinite=True));self.optimizer.step();self.optimizer.zero_grad(set_to_none=True)
        except BaseException:self.optimizer.zero_grad(set_to_none=True);raise
        observations=self.gather(dict(CE_sum=CE,KL_sum=KL,targets=targets,context=context))
        if sum(o['targets'] for o in observations)!=global_targets:raise ValueError('Every complete global target must be counted exactly once')
        ce=math.fsum(o['CE_sum'] for o in observations);kl=math.fsum(o['KL_sum'] for o in observations)
        record=dict(step=self.completed+1,method=self.method,original_function_positions=functions,complete_view_positions=all_views,
            complete_presentations=len(all_views),complete_context_tokens=sum(o['context'] for o in observations),
            response_EOS_targets=global_targets,CE_sum=ce,KL_sum=kl,token_mean_CE=ce/global_targets,
            token_mean_reference_KL=kl/global_targets,KL_weight=self.KL_weight,gradient_norm_before_clip=norm,
            view_exposures=dict(Counter(self.rows[p]['view'] for p in all_views)),supervised_not_RL=True,
            genuine_human_preference_RLHF=False,goal_achieved=False)
        self.completed+=1;self.metrics.append(record);return record

    def cursor(self,completed):return dict(completed=completed,orders=self.orders,settings=self.settings())

    def state(self,sources):
        return dict(format=FORMAT,settings=self.settings(),core=make_state(self.model,self.optimizer,groups=('actor',),
            sources=sources,completed=self.completed,planned=self.planned,cursor=self.cursor(self.completed),metrics=self.metrics,
            device=next(self.model.parameters()).device),private_state_is_not_whole_distributed_checkpoint=True)

    def resume(self,state,sources,*,allow_completed=False):
        core=state.get('core',{});completed=core.get('completed')
        if (state.get('format')!=FORMAT or state.get('settings')!=self.settings() or type(completed) is not int
                or not 0<completed<=self.planned or len(core.get('metrics',[]))!=completed):raise ValueError('Exact warm-up settings/source cursor required')
        for i,r in enumerate(core['metrics']):
            views=self.views(self.positions(i));ids_labels=[self.pack.sequence(p) for p in views]
            targets=sum(sum(y!=-100 for y in labels) for ids,labels in ids_labels)
            if (r.get('step')!=i+1 or r.get('method')!=self.method or r.get('original_function_positions')!=self.positions(i)
                    or r.get('complete_view_positions')!=views or r.get('response_EOS_targets')!=targets
                    or r.get('complete_context_tokens')!=sum(len(ids) for ids,labels in ids_labels)
                    or r.get('token_mean_CE')!=r['CE_sum']/targets or r.get('token_mean_reference_KL')!=r['KL_sum']/targets
                    or r.get('KL_weight')!=self.KL_weight):raise ValueError('Every actual warm-up source/complete-target denominator must reconstruct')
        self.completed,self.metrics=restore_state(self.model,self.optimizer,core,groups=('actor',),expected_sources=sources,
            planned=self.planned,expected_cursor=self.cursor(completed),device=next(self.model.parameters()).device,allow_completed=allow_completed)
