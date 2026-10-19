"""Whole-source attention epoch, exact collective snapshots and frozen text.

Only original vision and the new head train. This kernel neither selects a
base policy nor acquires GPUs. A caller must own the resource window.
"""
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import json
import math
import random

import torch
from torch import distributed as dist

from .codenet_checkpoint_v1 import restore_rng,save_collective
from .point_grounding_head_v1 import PointGroundingHead


def cpu(value):
    if isinstance(value,torch.Tensor):
        return value.detach().cpu()
    if isinstance(value,dict):
        return {k:cpu(v) for k,v in value.items()}
    if isinstance(value,list):
        return [cpu(v) for v in value]
    if isinstance(value,tuple):
        return tuple(cpu(v) for v in value)
    return value


def frozen_digest(head):
    digest=hashlib.sha256()
    for name,value in sorted(head.base.state_dict().items()):
        if name.startswith('vision.'):
            continue
        digest.update(json.dumps([name,list(value.shape),str(value.dtype)],separators=(',',':')).encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def collective_check(function):
    value=None;error=None
    try:
        value=function()
    except Exception as problem:
        error=type(problem).__name__+': '+str(problem)
    if dist.is_initialized():
        errors=[None]*dist.get_world_size();dist.all_gather_object(errors,error)
    else:
        errors=[error]
    if any(errors):
        raise ValueError('Collective attention preflight failed: '+repr(errors))
    return value


def make_optimizer(head,options):
    vision=[p for p in head.base.vision.parameters() if p.requires_grad]
    projections=[*head.query_projection.parameters(),*head.patch_projection.parameters()]
    return torch.optim.AdamW([
        dict(params=vision,lr=options['vision_learning_rate'],base_lr=options['vision_learning_rate'],name='vision'),
        dict(params=projections,lr=options['head_learning_rate'],base_lr=options['head_learning_rate'],name='head')],
        betas=tuple(options['betas']),weight_decay=options['weight_decay'],foreach=False)


class PointerStage:
    def __init__(self,head,optimizer,data,options,sources,device,amp=None):
        if type(head) is not PointGroundingHead or data.role!='train' or data.purpose!='optimization':
            raise ValueError('Own grounding architecture and admitted original optimization role required')
        self.world=dist.get_world_size() if dist.is_initialized() else 1
        self.rank=dist.get_rank() if dist.is_initialized() else 0
        self.options=dict(options);self.sources=dict(sources);self.device=torch.device(device)
        for key in ['microbatch_per_rank','accumulation','global_queries_per_update','seed']:
            if type(options[key]) is not int or options[key]<1:
                raise ValueError('Explicit positive source sampling options required')
        if options['world_size']!=self.world or options['global_queries_per_update']!=self.world*options['microbatch_per_rank']*options['accumulation']:
            raise ValueError('Actual collective query population must match the sealed global batch')
        if (not len(data) or options['epochs']!=1 or options['steps']!=math.ceil(len(data)/options['global_queries_per_update'])
                or options['planned_complete_query_exposures']!=options['steps']*options['global_queries_per_update']
                or options['rounding_repeated_queries']!=options['planned_complete_query_exposures']-len(data)):
            raise ValueError('Exactly one complete source pass and explicit whole-query rounding required')
        self.head=head;self.optimizer=optimizer;self.data=data;self.steps=options['steps'];self.completed=0;self.metrics=[]
        self.parameters=[p for p in head.parameters() if p.requires_grad]
        self.frozen=[p for p in head.parameters() if not p.requires_grad]
        actual=[p for group in optimizer.param_groups for p in group['params']]
        if (len(actual)!=len(self.parameters) or {id(p) for p in actual}!={id(p) for p in self.parameters}
                or any(p.requires_grad for n,p in head.base.named_parameters() if not n.startswith('vision.'))):
            raise ValueError('Exactly original vision/new head train; no frozen weight belongs in Adam')
        head.trainable_inventory()
        self.order=list(range(len(data)));random.Random(options['seed']).shuffle(self.order)
        self.order_sha256=hashlib.sha256(json.dumps(self.order,separators=(',',':')).encode()).hexdigest()
        self.amp=amp if amp is not None else nullcontext
        # DDP rebuilds its gradient buckets after initial iterations. A new
        # wrapper after resume may sum in a different floating-point order.
        # Reduce once per update in parameter order, as in the existing exact
        # complete-code kernel, keeping accumulation/recovery order stable.
        self.network=head

    def positions(self,step,rank,micro):
        if not 0<=step<self.steps or not 0<=rank<self.world or not 0<=micro<self.options['accumulation']:
            raise ValueError('Actual sealed collective source draw required')
        width=self.options['microbatch_per_rank']
        start=step*self.options['global_queries_per_update']+(micro*self.world+rank)*width
        return [self.order[(start+i)%len(self.order)] for i in range(width)]

    def schedule(self,step):
        warm=max(1,math.ceil(self.steps*self.options['warmup_fraction']))
        if step<warm:
            return (step+1)/warm
        progress=(step-warm)/max(1,self.steps-warm)
        floor=self.options['cosine_floor_fraction']
        return floor+(1-floor)*.5*(1+math.cos(math.pi*progress))

    def update(self):
        if self.completed>=self.steps:
            raise ValueError('Fixed complete original attention epoch already finished')
        self.head.train();self.optimizer.zero_grad(set_to_none=True)
        factor=self.schedule(self.completed)
        for group in self.optimizer.param_groups:
            group['lr']=group['base_lr']*factor
        local=[];loss_sum=0.
        for micro in range(self.options['accumulation']):
            positions=self.positions(self.completed,self.rank,micro)
            def encode():
                batch=self.data.batch(positions,self.device)
                if (batch['pixel_values'].shape!=(len(positions),3,self.options['image_size'],self.options['image_size'])
                        or batch['input_ids'].shape[1]>self.options['maximum_whole_query_tokens']
                        or batch['attention_mask'].shape!=batch['input_ids'].shape):
                    raise ValueError('Unclipped exact original image and whole-query budget required')
                return batch
            batch=collective_check(encode)
            with nullcontext():
                def forward():
                    with self.amp():
                        loss=self.network(**batch)['loss']
                    if not bool(torch.isfinite(loss)):
                        raise ValueError('Finite original attention objective required')
                    return loss
                loss=collective_check(forward)
                (loss/self.options['accumulation']).backward()
            loss_sum+=float(loss.detach())/self.options['accumulation']
            local.append(dict(positions=positions,original_attention_samples=int(batch['targets'].sum()),
                complete_query_tokens=int(batch['attention_mask'].sum())))
        def check_present():
            if any(p.grad is None for p in self.parameters) or any(p.grad is not None for p in self.frozen):
                raise ValueError('All original vision/head gradients and no frozen-language/critic gradients required')
        collective_check(check_present)
        if self.world>1:
            for parameter in self.parameters:
                dist.all_reduce(parameter.grad,op=dist.ReduceOp.SUM);parameter.grad.div_(self.world)
        def check_gradients():
            return float(torch.nn.utils.clip_grad_norm_(self.parameters,self.options['gradient_norm_clip'],error_if_nonfinite=True))
        norm=collective_check(check_gradients)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        observation=dict(rank=self.rank,micros=local,query_mean_nll=loss_sum)
        if self.world>1:
            observations=[None]*self.world;dist.all_gather_object(observations,observation)
        else:
            observations=[observation]
        self.completed+=1
        record=dict(step=self.completed,query_mean_nll=sum(r['query_mean_nll'] for r in observations)/self.world,
            global_queries=self.options['global_queries_per_update'],
            original_attention_samples=sum(m['original_attention_samples'] for r in observations for m in r['micros']),
            whole_query_tokens=sum(m['complete_query_tokens'] for r in observations for m in r['micros']),
            gradient_norm_before_clipping=norm,vision_learning_rate=self.optimizer.param_groups[0]['lr'],
            head_learning_rate=self.optimizer.param_groups[1]['lr'],workers=observations)
        self.metrics.append(record)
        return record

    def sampler_state(self):
        return dict(completed_steps=self.completed,population=len(self.data),seed=self.options['seed'],
            source_exposures=self.completed*self.options['global_queries_per_update'],order_sha256=self.order_sha256)

    def validate_Adam(self):
        if len(self.optimizer.state)!=len(self.parameters):
            raise ValueError('Every original vision/head parameter needs complete Adam state')
        for parameter in self.parameters:
            state=self.optimizer.state[parameter]
            if (float(state['step'])!=self.completed or any(key not in state or state[key].shape!=parameter.shape
                    or not bool(torch.isfinite(state[key]).all()) for key in ['exp_avg','exp_avg_sq'])
                    or not bool(torch.isfinite(parameter).all())):
                raise ValueError('Exact finite vision/head Adam steps, tensors and moments required')

    def save(self,path,maximum_bytes,minimum_free_bytes):
        collective_check(self.validate_Adam)
        def bundle(states):
            return dict(training_kind='original_human_pointer_vision_head_v1',model=cpu(self.head.state_dict()),
                optimizer=cpu(self.optimizer.state_dict()),config=asdict(self.head.base.config),head_rank=self.head.rank,
                steps=self.completed,planned_steps=self.steps,world_size=self.world,training_options=self.options,
                sources=self.sources,sampler_state=self.sampler_state(),worker_rng_states=states,
                metrics=self.metrics,trainable_parameter_names=[n for n,p in self.head.named_parameters() if p.requires_grad],
                actual_inventory=self.head.trainable_inventory(),model_promoted=False,goal_achieved=False)
        save_collective(path,bundle,self.device,self.world,self.rank,maximum_bytes,minimum_free_bytes)

    def resume(self,state):
        if (state.get('training_kind')!='original_human_pointer_vision_head_v1' or state['sources']!=self.sources
                or state['world_size']!=self.world or state['training_options']!=self.options
                or state['config']!=asdict(self.head.base.config) or state['head_rank']!=self.head.rank
                or state['planned_steps']!=self.steps or not 0<state['steps']<self.steps
                or len(state['worker_rng_states'])!=self.world or len(state['metrics'])!=state['steps']
                or state['trainable_parameter_names']!=[n for n,p in self.head.named_parameters() if p.requires_grad]):
            raise ValueError('Exact own point-model/source/optimizer/budget/four-worker resume contract required')
        expected=dict(self.sampler_state(),completed_steps=state['steps'],source_exposures=state['steps']*self.options['global_queries_per_update'])
        if state['sampler_state']!=expected:
            raise ValueError('Exact original attention source order/cursor required')
        # Validate actual tensor, Adam and local RNG schemas before changing
        # any live parameter, optimizer moment, source cursor or global RNG.
        current=self.head.state_dict();trainable=set(state['trainable_parameter_names'])
        if state['model'].keys()!=current.keys():
            raise ValueError('Every actual own model tensor is required for resume')
        for name,tensor in state['model'].items():
            if (tensor.shape!=current[name].shape or tensor.dtype!=current[name].dtype
                    or not bool(torch.isfinite(tensor).all())
                    or (name not in trainable and not torch.equal(tensor.cpu(),current[name].detach().cpu()))):
                raise ValueError('Finite exact model tensors and unchanged frozen original language required')
        groups=state['optimizer']['param_groups']
        if len(groups)!=len(self.optimizer.param_groups):
            raise ValueError('Exact original optimizer groups required')
        parameter_ids=[]
        for group,live_group in zip(groups,self.optimizer.param_groups):
            if len(group['params'])!=len(live_group['params']) or any(group[k]!=v for k,v in live_group.items() if k not in ['params','lr']):
                raise ValueError('Exact original Adam options and parameter order required')
            parameter_ids.extend(group['params'])
        if len(set(parameter_ids))!=len(self.parameters) or set(parameter_ids)!=state['optimizer']['state'].keys():
            raise ValueError('Every original trainable tensor needs exactly one Adam state')
        for key,parameter in zip(parameter_ids,self.parameters):
            entry=state['optimizer']['state'][key]
            if (float(entry['step'])!=state['steps'] or any(entry[k].shape!=parameter.shape
                    or not bool(torch.isfinite(entry[k]).all()) for k in ['exp_avg','exp_avg_sq'])
                    or bool((entry['exp_avg_sq']<0).any())):
                raise ValueError('Finite exact original Adam moments and steps required before resume')
        for rng in state['worker_rng_states']:
            random.Random().setstate(rng['python'])
            torch.Generator(device='cpu').set_state(rng['torch'].cpu())
            import numpy as np
            n=rng['numpy'];np.random.RandomState(0).set_state((n['kind'],np.asarray(n['keys'],dtype=np.uint32),n['position'],n['has_gauss'],n['cached_gaussian']))
            if (rng['cuda'] is None)!=(self.device.type!='cuda'):
                raise ValueError('Original exact CPU/CUDA RNG device contract required')
            if self.device.type=='cuda':
                torch.Generator(device=self.device).set_state(rng['cuda'].cpu())
        self.head.load_state_dict(state['model'],strict=True);self.optimizer.load_state_dict(state['optimizer'])
        self.completed=state['steps'];self.metrics=list(state['metrics']);self.validate_Adam()
        restore_rng(state['worker_rng_states'][self.rank],self.device)
