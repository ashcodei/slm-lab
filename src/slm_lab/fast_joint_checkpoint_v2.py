"""Complete counted model/Adam and private per-worker RNG joint snapshots."""
import math
import random

import numpy as np
import torch

from .codenet_checkpoint_v1 import restore_rng
from .fast_joint_inference_v2 import export,sources,FORMAT as NATIVE_FORMAT
from .deterministic_positions_v1 import PROTOCOL

FORMAT='own-fast-joint-complete-Adam-all-worker-RNG-v2'


def cpu(value):
    if isinstance(value,torch.Tensor):return value.detach().to(device='cpu',copy=True)
    if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu(v) for v in value)
    return value


def bundle(model,optimizer,plan,digest,completed,metrics,rngs,mode):
    return dict(format=FORMAT,mode=mode,model=export(model,plan,digest,'capacity' if mode=='capacity' else 'trained',completed),
        optimizer=cpu(optimizer.state_dict()),rngs=rngs,completed=completed,metrics=list(metrics),
        training_options=plan['training_options'],run_plan_sha256=digest,world_size=len(rngs),goal_achieved=False)


def restore(model,optimizer,payload,plan,digest,device,rank,world,mode):
    if (payload.get('format')!=FORMAT or payload.get('mode')!=mode or payload.get('run_plan_sha256')!=digest
            or payload.get('world_size')!=world or payload.get('training_options')!=plan['training_options']
            or type(payload.get('completed')) is not int or payload['completed']<1
            or len(payload.get('metrics',[]))!=payload['completed'] or len(payload.get('rngs',[]))!=world):
        raise ValueError('Exact joint profile, source cursor, metrics and every worker RNG required before mutation')
    raw=payload['model']
    if raw.get('format')!=NATIVE_FORMAT or raw.get('position_interpolation')!=PROTOCOL:
        raise ValueError('Exact deterministic-position computation required before joint recovery')
    model.validate_export(raw['model'],expected_sources=sources(plan,digest))
    if raw['completed_updates']!=payload['completed'] or raw['run_plan_sha256']!=digest:raise ValueError('Matching actual joint update cursors required')
    for state in payload['rngs']:
        random.Random().setstate(state['python']);n=state['numpy']
        np.random.RandomState().set_state((n['kind'],np.asarray(n['keys'],dtype=np.uint32),n['position'],n['has_gauss'],n['cached_gaussian']))
        torch.Generator(device='cpu').set_state(state['torch'].cpu())
        if torch.device(device).type=='cuda':
            if state['cuda'] is None:raise ValueError('Every actual private CUDA RNG state required')
            try:torch.Generator(device=device).set_state(state['cuda'].cpu())
            except (RuntimeError,TypeError) as e:raise ValueError('Invalid private CUDA RNG state') from e
        elif state['cuda'] is not None:raise ValueError('CPU recovery cannot import a CUDA worker profile')
    saved=payload['optimizer'];live=optimizer.state_dict()
    if len(saved['param_groups'])!=len(live['param_groups']):raise ValueError('Exact joint Adam parameter groups required')
    shapes={}
    for actual,sg,lg in zip(optimizer.param_groups,saved['param_groups'],live['param_groups']):
        if sg['params']!=lg['params'] or sg['name']!=lg['name'] or tuple(sg['betas'])!=tuple(lg['betas']) or sg['weight_decay']!=lg['weight_decay']:
            raise ValueError('Exact named full-weight Adam partition and hyperparameters required')
        shapes.update({key:p.shape for key,p in zip(sg['params'],actual['params'])})
    for key,state in saved['state'].items():
        if key not in shapes or set(state)!={'step','exp_avg','exp_avg_sq'} or state['exp_avg'].shape!=shapes[key] or state['exp_avg_sq'].shape!=shapes[key]:
            raise ValueError('Complete actual Adam moments must match their counted parameter')
        if (any(not bool(torch.isfinite(state[k]).all()) for k in state) or state['step'].numel()!=1
                or not 1<=float(state['step'])<=payload['completed']):raise ValueError('Finite actual per-leaf Adam cursor and moments required')
    model.import_export(raw['model'],expected_sources=sources(plan,digest));optimizer.load_state_dict(saved)
    restore_rng(payload['rngs'][rank],device)
    return payload['completed'],list(payload['metrics'])
