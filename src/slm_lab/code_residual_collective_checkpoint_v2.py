"""Complete shared-adapter state plus exact per-rank RNG; no training launcher.

Collective validation failures reach every worker before weights, Adam or RNG
change. Hardware/commit failures stop all workers and require inspection.
"""
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import distributed as dist

from .code_residual_checkpoint_v1 import (_optimizer_parameters,_validate_adam,
    _validate_rng,_digest,make_training_state)
from .codenet_checkpoint_v1 import (FileBindings,atomic_checkpoint,collect_worker_rng,restore_rng)
from .corpus import file_hash
from .recurrent_ablation import state_digest

FORMAT='own-code-residual-collective-state-v2'


def fingerprint(value):
    """Value fingerprint including scalar tensor bytes and ordered container types."""
    h=hashlib.sha256()
    def visit(x):
        # state_dict() returns OrderedDict; the lossless CPU snapshot uses dict.
        # Named tensor mappings have the same value semantics, with parameter
        # order bound separately by residual_parameter_names.
        h.update(('mapping' if isinstance(x,dict) else type(x).__name__).encode())
        if isinstance(x,torch.Tensor):
            t=x.detach().cpu().contiguous();h.update(str((t.dtype,tuple(t.shape))).encode())
            h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x,dict):
            h.update(str(len(x)).encode())
            for k in sorted(x,key=lambda k:(type(k).__name__,repr(k))):visit(k);visit(x[k])
        elif isinstance(x,(list,tuple)):
            h.update(str(len(x)).encode())
            for item in x:visit(item)
        elif x is None or type(x) in [str,int,bool]:h.update(repr(x).encode())
        elif type(x) is float:
            if not math.isfinite(x):raise ValueError('Finite metadata/RNG scalar required')
            h.update(x.hex().encode())
        else:raise ValueError('Unsupported checkpoint fingerprint value')
    visit(value);return h.hexdigest()


def _group():
    if not dist.is_initialized() or dist.get_world_size()<2:
        raise ValueError('An initialized multiworker process group is required')
    return dist.get_world_size(),dist.get_rank()


def _validated(fn):
    world,_=_group();error=None;value=None
    try:value=fn()
    except Exception as e:error=type(e).__name__+': '+str(e)
    errors=[None]*world;dist.all_gather_object(errors,error)
    if any(errors):raise ValueError('Collective preflight failed: '+repr(errors))
    return value


def _same(value):
    world,_=_group();values=[None]*world;dist.all_gather_object(values,value)
    if len(set(values))!=1:raise ValueError('Workers disagree on exact model/Adam/source/progress state')


def _metadata(model,sources,steps,planned_steps,sampler_state):
    if (not all(_digest(v) for v in sources.values()) or type(steps) is not int
            or type(planned_steps) is not int or not 0<steps<=planned_steps
            or not isinstance(sampler_state,dict) or type(sampler_state.get('updates')) is not int
            or sampler_state.get('updates')!=steps):
        raise ValueError('Exact source hashes, update budget and plan-derived sampler counters required')
    json.dumps(sampler_state,allow_nan=False)
    return dict(**sources,steps=steps,planned_steps=planned_steps,sampler_state=sampler_state,
        config=asdict(model.config),residual_width=model.width,model_counts=model.parameter_counts(),
        residual_parameter_names=[n for n,_ in model.residuals.named_parameters()])


def save_collective_residual(path,model,optimizer,*,base_checkpoint_sha256,tokenizer_sha256,
        run_plan_sha256,completed_steps,planned_steps,sampler_state,metrics,device,
        maximum_bytes,minimum_free_bytes):
    world,rank=_group()
    sources=dict(base_checkpoint_sha256=base_checkpoint_sha256,tokenizer_sha256=tokenizer_sha256,
        run_plan_sha256=run_plan_sha256)
    def preflight():
        parameters=_optimizer_parameters(model,optimizer)
        if any(p.grad is not None for p in model.parameters()):raise ValueError('Snapshot only at a clean optimizer boundary')
        _validate_adam(optimizer.state_dict(),parameters,completed_steps)
        if (not isinstance(metrics,list) or [m.get('step') for m in metrics]!=list(range(1,completed_steps+1))):
            raise ValueError('Every completed update needs one ordered metric')
        json.dumps(metrics,allow_nan=False)
        meta=_metadata(model,sources,completed_steps,planned_steps,sampler_state)
        return meta,fingerprint(dict(meta=meta,base=state_digest(model.base.state_dict()),
            residuals=model.residuals.state_dict(),optimizer=optimizer.state_dict()))
    meta,signature=_validated(preflight);_same(signature)
    def io_contract():
        if type(maximum_bytes) is not int or maximum_bytes<1 or type(minimum_free_bytes) is not int or minimum_free_bytes<0:
            raise ValueError('Explicit positive complete-checkpoint cap and nonnegative reserve required')
        return fingerprint(dict(path=str(Path(path).resolve()),maximum_bytes=maximum_bytes,minimum_free_bytes=minimum_free_bytes))
    _same(_validated(io_contract))
    rngs=collect_worker_rng(device,world)
    _validated(lambda:[_validate_rng(r,device) for r in rngs])
    outcome=[None]
    if rank==0:
        try:
            state=make_training_state(model,optimizer,**sources,completed_steps=completed_steps,
                planned_steps=planned_steps,device=device,metrics=metrics)
            state.pop('rng');state.update(format=FORMAT,world_size=world,worker_rng_states=rngs,
                sampler_state=sampler_state,shared_state_fingerprint=signature,
                residual_state_sha256=state_digest(state['residuals']),
                optimizer_state_sha256=fingerprint(state['optimizer']),worker_rng_states_sha256=fingerprint(rngs))
            atomic_checkpoint(path,state,maximum_bytes,minimum_free_bytes)
            outcome[0]=dict(checkpoint=str(Path(path)),checkpoint_sha256=file_hash(path),
                checkpoint_bytes=Path(path).stat().st_size,steps=completed_steps,world_size=world,format=FORMAT)
        except Exception as e:outcome[0]=dict(error=type(e).__name__+': '+str(e))
    dist.broadcast_object_list(outcome,src=0)
    if 'error' in outcome[0]:raise ValueError('Collective snapshot failed: '+outcome[0]['error'])
    return outcome[0]


def restore_collective_residual(path,model,optimizer,*,expected_checkpoint_sha256,
        base_checkpoint_sha256,tokenizer_sha256,run_plan_sha256,planned_steps,
        sampler_state_for_step,device,allow_completed=False):
    world,rank=_group()
    sources=dict(base_checkpoint_sha256=base_checkpoint_sha256,tokenizer_sha256=tokenizer_sha256,
        run_plan_sha256=run_plan_sha256)
    def preflight():
        guard=FileBindings({str(Path(path)):expected_checkpoint_sha256})
        state=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        parameters=_optimizer_parameters(model,optimizer)
        step=state.get('steps')
        if (state.get('format')!=FORMAT or state.get('world_size')!=world or type(step) is not int
                or not 0<step<=planned_steps or state.get('planned_steps')!=planned_steps
                or (step==planned_steps and not allow_completed)
                or state.get('candidate_promoted') is not False or state.get('goal_achieved') is not False):
            raise ValueError('Exact worker count/partial or explicitly completed budget required')
        meta=_metadata(model,sources,step,planned_steps,sampler_state_for_step(step))
        if fingerprint({k:state.get(k) for k in meta})!=fingerprint(meta):raise ValueError('Own sources/config/partition/progress do not match')
        expected=model.residuals.state_dict();residuals=state.get('residuals')
        if (not isinstance(residuals,dict) or set(residuals)!=set(expected)
                or any(residuals[n].shape!=t.shape or residuals[n].dtype!=t.dtype
                    or not bool(torch.isfinite(residuals[n]).all()) for n,t in expected.items())):
            raise ValueError('Complete finite exact residual tensor inventory required')
        _validate_adam(state.get('optimizer'),parameters,step)
        rngs=state.get('worker_rng_states')
        if not isinstance(rngs,list) or len(rngs)!=world:raise ValueError('Every original rank needs its own complete RNG state')
        for r in rngs:_validate_rng(r,device)
        if (state_digest(residuals)!=state.get('residual_state_sha256')
                or fingerprint(state['optimizer'])!=state.get('optimizer_state_sha256')
                or fingerprint(rngs)!=state.get('worker_rng_states_sha256')
                or state_digest(model.base.state_dict())!=state.get('frozen_backbone_state_sha256')
                or fingerprint(dict(meta=meta,base=state['frozen_backbone_state_sha256'],
                    residuals=residuals,optimizer=state['optimizer']))!=state.get('shared_state_fingerprint')):
            raise ValueError('Every residual/Adam/RNG/backbone byte must match its exact saved state')
        if not isinstance(state.get('metrics'),list) or [m.get('step') for m in state['metrics']]!=list(range(1,step+1)):
            raise ValueError('Complete ordered original metrics required')
        json.dumps(state['metrics'],allow_nan=False);guard.check(full=True)
        return state,fingerprint(dict(meta=meta,shared=state['shared_state_fingerprint'],rngs=rngs,checkpoint_sha256=expected_checkpoint_sha256))
    state,signature=_validated(preflight);_same(signature)
    def apply():
        model.load_adapter_state(state['residuals']);optimizer.load_state_dict(state['optimizer'])
        optimizer.zero_grad(set_to_none=True);restore_rng(state['worker_rng_states'][rank],device)
    _validated(apply)
    # Confirm the shared optimizer and parameters agree after the collective commit.
    _same(fingerprint(dict(residuals=model.residuals.state_dict(),optimizer=optimizer.state_dict())))
    return state['steps'],state['sampler_state'].copy(),state['metrics'].copy()
