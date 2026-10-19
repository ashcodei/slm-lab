"""Private Adam/RNG recovery for explicit dense policy/critic groups only."""
import torch

from .codenet_checkpoint_v1 import capture_rng,restore_rng
from .code_residual_checkpoint_v1 import _cpu,_validate_adam,_validate_rng
from .own_dense_actor_v1 import OwnDenseActor

FORMAT='own-dense-policy-critic-selected-Adam-RNG-source-cursor-v1'


def selected(model,optimizer,groups):
    if type(model) is not OwnDenseActor:raise ValueError('Exact complete own dense policy graph required')
    model.assert_shared();named=model.named_selected(groups)
    if (type(optimizer) is not torch.optim.AdamW or len(optimizer.param_groups)!=1
            or [id(p) for p in optimizer.param_groups[0]['params']]!=[id(p) for _,p in named]
            or optimizer.param_groups[0].get('foreach') is not False or optimizer.param_groups[0].get('fused') not in (None,False)
            or any(optimizer.param_groups[0].get(k,False) for k in ('amsgrad','maximize','capturable','differentiable'))):
        raise ValueError('One ordinary exact-order explicitly selected policy/critic AdamW group required')
    return named


def make_state(model,optimizer,*,groups,sources,completed,planned,cursor,metrics,device):
    named=selected(model,optimizer,groups)
    if (type(completed) is not int or type(planned) is not int or not 0<completed<=planned
            or not isinstance(cursor,dict) or not cursor or not isinstance(metrics,list) or len(metrics)!=completed):
        raise ValueError('Exact actual source cursor/update budget and every metric required')
    adam=_cpu(optimizer.state_dict());_validate_adam(adam,[p for _,p in named],completed)
    rng=capture_rng(device);_validate_rng(rng,device)
    return dict(format=FORMAT,model=model.export(sources),optimizer=adam,groups=list(groups),optimized_names=[n for n,_ in named],
        completed=completed,planned=planned,cursor=_cpu(cursor),metrics=_cpu(metrics),rng=rng,world_size=1,
        candidate_promoted=False,goal_achieved=False)


def restore_state(model,optimizer,state,*,groups,expected_sources,planned,expected_cursor,device,allow_completed=False):
    named=selected(model,optimizer,groups);completed=state.get('completed')
    if (state.get('format')!=FORMAT or state.get('world_size')!=1 or state.get('groups')!=list(groups)
            or state.get('optimized_names')!=[n for n,_ in named] or state.get('planned')!=planned
            or type(completed) is not int or not 0<completed<=planned or (completed==planned and not allow_completed)
            or state.get('cursor')!=expected_cursor or not expected_cursor
            or not isinstance(state.get('metrics'),list) or len(state['metrics'])!=completed
            or state.get('candidate_promoted') is not False or state.get('goal_achieved') is not False):
        raise ValueError('Exact dense policy groups/order/source cursor and complete budget required before mutation')
    model.validate_export(state['model'],expected_sources=expected_sources)
    _validate_adam(state.get('optimizer'),[p for _,p in named],completed);_validate_rng(state.get('rng'),device)
    if state['optimizer']['param_groups']!=optimizer.state_dict()['param_groups']:
        raise ValueError('Original fixed selected AdamW settings required')
    names={n for n,_ in named};live=model.parts.state_dict()
    if any(not torch.equal(t.detach().cpu(),state['model']['parts'][n]) for n,t in live.items() if n not in names):
        raise ValueError('Every unused policy/critic tensor must remain exact before recovery')
    with torch.no_grad():
        for name,p in named:p.copy_(state['model']['parts'][name])
    optimizer.load_state_dict(_cpu(state['optimizer']));optimizer.zero_grad(set_to_none=True);restore_rng(state['rng'],device)
    return completed,_cpu(state['metrics'])
