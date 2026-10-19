"""Complete ordinary Adam/RNG recovery for fully plastic own dense language.

Every original dense language matrix/table/norm is optimized; this is not
adapter-only learning. Vision, critic and auxiliary heads remain outside this
particular optimizer and are validated before a language-only restore.
"""
import torch

from .codenet_checkpoint_v1 import capture_rng,restore_rng
from .code_residual_checkpoint_v1 import _cpu,_validate_adam,_validate_rng
from .own_dense_growth_v1 import OwnDenseAssistant

FORMAT='own-full-dense-language-Adam-RNG-source-cursor-v1'


def selected(model,optimizer):
    if type(model) is not OwnDenseAssistant:raise ValueError('Exact complete own dense model required')
    model.assert_inventory();named=model.language_named_parameters();parameters=[p for _,p in named]
    if (type(optimizer) is not torch.optim.AdamW or len(optimizer.param_groups)!=1
            or [id(p) for p in optimizer.param_groups[0]['params']]!=[id(p) for p in parameters]
            or optimizer.param_groups[0].get('foreach') is not False or optimizer.param_groups[0].get('fused') not in (None,False)
            or any(optimizer.param_groups[0].get(k,False) for k in ('amsgrad','maximize','capturable','differentiable'))):
        raise ValueError('One ordinary exact-order full dense-language AdamW group required')
    return named


def make_state(model,optimizer,*,sources,completed,planned,cursor,metrics,device):
    named=selected(model,optimizer)
    if (type(completed) is not int or type(planned) is not int or not 0<completed<=planned
            or not isinstance(metrics,list) or len(metrics)!=completed or not isinstance(cursor,dict) or not cursor):
        raise ValueError('Actual complete update budget, each metric and explicit whole-source cursor required')
    adam=_cpu(optimizer.state_dict());_validate_adam(adam,[p for _,p in named],completed)
    rng=capture_rng(device);_validate_rng(rng,device)
    return dict(format=FORMAT,model=model.export(sources),optimizer=adam,optimized_names=[n for n,_ in named],
        completed=completed,planned=planned,cursor=_cpu(cursor),metrics=_cpu(metrics),rng=rng,world_size=1,
        candidate_promoted=False,goal_achieved=False)


def restore_state(model,optimizer,state,*,expected_sources,planned,expected_cursor,device,allow_completed=False):
    named=selected(model,optimizer);completed=state.get('completed')
    if (state.get('format')!=FORMAT or state.get('world_size')!=1 or state.get('planned')!=planned
            or type(completed) is not int or not 0<completed<=planned or (completed==planned and not allow_completed)
            or state.get('optimized_names')!=[n for n,_ in named] or state.get('cursor')!=expected_cursor or not expected_cursor
            or not isinstance(state.get('metrics'),list) or len(state['metrics'])!=completed
            or state.get('candidate_promoted') is not False or state.get('goal_achieved') is not False):
        raise ValueError('Exact complete own dense language state/source cursor/budget required before mutation')
    model.validate_export(state['model'],expected_sources=expected_sources)
    _validate_adam(state.get('optimizer'),[p for _,p in named],completed);_validate_rng(state.get('rng'),device)
    if state['optimizer']['param_groups']!=optimizer.state_dict()['param_groups']:
        raise ValueError('Exact full-language ordinary AdamW hyperparameters required')
    names={n for n,_ in named};live=model.network.state_dict();saved=state['model']
    if (any(not torch.equal(t.detach().cpu(),saved['network'][n].detach().cpu()) for n,t in live.items() if n not in names and n!='lm_head.weight')
            or any(not torch.equal(t.detach().cpu(),saved['heads'][n].detach().cpu()) for n,t in model.heads.state_dict().items())):
        raise ValueError('All fixed vision/critic/auxiliary tensors must remain equal before recovery')
    # The tied head aliases embedding.weight; never copy unused tensors or
    # increment their versions merely because the export contains them.
    with torch.no_grad():
        for name,p in named:p.copy_(saved['network'][name])
    # CPU load_state_dict may alias the supplied CPU moments. A subsequent
    # update must not mutate the caller's retained recovery payload.
    optimizer.load_state_dict(_cpu(state['optimizer']));optimizer.zero_grad(set_to_none=True);restore_rng(state['rng'],device)
    return completed,_cpu(state['metrics'])
