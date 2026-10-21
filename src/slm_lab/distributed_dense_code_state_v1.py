"""One common dense/Adam checkpoint retaining every replica's private RNG."""
from .codenet_checkpoint_v1 import capture_rng, restore_rng
from .code_residual_checkpoint_v1 import _validate_rng, _validate_adam
from .distributed_dense_code_step_v1 import gather, agree
from .own_dense_checkpoint_v1 import make_state, restore_state, selected, FORMAT as COMMON_FORMAT
import torch
from torch import distributed as dist

FORMAT = 'common-full-dense-Adam-four-private-RNG-v1'


def make(model, optimizer, *, sources, completed, planned, metrics, device, object_group=None):
    rngs = gather(capture_rng(device), object_group)
    state = None
    error = None
    if dist.get_rank() == 0:
        try:
            common = make_state(model, optimizer, sources=sources, completed=completed, planned=planned,
                cursor={'next_global_batch': completed}, metrics=metrics, device=device)
            state = dict(format=FORMAT, world_size=dist.get_world_size(), common=common, private_rngs=rngs)
        except Exception as exc:
            error = repr(exc)
    agree(error, object_group)
    return state


def restore(model, optimizer, state, *, sources, planned, device, object_group=None):
    error = None
    try:
        if (state.get('format') != FORMAT or state.get('world_size') != dist.get_world_size()
                or len(state.get('private_rngs', [])) != dist.get_world_size()):
            raise ValueError('Exact original replica/RNG population is required')
        for rng in state['private_rngs']:
            _validate_rng(rng, device)
        completed = state['common']['completed']
        common = state['common']
        named = selected(model, optimizer)
        if (common.get('format') != COMMON_FORMAT or common.get('world_size') != 1
                or common.get('planned') != planned or type(completed) is not int or not 0 < completed <= planned
                or common.get('optimized_names') != [n for n, _ in named]
                or common.get('cursor') != {'next_global_batch': completed}
                or len(common.get('metrics', [])) != completed
                or common.get('candidate_promoted') is not False or common.get('goal_achieved') is not False):
            raise ValueError('Every common source cursor, optimizer scope and update must match')
        model.validate_export(common['model'], expected_sources=sources)
        _validate_adam(common['optimizer'], [p for _, p in named], completed)
        _validate_rng(common['rng'], device)
        if common['optimizer']['param_groups'] != optimizer.state_dict()['param_groups']:
            raise ValueError('Exact original ordinary Adam settings are required')
        names = {n for n, _ in named}
        saved = common['model']
        if (any(not torch.equal(t.detach().cpu(), saved['network'][n].detach().cpu())
                for n, t in model.network.state_dict().items() if n not in names and n != 'lm_head.weight')
                or any(not torch.equal(t.detach().cpu(), saved['heads'][n].detach().cpu())
                for n, t in model.heads.state_dict().items())):
            raise ValueError('All original fixed values must match before any rank restores')
    except Exception as exc:
        error = repr(exc)
    agree(error, object_group)
    completed, rows = restore_state(model, optimizer, state['common'], expected_sources=sources,
        planned=planned, expected_cursor={'next_global_batch': completed}, device=device, allow_completed=True)
    restore_rng(state['private_rngs'][dist.get_rank()], device)
    return completed, rows
