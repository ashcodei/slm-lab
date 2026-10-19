"""Actual longest complete view/function-group backward, no optimizer commit."""
import copy

from .codenet_checkpoint_v1 import capture_rng
from .execution_group_capacity_v2 import GradientBoundaryReached,equal
from .verified_views_warmup_v1 import VerifiedViewsWarmup
from .whole_human_sft_evidence_v1 import tensor_digest


def extremes(stage):
    rows=stage.pack.index[stage.pack.positions]
    positions=[max(range(len(stage.rows)),key=lambda p:int(rows[p][field])) for field in ['length']]
    positions.append(max(range(len(stage.rows)),key=lambda p:int(rows[p]['length']-rows[p]['target_start'])))
    return list(dict.fromkeys(stage.rows[p]['original_source_position'] for p in positions))


def stress(stage,position):
    if type(stage) is not VerifiedViewsWarmup or position not in stage.by_function or stage.completed>=stage.planned:
        raise ValueError('Exact real view learner and bounded complete source group required')
    before=dict(weights=tensor_digest(stage.model.named_parameters()),metadata=stage.signature(stage.model.parameters()),
        optimizer=copy.deepcopy(stage.optimizer.state_dict()),RNG=capture_rng(next(stage.model.parameters()).device),
        orders=copy.deepcopy(stage.orders),completed=stage.completed,metrics=copy.deepcopy(stage.metrics))
    original_step=stage.optimizer.step;boundary={}
    def no_commit(*args,**kwargs):
        boundary['actual_actor_gradient_tensors']=len(stage.parameters)
        raise GradientBoundaryReached('Actual complete-source gradients reduced and clipped; optimizer commit prohibited')
    try:
        epoch,within=divmod(stage.completed,stage.steps_per_epoch)
        a=within*stage.width;b=min(a+stage.width,len(stage.orders[epoch]));stage.orders[epoch][a:b]=[position]*(b-a)
        functions=stage.positions(stage.completed);views=stage.views(functions)
        stage.optimizer.step=no_commit
        try:stage.update()
        except GradientBoundaryReached:pass
        else:raise ValueError('Actual extreme must stop at its first optimizer boundary')
    finally:stage.optimizer.step=original_step;stage.orders=before['orders']
    if (not boundary or before['weights']!=tensor_digest(stage.model.named_parameters())
            or before['metadata']!=stage.signature(stage.model.parameters()) or not equal(before['optimizer'],stage.optimizer.state_dict())
            or not equal(before['RNG'],capture_rng(next(stage.model.parameters()).device))
            or stage.completed!=before['completed'] or stage.metrics!=before['metrics']):
        raise ValueError('Actual complete extreme must leave every model/Adam/RNG/source state unchanged')
    return dict(original_function_position=position,complete_view_positions=views,actual_complete_presentations=len(views),
        repeated_source_extreme_is_capacity_only_not_new_training_data=True,no_optimizer_commit=True,
        unchanged_weights_metadata_Adam_private_RNG_source_cursor=True,**boundary,production_updates=0,goal_achieved=False)
