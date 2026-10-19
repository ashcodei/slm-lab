import copy

import pytest

from slm_lab.apps_epoch_evidence_v1 import exposures


def fixture():
    # Hand-retained draw order: code population5, seed53; rehearsal population3,
    # seeds1053/1054. The first rehearsal crosses two permutation boundaries.
    codes=[[0,2,3,1],[4,0,4,2]]
    replay=[[0,2,1,1,2,0,1,2],[2,3,5,1,6,4,0,5]]
    rows=[dict(target_tokens=i+2) for i in range(5)]
    metrics=[]
    for step,(code,old) in enumerate(zip(codes,replay)):
        name=['first','original_packed_code'][step]
        n=sum(rows[i]['target_tokens'] for i in code);nr=sum(i+1 for i in old)
        objectives={'complete_original_human_program':dict(global_target_tokens=n,coefficient=1.,token_mean_nll=2.,weighted_objective=2.),
            name:dict(global_target_tokens=nr,coefficient=.5,token_mean_nll=3.,weighted_objective=1.5)}
        metrics.append(dict(step=step+1,replay_family=name,replay_occurrence=0,
            replay_response_field='response' if step==0 else None,objectives=objectives,
            summed_objective=3.5,total_target_tokens=n+nr,worker_source_positions=[
                dict(code_positions=[code[r]],replay_positions=[old[r],old[4+r]]) for r in range(4)]))
    return rows,metrics


def check(rows,metrics,**kwargs):
    return exposures(metrics,rows,[('first',3),('original_packed_code',7)],seed=53,planned_steps=2,
        world=4,code_accumulation=1,replay_accumulation=2,complete=True,
        replay_target_count=lambda name,i,field:i+1,**kwargs)


def test_complete_original_and_rehearsal_orders_cross_epoch_boundaries():
    rows,metrics=fixture()
    result=check(rows,metrics)
    assert result['original_program_exposures']==8 and result['distinct_original_program_positions']==5
    assert result['rounding_or_prefix_repeats']==3 and result['full_original_program_pass']
    assert result['per_rehearsal_family']['first']['logical_record_exposures']==8
    assert result['per_rehearsal_family']['first']['distinct_logical_positions']==3


def test_wrong_replay_cursor_position_field_or_target_is_rejected():
    rows,original=fixture()
    for alter in [lambda m:m[0]['worker_source_positions'][0]['replay_positions'].__setitem__(0,1),
            lambda m:m[1].__setitem__('replay_response_field','response'),
            lambda m:m[1]['objectives']['original_packed_code'].__setitem__('global_target_tokens',999),
            lambda m:m[1]['objectives']['original_packed_code'].__setitem__('coefficient',1.)]:
        metrics=copy.deepcopy(original);alter(metrics)
        with pytest.raises(ValueError):check(rows,metrics)
    with pytest.raises(ValueError,match='cursor/population'):
        check(rows,original,sampler_state={'updates':2})
