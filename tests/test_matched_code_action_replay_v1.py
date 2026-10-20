import copy

import pytest
import torch

from slm_lab.code_tasks import TASKS
from slm_lab.matched_code_action_replay_v1 import ActionPrefixReplay


def record(index=0):
    return dict(endpoint='initial',frame='completion',case=index+1,language='Python',task=TASKS[index][0],
        complete_prompt_ids=[1,4,5],actual_response_ids=[7,2],seconds=12.5,stopped_on_EOS=True,ended_on_budget=False)


def test_retained_complete_ids_skip_generator_and_missing_suffix_invokes_it_once():
    original=record();replay=ActionPrefixReplay([original]);original['actual_response_ids'][0]=99
    ids=torch.tensor([[1,4,5]])
    def forbidden():pytest.fail('Retained actions must not be newly generated')
    output,meta=replay.deliver(ids,replay.layout[0],forbidden)
    assert output.tolist()==[[1,4,5,7,2]] and not meta['new_neural_generation']
    assert meta['original_neural_generation_seconds']==12.5
    calls=[]
    def fresh():calls.append(True);return torch.tensor([[1,4,5,8,2]])
    output,meta=replay.deliver(ids,replay.layout[1],fresh)
    assert output.tolist()==[[1,4,5,8,2]] and meta['new_neural_generation']
    assert len(calls)==1 and replay.retained_uses==replay.fresh_uses==1 and replay.delivered==2


def test_reordered_sources_changed_prompts_bad_EOS_budget_and_incomplete_generation_reject_without_advancing():
    for field,value in [('case',2),('task',TASKS[1][0]),('stopped_on_EOS',False),('ended_on_budget',True),('actual_response_ids',[7]),('actual_response_ids',[2,7,2])]:
        bad=record();bad[field]=value
        with pytest.raises(ValueError):ActionPrefixReplay([bad])
    replay=ActionPrefixReplay([record()]);before=copy.deepcopy(replay.__dict__)
    with pytest.raises(ValueError,match='prompt IDs'):replay.deliver(torch.tensor([[1,4,6]]),replay.layout[0],lambda:None)
    assert replay.__dict__==before
    ids=torch.tensor([[1,4,5]]);replay.deliver(ids,replay.layout[0],lambda:None)
    before=replay.delivered
    with pytest.raises(ValueError):replay.deliver(ids,replay.layout[1],lambda:torch.tensor([[1,4,5,7]]))
    assert replay.delivered==before and replay.fresh_uses==0
