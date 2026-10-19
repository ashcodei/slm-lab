from collections import Counter, defaultdict

import pytest
import torch
from contextlib import nullcontext

from slm_lab.codenet_replay_schedule_v1 import CoverageReplaySchedule
from slm_lab.codenet_runner_v1 import training_update
from slm_lab.codenet_stage_v1 import EpochOrder


def test_complete_replay_floor_reaches_every_row_with_disclosed_rounding_and_extra_updates():
    populations=(('code',7),('dialogue',31),('image_reading',107))
    schedule=CoverageReplaySchedule(populations,24,8)
    counts=defaultdict(Counter)
    cursors=Counter()
    for step in range(24):
        name,occurrence=schedule.at(step)
        assert occurrence==cursors[name]
        cursors[name]+=1
        sampler=EpochOrder(dict(populations)[name],83)
        for rank in range(4):
            for group in sampler.rank_groups(occurrence,rank,4,1,2):
                counts[name][group[0]]+=1
    assert sum(sum(values.values()) for values in counts.values())==24*8
    for name,size in populations:
        assert set(counts[name])==set(range(size))
    assert tuple(cursors[name] for name,size in populations)==schedule.quotas
    assert schedule.minimum_required_updates==19
    assert schedule.entries[9:]==CoverageReplaySchedule(populations,24,8).entries[9:]


def test_insufficient_capacity_or_invalid_populations_cannot_silently_skip_replay_sources():
    with pytest.raises(ValueError,match='cannot reach every'):
        CoverageReplaySchedule((('large',101),('small',2)),10,8)
    for populations in [(),(('x',0),),(('x',3),('x',4)),(('x',True),)]:
        with pytest.raises(ValueError):CoverageReplaySchedule(populations,10,8)


def test_exact_quotas_and_absolute_family_cursors_hold_for_unequal_populations():
    for size in range(1,30):
        populations=(('first',size),('second',2*size+1),('third',size+7))
        schedule=CoverageReplaySchedule(populations,50,4)
        seen=defaultdict(list)
        for step in range(50):
            name,position=schedule.at(step);seen[name].append(position)
        for (name,size),quota in zip(populations,schedule.quotas):
            assert seen[name]==list(range(quota)) and quota*4>=size


def test_runner_rejects_insufficient_replay_before_reading_rows_or_updating_weights():
    class Code:
        def __len__(self):return 5
    class Replay:
        families=['too_large']
        def size(self,name):return 1000
        def row(self,*args):raise AssertionError('No source row should be consumed')
    model=torch.nn.Linear(1,1)
    before={n:p.detach().clone() for n,p in model.named_parameters()}
    optimizer=torch.optim.AdamW(model.parameters())
    options={'seed':1,'passes':1,'batch_size':1,'gradient_accumulation':1,
             'replay_batch_size':1,'replay_accumulation':1}
    with pytest.raises(ValueError,match='cannot reach every'):
        training_update(model,optimizer,Code(),None,Replay(),None,0,options,'cpu',nullcontext,0,1)
    assert not optimizer.state
    for name,p in model.named_parameters():torch.testing.assert_close(p.detach(),before[name],rtol=0,atol=0)
