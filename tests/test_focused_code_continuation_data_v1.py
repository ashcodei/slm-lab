from types import SimpleNamespace

import pytest
import torch

from slm_lab.focused_code_continuation_data_v1 import (CODE,EXECUTION,REPLAY,FocusedContinuationSchedule,collate_complete)


def test_padding_retains_every_original_output_EOS_and_never_supervises_prefix_or_padding():
    rows=[([1,4,5,6,2],[-100,-100,-100,6,2]),([1,7,8,2],[-100,-100,8,2])]
    batch=collate_complete(rows)
    assert batch['input_ids'].tolist()==[[1,4,5,6,2],[1,7,8,2,0]]
    assert batch['labels'].tolist()==[[-100,-100,-100,6,2],[-100,-100,8,2,-100]]
    assert batch['attention_mask'].sum()==9 and (batch['labels'][:,1:]!=-100).sum()==4
    with pytest.raises(ValueError):collate_complete([([1]+[4]*4095+[2],[-100]+[4]*4095+[2])])
    with pytest.raises(ValueError):collate_complete([([1,4,5,2],[-100,4,-100,2])])


def test_all_new_code_and_execution_sources_once_tail_ranks_preserved_and_replay_explicitly_separate():
    lengths={CODE:[5]*19,EXECUTION:[7]*6,**{name:[3+i%4 for i in range(23)] for name in REPLAY}}
    data=SimpleNamespace(purpose='optimization',lengths=lengths,targets={k:[1]*len(v) for k,v in lengths.items()})
    schedule=FocusedContinuationSchedule(data,seed=11,width=4)
    code=[p for name,positions in schedule.entries if name==CODE for p in positions]
    execution=[p for name,positions in schedule.entries if name==EXECUTION for p in positions]
    assert sorted(code)==list(range(19)) and sorted(execution)==list(range(6))
    assert len([1 for name,_ in schedule.entries if name in REPLAY])==5
    for name,positions in schedule.entries:
        shards=[positions[rank::4] for rank in range(4)]
        assert sorted(p for shard in shards for p in shard)==sorted(positions)
        assert sum(len(shard) for shard in shards)==len(positions)
    assert schedule.identity_sha256==FocusedContinuationSchedule(data,seed=11,width=4).identity_sha256
    assert schedule.identity_sha256!=FocusedContinuationSchedule(data,seed=12,width=4).identity_sha256
    assert schedule.selected_budget(data)[CODE]['presentations']==19


def test_production_cannot_silently_omit_auxiliary_task_and_preparation_is_explicit():
    lengths={CODE:[5]*19,**{name:[5]*23 for name in REPLAY}}
    data=SimpleNamespace(purpose='optimization',lengths=lengths,targets={k:[1]*len(v) for k,v in lengths.items()})
    with pytest.raises(ValueError):FocusedContinuationSchedule(data,width=4)
    data.purpose='source_preparation';schedule=FocusedContinuationSchedule(data,width=4)
    assert all(name!=EXECUTION for name,_ in schedule.entries)
    assert sum(len(p) for name,p in schedule.entries if name==CODE)==19


def test_length_buckets_reduce_real_padding_without_changing_complete_context_or_target_population():
    import random
    values=[8,4000]*256
    lengths={CODE:values,EXECUTION:[7]*6,**{name:[5]*23 for name in REPLAY}}
    data=SimpleNamespace(purpose='optimization',lengths=lengths,targets={k:[1]*len(v) for k,v in lengths.items()})
    schedule=FocusedContinuationSchedule(data,seed=11,width=4)
    actual=[positions for name,positions in schedule.entries if name==CODE]
    order=list(range(len(values)));random.Random(11).shuffle(order)
    naive=[order[i:i+4] for i in range(0,len(order),4)]
    padding=lambda groups:sum(max(values[p] for p in positions)*len(positions) for positions in groups)
    assert padding(actual)<padding(naive)
    assert sorted(p for positions in actual for p in positions)==list(range(len(values)))
    assert sum(values[p] for positions in actual for p in positions)==sum(values)
