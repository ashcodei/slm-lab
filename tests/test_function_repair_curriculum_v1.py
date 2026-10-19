from copy import deepcopy
from collections import Counter

import pytest

from slm_lab.function_repair_curriculum_v1 import target_pair,schedule,collate


def rows():
    original=dict(id='unit-source',split='train',group_id='unit-file',response='complete original code',source_candidate=dict(role='train'))
    repair=dict(source_function_id='unit-source',split='train',group_id='unit-file',response='complete original code',complete_original_function_target_unchanged=True)
    return original,repair


def test_two_distinct_prompts_retain_identical_complete_original_targets_and_padding_masks():
    original,repair=rows();a=([1,4,5,6,2],[-100,-100,5,6,2]);b=([1,7,8,9,5,6,2],[-100]*4+[5,6,2])
    assert target_pair(original,repair,a,b)==(5,7,3)
    batch=collate([a,b]);assert batch['attention_mask'].sum().item()==12
    assert (batch['labels'][:,1:]!=-100).sum().item()==6 and batch['labels'][0,-2:].tolist()==[-100,-100]
    bad=(b[0],[-100]*4+[5,8,2])
    with pytest.raises(ValueError):target_pair(original,repair,a,bad)


def test_whole_source_group_and_validation_roles_cannot_be_substituted():
    original,repair=rows();native=([1,4,5,2],[-100,-100,5,2])
    for field,value in [('split','validation'),('group_id','other-group'),('response','different target')]:
        bad=deepcopy(repair);bad[field]=value
        with pytest.raises(ValueError):target_pair(original,bad,native,native)


def test_complete_epochs_share_one_arm_independent_schedule_and_keep_short_last_batch():
    lengths=[3,9,4096,30,600,8,1024]
    batches=schedule(lengths,epochs=3,width=4,seed=11)
    assert batches==schedule(lengths,epochs=3,width=4,seed=11)
    assert Counter(p for batch in batches for p in batch)==Counter({i:3 for i in range(len(lengths))})
    assert sum(len(batch)==3 for batch in batches)==3 and len(batches)==6
