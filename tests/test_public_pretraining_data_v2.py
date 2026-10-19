from collections import Counter
from types import SimpleNamespace

import pytest

from slm_lab.codenet_token_pack_v1 import CodePackWriter,CompleteCodePack
from slm_lab.public_pretraining_data_v1 import RAW,TokenBudgetSchedule
from slm_lab.public_pretraining_data_v2 import FUNCTIONS,attach_function_metadata,function_batch


def pack(tmp_path):
    for name in ['tokenizer.json','model-config.json','DATASET-LICENSE.txt','publisher-README.md']:
        (tmp_path/name).write_text('unit metadata only')
    writer=CodePackWriter(tmp_path,minimum_free_gib=0)
    for i,role in enumerate(['train','validation','train']):
        ids=[1,4,10+i,20+i,2]
        writer.add(dict(id=str(i),split=role,prompt='unit contract',response='unit source'),ids,[-100,-100,*ids[2:]])
    writer.finish(dict(unit_fixture_not_real_data_admission=True))
    return CompleteCodePack(tmp_path,'train','audit')


def data():
    return SimpleNamespace(role='train',purpose='optimization',retention_identity_sha256='a'*64,
        retention_entries=[('previous',[0,1,0])],populations={RAW:2,'previous':2},
        lengths={RAW:[4096,4096],'previous':[5,5]},targets={RAW:[4095,4095],'previous':[3,3]},
        kinds={RAW:'text','previous':'text'})


def test_function_train_positions_native_EOS_and_previous_repeated_exposures_survive_rebatching(tmp_path):
    source=pack(tmp_path);d=data();attach_function_metadata(d,source)
    assert d.populations[FUNCTIONS]==2
    batch=function_batch(d,[0,1])
    assert int((batch['labels'][:,1:]!=-100).sum())==6
    assert 11 not in batch['input_ids'].flatten().tolist() # held fixture row never enters train
    schedule=TokenBudgetSchedule(d,context_budget=8192,maximum_rows=8)
    seen=Counter((n,p) for n,positions in schedule.entries for p in positions)
    assert seen['previous',0]==2 and seen[FUNCTIONS,0]==seen[FUNCTIONS,1]==1
    assert schedule.selected_budget(d)[FUNCTIONS]['response_EOS_targets']==6


def test_held_role_duplicate_family_and_invalid_function_positions_are_rejected(tmp_path):
    source=pack(tmp_path);d=data()
    held=CompleteCodePack(tmp_path,'validation','audit')
    with pytest.raises(ValueError,match='source-train'):attach_function_metadata(d,held)
    attach_function_metadata(d,source)
    with pytest.raises(ValueError,match='single sealed'):attach_function_metadata(d,source)
    with pytest.raises(ValueError,match='source-train function positions'):function_batch(d,[2])
