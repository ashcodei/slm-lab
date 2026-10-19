from copy import deepcopy
import json
from pathlib import Path

import pytest

from slm_lab.corpus import file_hash
from slm_lab.dense_word_evaluation_v1 import evaluate
from slm_lab.shared_word_data_v1 import SharedWordData
from slm_lab.shared_word_evaluation_v1 import wrong_permutation
from slm_lab.standalone_word_shards_v1 import positions,evaluate_shard,reconcile
from test_dense_word_evaluation_v1 import Tokenizer
from test_shared_word_evaluation_v1 import data
from test_own_dense_growth_v1 import model as make_model
from test_own_aligned_assistant_v1 import original_signature


def test_actual_dense_four_shards_preserve_full_word_population_global_donors_and_both_neural_paths(tmp_path):
    model,_=make_model();held=data();tokenizer=Tokenizer();before=original_signature(model)
    serial=evaluate(model,held,tokenizer,tmp_path/'serial.json',batch_size=2,generation_budget=3)
    paths=[tmp_path/f'word-{r}.json' for r in range(4)]
    for rank,path in enumerate(paths):evaluate_shard(model,held,tokenizer,path,rank=rank,world=4,batch_size=2,generation_budget=3)
    merged=reconcile(held,tokenizer,paths,tmp_path/'merged.json',generation_budget=3,ctc_frames=40)
    assert len(merged['results'])==len(held) and before==original_signature(model)
    for a,b in zip(serial['results'],merged['results']):
        assert {k:v for k,v in a.items() if k!='conditions'}=={k:v for k,v in b.items() if k!='conditions'}
        for condition in ['proper','blank','wrong']:
            assert a['conditions'][condition]['native']['generated_ids']==b['conditions'][condition]['native']['generated_ids']
            assert a['conditions'][condition]['native']['target_ids']==b['conditions'][condition]['native']['target_ids']
            assert a['conditions'][condition]['CTC']['path']==b['conditions'][condition]['CTC']['path']
            assert a['conditions'][condition]['native']['per_target_NLL']==pytest.approx(b['conditions'][condition]['native']['per_target_NLL'],abs=3e-6,rel=3e-6)


def test_train_roles_duplicate_shards_and_changed_original_wrong_donor_reject(tmp_path):
    model,_=make_model();held=data();tokenizer=Tokenizer();held.role='train'
    with pytest.raises(ValueError,match='held'):evaluate_shard(model,held,tokenizer,tmp_path/'bad.json',rank=0,world=4)
    held.role='valid';paths=[tmp_path/f'word-{r}.json' for r in range(4)]
    for rank,path in enumerate(paths):evaluate_shard(model,held,tokenizer,path,rank=rank,world=4,batch_size=2,generation_budget=3)
    with pytest.raises(ValueError):reconcile(held,tokenizer,[paths[0]]*4,tmp_path/'duplicate.json',generation_budget=3,ctc_frames=40)
    raw=json.loads(paths[3].read_text());raw['results'][0]['wrong_image_id']=held.rows[3]['id']
    journal=paths[3].with_suffix('.jsonl');journal.write_text(''.join(json.dumps(r)+'\n' for r in raw['results']))
    raw['raw_journal_sha256']=file_hash(journal);paths[3].write_text(json.dumps(raw))
    with pytest.raises(ValueError,match='wrong-image mapping'):reconcile(held,tokenizer,paths,tmp_path/'changed.json',generation_budget=3,ctc_frames=40)


def test_all3429actual_original_held_crops_and_native_targets_are_preserved_once_across_four_replicas():
    root=Path(__file__).resolve().parents[1];held=SharedWordData(root/'data/shared-byte-words-v1','valid','evaluation')
    shards=[positions(held,rank,4) for rank in range(4)]
    assert len(held)==3429 and [len(row) for row in shards]==[858,857,857,857]
    assert sorted(p for row in shards for p in row)==list(range(3429))
    assert sum(r['native']['length'] for r in held.rows)==68147
    assert sum(r['native']['length']-r['native']['target_start'] for r in held.rows)==16712
    wrong=wrong_permutation(held.rows);assert sorted(wrong)==list(range(3429))
    for i,j in enumerate(wrong):
        a,b=held.rows[i],held.rows[j]
        assert a['byte_family']==b['byte_family'] and a['group_id']!=b['group_id'] and a['response']!=b['response']
