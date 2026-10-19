from copy import deepcopy
import json

import pytest
import torch

from slm_lab.corpus import file_hash
from slm_lab.deterministic_positions_v1 import install
from slm_lab.held_human_shards_v1 import evaluate_shard as actor_shard,reconcile as actor_merge
from slm_lab.own_actor_dense_fold_v1 import fold
from slm_lab.standalone_held_human_v1 import CachedHeldReference,evaluate_shard,reconcile
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
from test_dense_human_preference_evaluation_v1 import held
from test_held_human_shards_v1 import profile
from test_machine_code_preference_stage_v1 import trained


def fixture(tmp_path):
    m=trained();install(m.base);d=held();source=tmp_path/'actor.json'
    shard=tmp_path/'actor-shard.json';actor_shard(m,d,shard,rank=0,world=1,provenance=profile())
    old=actor_merge(d,[shard],source);dense,_=fold(m)
    cache=CachedHeldReference(d,source,expected_sha256=file_hash(source),expected_profile=old['profile'])
    return dense,d,cache,old


def test_actual_folded_student_only_forwards_full_masks_and_cached_reference_match_actor(tmp_path,monkeypatch):
    model,data,cache,old=fixture(tmp_path);original=model.forward;calls=[]
    def forward(*a,**kw):calls.append(kw['input_ids'].shape);return original(*a,**kw)
    monkeypatch.setattr(model,'forward',forward);initial=tensor_digest(model.named_parameters())
    paths=[tmp_path/f'dense-{r}.json' for r in range(2)]
    for rank,path in enumerate(paths):evaluate_shard(model,data,path,cache,rank=rank,world=2,provenance=profile())
    result=reconcile(data,paths,tmp_path/'merged.json',cache)
    assert len(calls)==4 and result['actual_student_values']==model.counts()['total_unique']
    assert result['student_only_neural_inference'] and result['reference_scores_reused_not_new_reference_forwards']
    for row,before in zip(result['results'],old['results']):
        for field in ['chosen','rejected']:
            assert row['reference_'+field+'_logp']==before['reference_'+field+'_logp']
            assert row['policy_'+field+'_logp']==pytest.approx(before['policy_'+field+'_logp'],abs=5e-6,rel=3e-6)
    assert initial==tensor_digest(model.named_parameters()) and all(p.grad is None for p in model.parameters())


def test_wrong_reference_profile_or_held_native_masks_reject_before_any_deliveries(tmp_path):
    model,data,cache,old=fixture(tmp_path);wrong=deepcopy(old['profile']);wrong['frozen_dense_sha256']='0'*64
    with pytest.raises(ValueError,match='Exact completed'):CachedHeldReference(data,cache.path,expected_sha256=cache.sha256,expected_profile=wrong)
    data.role='train';path=tmp_path/'invalid.json'
    with pytest.raises(ValueError,match='evaluation-only'):evaluate_shard(model,data,path,cache,rank=0,world=1,provenance=profile())
    assert not path.exists() and not path.with_suffix('.jsonl').exists();data.role='valid'
    data.rows[0]['native']['chosen']['target_start']+=1
    with pytest.raises(ValueError):cache.values(0)


def test_duplicate_shard_and_fabricated_cached_reference_value_cannot_reconcile(tmp_path):
    model,data,cache,old=fixture(tmp_path);paths=[tmp_path/f'dense-{r}.json' for r in range(2)]
    for rank,path in enumerate(paths):evaluate_shard(model,data,path,cache,rank=rank,world=2,provenance=profile())
    with pytest.raises(ValueError):reconcile(data,[paths[0],paths[0]],tmp_path/'duplicate.json',cache)
    raw=json.loads(paths[1].read_text());raw['results'][0]['reference_chosen_logp']-=.25
    journal=paths[1].with_suffix('.jsonl');journal.write_text(''.join(json.dumps(r)+'\n' for r in raw['results']))
    raw['raw_journal_sha256']=file_hash(journal);paths[1].write_text(json.dumps(raw))
    with pytest.raises(ValueError,match='cached reference'):reconcile(data,paths,tmp_path/'fabricated.json',cache)


def test_original_reference_bytes_changed_after_loading_reject_before_student_forward(tmp_path,monkeypatch):
    model,data,cache,old=fixture(tmp_path);cache.path.write_text(cache.path.read_text()+' ')
    def forbidden(*args,**kwargs):raise AssertionError('Changed reference evidence must reject before a student forward')
    monkeypatch.setattr(model,'forward',forbidden);path=tmp_path/'changed.json'
    with pytest.raises(ValueError,match='changed'):evaluate_shard(model,data,path,cache,rank=0,world=1,provenance=profile())
    assert not path.exists() and not path.with_suffix('.jsonl').exists()
