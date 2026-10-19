import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]


def small_model():
    from slm_lab.model import ModelConfig
    from slm_lab.own_dense_growth_v1 import OwnDenseAssistant,new_backbone
    from slm_lab.own_dense_actor_v1 import OwnDenseActor
    torch.set_num_threads(1)
    c=ModelConfig(vocab_size=32768,hidden_size=32,intermediate_size=64,num_layers=2,num_heads=4,num_kv_heads=2,
        max_text_length=4096,vision_hidden_size=8,vision_intermediate_size=16,vision_layers=1,vision_heads=2,image_size=16,patch_size=4,pool_size=2)
    base=OwnDenseAssistant(new_backbone(c,957319),reward_hidden=8,word_temporal=4,pointing_rank=8,router_hidden=8,detector_hidden=4)
    model=OwnDenseActor(base,rank=2,seed=957319)
    opt=torch.optim.AdamW([p for _,p in model.named_selected(('actor',))],lr=.001,foreach=False)
    return model,opt


def test_causal_reward_actual_kernel_fixed_horizon_does_not_commit_or_consume_a_source():
    from slm_lab.execution_group_causal_v3 import CausalExecutionGroup
    from slm_lab.execution_group_capacity_v2 import stress
    model,opt=small_model();stage=CausalExecutionGroup(model,opt,ROOT,method='execution_hybrid',group_size=2,max_new_tokens=3)
    report=stress(stage,stage.order[0])
    assert report['complete_global_fixed_actions']==6 and report['no_optimizer_commit']
    assert stage.completed==0 and not opt.state


def test_whole_source_views_extreme_backward_does_not_commit_mutate_Adam_or_consume_source():
    from slm_lab.verified_views_warmup_v1 import VerifiedViewsWarmup
    from slm_lab.verified_views_capacity_v1 import extremes,stress
    model,opt=small_model();stage=VerifiedViewsWarmup(model,opt,ROOT,method='views',epochs=1,functions_per_update=1)
    # Allocate actual moments before the no-commit stress check.
    stage.update();before=stage.completed;report=stress(stage,extremes(stage)[0])
    assert report['actual_complete_presentations']==3 and report['no_optimizer_commit']
    assert report['unchanged_weights_metadata_Adam_private_RNG_source_cursor'] and stage.completed==before


def driver():
    sys.path.insert(0,str(ROOT/'scripts'));spec=importlib.util.spec_from_file_location('verified_execution_driver_test',ROOT/'scripts/train_verified_views_execution_v1.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_live_predecessor_or_worker_prevents_GPU_acquisition(tmp_path,monkeypatch):
    m=driver();m.PREVIOUS=tmp_path;prior=dict(owner=dict(pid=101,ticks='202'))
    monkeypatch.setattr(m.owner.resources,'same',lambda actor:dict(state='R'))
    assert not m.previous_ready(prior)
    (tmp_path/'children.json').write_text(json.dumps([dict(pid=303,ticks='404')]))
    monkeypatch.setattr(m.owner.resources,'same',lambda actor:dict(state='R') if actor['pid']==303 else None)
    assert not m.previous_ready(prior)


def test_failed_or_partial_predecessor_requires_restoration_and_exact_preserved_handover(tmp_path,monkeypatch):
    from slm_lab.corpus import file_hash
    m=driver();m.PREVIOUS=tmp_path;m.HANDOVER=tmp_path/'handover.json';prior=dict(owner=dict(pid=101,ticks='202'))
    monkeypatch.setattr(m.owner.resources,'same',lambda actor:None);(tmp_path/'children.json').write_text('[]')
    with pytest.raises(ValueError,match='restored'):m.previous_ready(prior)
    (tmp_path/'restored.json').write_text('{}')
    with pytest.raises(ValueError,match='finish'):m.previous_ready(prior)
    kept=tmp_path/'retained-state.pt';kept.write_bytes(b'fixture state identity, not a real model')
    record=dict(retired_exact_owner=prior['owner'],partial_epoch_and_endpoints_explicitly_incomplete=True,
        initial_and_all_evaluated_checkpoints_preserved=True,retained_evidence_sha256={str(kept):file_hash(kept)})
    m.HANDOVER.write_text(json.dumps(record));assert m.previous_ready(prior)
    kept.write_bytes(b'changed state')
    with pytest.raises(ValueError,match='evidence changed'):m.previous_ready(prior)
    record['retired_exact_owner']['ticks']='999';m.HANDOVER.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='handover'):m.previous_ready(prior)
