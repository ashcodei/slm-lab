from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from slm_lab.public_pretraining_run_contract_v1 import stage_ready,scratch_memory_safe


def test_exact_predecessor_and_all_registered_gpu_children_must_end(tmp_path):
    identity=dict(pid=123,ticks='456')
    same=lambda p:p.get('live',False)
    assert not stage_ready(tmp_path,dict(live=True),same)
    with pytest.raises(ValueError,match='must complete'):stage_ready(tmp_path,identity,same)
    (tmp_path/'plan.json').write_text(json.dumps(dict(owner=identity)))
    (tmp_path/'complete.json').write_text('{}')
    (tmp_path/'children.json').write_text(json.dumps([dict(live=True)]))
    assert not stage_ready(tmp_path,identity,same,gpu_stage=True)
    (tmp_path/'children.json').write_text('[]')
    with pytest.raises(ValueError,match='restore'):stage_ready(tmp_path,identity,same,gpu_stage=True)
    (tmp_path/'restored.json').write_text('{}')
    assert stage_ready(tmp_path,identity,same,gpu_stage=True)
    (tmp_path/'failure.json').write_text('{}')
    with pytest.raises(ValueError,match='partial'):stage_ready(tmp_path,identity,same,gpu_stage=True)


def test_tmpfs_is_charged_to_actual_ram_and_disk_file_cache_is_reclaimable():
    gib=1024**3
    stat=dict(anon=30*gib,shmem=1*gib,kernel=2*gib,file=220*gib)
    assert scratch_memory_safe(stat,256*gib)['resident_nonreclaimable_bytes']==33*gib
    stat['shmem']=130*gib
    with pytest.raises(ValueError,match='RAM limit'):scratch_memory_safe(stat,256*gib)


def test_full_schedule_positions_and_all_nine_capacity_modalities_are_bound():
    from slm_lab.public_pretraining_run_contract_v1 import CAPACITY_FAMILIES,schedule_for
    from slm_lab.public_pretraining_data_v2 import TokenBudgetSchedule,RAW
    names=list(CAPACITY_FAMILIES)
    d=SimpleNamespace(role='train',purpose='optimization',retention_identity_sha256='a'*64,
        lengths={n:[8,9] for n in names},targets={n:[5,6] for n in names},
        kinds={n:('word' if n=='words' else 'image' if n.startswith('images:') else 'text') for n in names},
        retention_entries=[(n,[0,1]) for n in names if n!=RAW])
    d.budget=lambda:{n:dict(examples=2,complete_context_tokens=17,response_EOS_targets=11,kind=d.kinds[n]) for n in names}
    s=TokenBudgetSchedule(d,context_budget=128,maximum_rows=4,image_prefix=4,seed=4)
    plan=dict(actual_schedule_settings=s.settings,source_schedule_sha256=s.identity_sha256,planned_steps=len(s),
        selected_source_budget=s.selected_budget(d),complete_source_budget=d.budget(),
        capacity_global_batches=[[n,next(p for family,p in s.entries if family==n)] for n in names])
    assert schedule_for(d,plan,'train').entries==s.entries
    assert len(schedule_for(d,plan,'capacity'))==9
    for alteration in ('target','family','position'):
        bad=deepcopy(plan)
        if alteration=='target':bad['selected_source_budget'][RAW]['response_EOS_targets']-=1
        elif alteration=='family':bad['capacity_global_batches'].pop()
        else:bad['capacity_global_batches'][0][1]=[-1]
        with pytest.raises(ValueError):schedule_for(d,bad,'capacity')
