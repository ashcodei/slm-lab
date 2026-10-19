"""Exact full-source schedules and resource bounds for own public pretraining."""
from collections import Counter
import hashlib
import json
from pathlib import Path

from .public_pretraining_data_v2 import TokenBudgetSchedule,RAW,FUNCTIONS
from .focused_code_continuation_data_v3 import CODE,EXECUTION,STATE

CAPACITY_FAMILIES=(RAW,CODE,FUNCTIONS,EXECUTION,STATE,'packed:original_packed_code','education',
    'images:additional_photo_captions','words')


def schedule_for(data,plan,mode):
    if mode not in {'train','capacity'}:raise ValueError('Explicit owned pretraining mode required')
    settings=plan['actual_schedule_settings']
    schedule=TokenBudgetSchedule(data,**settings)
    if (schedule.identity_sha256!=plan['source_schedule_sha256'] or len(schedule)!=plan['planned_steps']
            or schedule.selected_budget(data)!=plan['selected_source_budget'] or data.budget()!=plan['complete_source_budget']):
        raise ValueError('Every raw/function/retention source occurrence and original target must equal the frozen plan')
    if mode=='capacity':
        entries=plan['capacity_global_batches']
        if tuple(n for n,_ in entries)!=CAPACITY_FAMILIES:
            raise ValueError('Every distinct actual full-size raw/code/function/state/image/word capacity source is required')
        for name,positions in entries:
            if not any(n==name and p==positions for n,p in schedule.entries):
                raise ValueError('Capacity must use an actual whole training batch, without invented or held positions')
        schedule.entries=[(n,list(p)) for n,p in entries]
        schedule.identity_sha256=hashlib.sha256(json.dumps(dict(production=schedule.identity_sha256,
            actual_capacity=schedule.entries),sort_keys=True).encode()).hexdigest()
    return schedule


def stage_ready(folder,identity,same,*,gpu_stage=False):
    folder=Path(folder)
    if same(identity):return False
    if (folder/'failure.json').exists() or not (folder/'complete.json').exists():
        raise ValueError('The exact full predecessor must complete; failed or partial work cannot be substituted')
    plan=json.loads((folder/'plan.json').read_text())
    if plan['owner']!=identity:raise ValueError('Exact predecessor process birth required')
    if gpu_stage:
        if any(same(child) for child in json.loads((folder/'children.json').read_text())):return False
        if not (folder/'restored.json').exists():raise ValueError('Every original GPU worker must finish and restore ownership')
    return True


def scratch_memory_safe(stat,limit,*,scratch_bytes=34*1024**3,working_headroom=96*1024**3):
    # Linux includes tmpfs/shmem in file. Reclaimable disk cache is not counted
    # twice as resident anonymous memory or as free additional RAM.
    resident=stat['anon']+stat['shmem']+stat['kernel']
    if resident+scratch_bytes+working_headroom>limit:
        raise ValueError('Disposable capacity snapshots and complete CPU working sets must fit the actual cgroup RAM limit')
    return dict(resident_nonreclaimable_bytes=resident,planned_tmpfs_upper_bytes=scratch_bytes,
        required_working_headroom_bytes=working_headroom,cgroup_limit_bytes=limit,tmpfs_is_volatile_not_persistent=True)
