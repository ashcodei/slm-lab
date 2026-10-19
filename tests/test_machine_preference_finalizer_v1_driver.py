import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest


def driver():
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'scripts'))
    spec=importlib.util.spec_from_file_location('machine_finalizer_unit',root/'scripts/finalize_machine_preference_student_CPU_v1.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def identity():
    return dict(pid=os.getpid(),ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19])


def fixture(tmp_path):
    m=driver();m.SOURCE=tmp_path;m.CHECKPOINT=tmp_path/'actor.pt';me=identity()
    def save(name,value):(tmp_path/name).write_text(json.dumps(value))
    m.CHECKPOINT.write_bytes(b'unit-metadata-only')
    save('plan.json',dict(owner=me,original_machine_pairs=1620,planned_steps=102,
        all_original_machine_pairs_once_each_arm=True,human_preferences=False))
    save('training-complete.json',dict(all1620original_machine_pairs_once_each_arm=True,actual_updates_each_arm=102,human_preferences=False))
    for arm in ['chosen_CE','machine_DPO']:
        save(arm+'-training-complete.json',dict(actual_updates=102,original_machine_pairs=1620,
            all_original_machine_pairs_once=True,rounded_repeats=0,human_preferences=False))
        workers=[]
        for rank in range(4):
            path=tmp_path/f'{arm}-private-{rank}.pt';path.write_bytes(('unit-private-'+str(rank)).encode())
            workers.append(dict(rank=rank,path=str(path),sha256=m.file_hash(path),actual_completed=102))
        save('train-'+arm+'-step102-distributed-state.json',dict(completed=102,all_four_private_states=workers,
            complete_distributed_state_requires_every_private_file=True))
    return m,me


def test_live_exact_owner_waits_and_terminal_owner_never_substitutes_partial_actor(tmp_path):
    m=driver();m.SOURCE=tmp_path;m.CHECKPOINT=tmp_path/'actor.pt';me=identity()
    assert m.source_ready(me) is False
    with pytest.raises(ValueError,match='ended'):m.source_ready(dict(me,ticks='0'))


def test_both_complete_original_source_epochs_and_exact_owner_are_required(tmp_path):
    m,me=fixture(tmp_path);assert m.source_ready(me)
    path=tmp_path/'chosen_CE-training-complete.json';r=json.loads(path.read_text());r['actual_updates']=101;path.write_text(json.dumps(r))
    with pytest.raises(ValueError,match='Both complete'):m.source_ready(me)
    r['actual_updates']=102;path.write_text(json.dumps(r))
    with pytest.raises(ValueError,match='exact owner'):m.source_ready(dict(me,ticks='0'))


def test_every_actual_private_worker_file_is_bound_and_corruption_rejects(tmp_path):
    m,me=fixture(tmp_path);assert m.source_ready(me)
    path=tmp_path/'machine_DPO-private-3.pt';path.write_bytes(b'changed-but-still-present')
    with pytest.raises(ValueError,match='Every exact committed'):m.source_ready(me)


def test_repeated_source_human_label_or_recorded_failure_cannot_be_finalized(tmp_path):
    m,me=fixture(tmp_path);path=tmp_path/'machine_DPO-training-complete.json';r=json.loads(path.read_text());r['rounded_repeats']=1;path.write_text(json.dumps(r))
    with pytest.raises(ValueError,match='Both complete'):m.source_ready(me)
    r['rounded_repeats']=0;r['human_preferences']=True;path.write_text(json.dumps(r))
    with pytest.raises(ValueError,match='Both complete'):m.source_ready(me)
    (tmp_path/'failure.json').write_text('{}')
    with pytest.raises(ValueError,match='failed'):m.source_ready(me)
