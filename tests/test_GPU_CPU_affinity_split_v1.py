"""Real fresh CPU children exercise restoration; no model/GPU work."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'scripts/probe_GPU_CPU_affinity_split_v1.py'
spec=importlib.util.spec_from_file_location('affinity_trial',SCRIPT)
trial=importlib.util.module_from_spec(spec);spec.loader.exec_module(trial)


def child(root):
    ready=root/'ready.json';flag=root/'spawn-second';second=root/'second.json'
    program='\n'.join(['import json,os,signal,subprocess,sys,threading,time','from pathlib import Path',
        'children=[]','threading.Thread(target=lambda:time.sleep(60),daemon=True).start()',
        'def end(*args): raise SystemExit(0)','signal.signal(signal.SIGTERM,end)',
        'try:',
        '    children.append(subprocess.Popen([sys.executable,"-c","import time;time.sleep(60)"]))',
        '    Path('+repr(str(ready))+').write_text(json.dumps({"grandchild":children[0].pid}))',
        '    spawned=False',
        '    while True:',
        '        if Path('+repr(str(flag))+').exists() and not spawned:',
        '            children.append(subprocess.Popen([sys.executable,"-c","import time;time.sleep(60)"]))',
        '            Path('+repr(str(second))+').write_text(json.dumps({"grandchild":children[-1].pid}))',
        '            spawned=True',
        '        time.sleep(.01)',
        'finally:',
        '    for child in children:',
        '        if child.poll() is None:child.terminate()',
        '    for child in children:child.wait(timeout=5)'])
    process=subprocess.Popen([sys.executable,'-c',program])
    deadline=time.monotonic()+10
    while not ready.exists():
        assert process.poll() is None and time.monotonic()<deadline
        time.sleep(.02)
    return process,flag,second


def plan(root,process):
    original=sorted(os.sched_getaffinity(process.pid));restricted=original[:-2] if len(original)>2 else original[:1]
    assert original!=restricted
    value=dict(owner=trial.identity(os.getpid()),targets=[trial.identity(process.pid)],original_cpus=original,
        restricted_cpus=restricted,expires_at_unix=time.time()+60)
    trial.write(root/'owner-plan.json',value)
    return value


def test_actual_threads_existing_and_new_descendants_restore_without_signals_or_model_work(tmp_path):
    process,flag,second=child(tmp_path);value=plan(tmp_path,process)
    try:
        before=trial.threads(value['targets'][0]);assert len(before)>=3
        assert trial.apply_split(tmp_path,value)>=3
        assert all(t['affinity']==value['restricted_cpus'] for t in trial.threads(value['targets'][0]))
        flag.touch();deadline=time.monotonic()+10
        while not second.exists():
            assert process.poll() is None and time.monotonic()<deadline
            time.sleep(.02)
        # This descendant was born after the last apply/register operation.
        second_pid=json.loads(second.read_text())['grandchild']
        assert sorted(os.sched_getaffinity(second_pid))==value['restricted_cpus']
        trial.restore(tmp_path,value,'actual test restoration')
        assert process.poll() is None
        assert all(t['affinity']==value['original_cpus'] for t in trial.threads(value['targets'][0]))
        assert json.loads((tmp_path/'restored.json').read_text())['processes_signalled']==0
        with pytest.raises(ValueError):trial.apply_split(tmp_path,value)
    finally:
        trial.restore(tmp_path,value,'test finally')
        process.terminate();process.wait(timeout=5)


@pytest.mark.parametrize('trigger',['owner_ended','lifetime_expired'])
def test_real_watchdog_restores_exact_surviving_threads_on_owner_death_or_expiry(tmp_path,trigger):
    process,_,_=child(tmp_path);value=plan(tmp_path,process)
    try:
        trial.apply_split(tmp_path,value)
        if trigger=='owner_ended':
            short=subprocess.Popen([sys.executable,'-c','import time;time.sleep(.1)'])
            value['owner']=trial.identity(short.pid);short.wait(timeout=5)
        else:value['expires_at_unix']=time.time()-1
        trial.write(tmp_path/'owner-plan.json',value)
        observer=subprocess.Popen([sys.executable,str(SCRIPT),'--watchdog',str(tmp_path)])
        observer.wait(timeout=10);assert observer.returncode==0
        assert process.poll() is None and all(t['affinity']==value['original_cpus'] for t in trial.threads(value['targets'][0]))
        assert json.loads((tmp_path/'restored.json').read_text())['exact_original_affinities_restored']
    finally:
        trial.restore(tmp_path,value,'test finally');process.terminate();process.wait(timeout=5)


def test_wrong_original_birth_rejects_before_any_thread_affinity_change(tmp_path):
    process,_,_=child(tmp_path);value=plan(tmp_path,process)
    try:
        value['targets'][0]['birth']=str(int(value['targets'][0]['birth'])+1)
        with pytest.raises(ValueError):trial.apply_split(tmp_path,value)
        assert sorted(os.sched_getaffinity(process.pid))==value['original_cpus']
        assert not (tmp_path/'affected.json').exists()
    finally:process.terminate();process.wait(timeout=5)
