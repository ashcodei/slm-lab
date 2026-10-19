import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('owned_affinity',ROOT/'scripts/widen_owned_CPU_checks_v1.py')
manager=importlib.util.module_from_spec(spec);spec.loader.exec_module(manager)


@pytest.fixture
def owned(tmp_path):
    mask=sorted(os.sched_getaffinity(0))
    if len(mask)<2:pytest.skip('At least two actually approved CPUs are required')
    entry=tmp_path/'worker.py';entry.write_text('''import os,time,threading,json
from pathlib import Path
os.nice(19);os.sched_setscheduler(0,os.SCHED_IDLE,os.sched_param(0))
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
threading.Thread(target=lambda:time.sleep(120),daemon=True).start()
print(json.dumps(dict(pid=os.getpid(),birth=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19])),flush=True)
time.sleep(120)
''')
    process=subprocess.Popen([sys.executable,str(entry)],stdout=subprocess.PIPE,text=True)
    try:
        owner=json.loads(process.stdout.readline());plan=tmp_path/'owner-plan.json';plan.write_text(json.dumps(dict(owner=owner,source_files_sha256={str(entry):hashlib.sha256(entry.read_bytes()).hexdigest()})))
        yield dict(**owner,owner_plan=plan,entrypoint=entry),mask,process
    finally:
        process.terminate();process.wait(timeout=10)


def test_every_actual_owned_thread_widens_without_priority_source_or_owner_plan_changes(owned):
    request,mask,process=owned;before=Path(request['owner_plan']).read_bytes()
    record=manager.widen(request,mask,protected=set())
    assert process.poll() is None and Path(request['owner_plan']).read_bytes()==before
    assert len(record['before']['threads'])>=2 and all(len(t['affinity'])==1 for t in record['before']['threads'])
    assert all(t['affinity']==mask and t['nice']==19 and t['policy']==os.SCHED_IDLE for t in record['after']['threads'])
    assert record['priority_and_policy_unchanged']


@pytest.mark.parametrize('damage',['wrong_birth','protected_PID','changed_source'])
def test_foreign_birth_protected_worker_or_changed_source_reject_before_any_thread_changes(owned,damage):
    request,mask,process=owned;original_mask=os.sched_getaffinity(process.pid);protected=set()
    if damage=='wrong_birth':request={**request,'birth':'0'}
    elif damage=='protected_PID':protected={process.pid}
    else:Path(request['entrypoint']).write_text('different source bytes\n')
    with pytest.raises(ValueError):manager.widen(request,mask,protected=protected)
    assert process.poll() is None and os.sched_getaffinity(process.pid)==original_mask
    assert all(os.sched_getaffinity(int(p.name))==original_mask for p in (Path('/proc')/str(process.pid)/'task').iterdir())
