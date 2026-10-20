import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/probe_supplemental_dense_code_GPU_v1.py'


def actor(pid):
    fields=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=pid,birth=fields[19])


def test_live_predecessor_or_partial_closed_scope_cannot_admit_new_GPU_work(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPT.parent));p=importlib.import_module('probe_supplemental_dense_code_GPU_v1')
    monkeypatch.setattr(p,'PREVIOUS',tmp_path);monkeypatch.setattr(p,'live',lambda _:True)
    def no_GPU():raise AssertionError('No GPU inspection or lease is allowed during this predecessor gate')
    monkeypatch.setattr(p.original,'gpu_snapshot',no_GPU)
    assert p.previous_ready({'pid':123,'birth':'456'}) is False
    monkeypatch.setattr(p,'live',lambda _:False)
    with pytest.raises(ValueError):p.previous_ready({'pid':123,'birth':'456'})
    (tmp_path/'complete.json').write_text(json.dumps(dict(complete=True,full_matched_assessments_complete=True,
        supplemental_GPU_contexts_released=True,full_source_training_updates_per_arm=230,old_processes_signalled=0)))
    (tmp_path/'restored.json').write_text(json.dumps(dict(own_children_terminal=True)))
    complete=dict(complete=True,programs=48,all425reserved_views_scored=True,
        original_bounded_knowledge_math_OCR_object_and_grounding_checks_complete=True,actual_registered_parameters=999706442)
    for endpoint in ['baseline','plain_tokens','eos_weight4']:
        d=tmp_path/endpoint;d.mkdir();(d/'assessment-complete.json').write_text(json.dumps(complete))
    assert p.previous_ready({'pid':123,'birth':'456'}) is True
    (tmp_path/'eos_weight4/assessment-complete.json').write_text(json.dumps(dict(complete,programs=24)))
    with pytest.raises(ValueError):p.previous_ready({'pid':123,'birth':'456'})


def test_actual_dense_probe_watchdog_stops_only_its_registered_disposable_child(tmp_path):
    child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)',str(SCRIPT),'--gpu-child'])
    child_actor=actor(child.pid);owner=subprocess.Popen([sys.executable,'-c','pass']);owner_actor=actor(owner.pid);owner.wait(timeout=5)
    protected={pid:actor(pid) if Path('/proc',str(pid)).exists() else None for pid in [293164,293165,293166,293167]}
    plan=dict(owner=owner_actor,old_owner=actor(os.getpid()),predecessor=actor(os.getpid()),expires_at_unix=time.time()+30)
    (tmp_path/'plan.json').write_text(json.dumps(plan));(tmp_path/'children.json').write_text(json.dumps([child_actor]))
    code=('import sys;from pathlib import Path;sys.path.insert(0,'+repr(str(SCRIPT.parent))+');'
        'import probe_supplemental_dense_code_GPU_v1 as p;p.OUT=Path('+repr(str(tmp_path))+');'
        'p.original.OUT=p.OUT;p.watchdog()')
    watcher=subprocess.Popen([sys.executable,'-c',code])
    try:
        watcher.wait(timeout=15);assert watcher.returncode==0
        child.wait(timeout=5);assert child.returncode!=0
        result=json.loads((tmp_path/'watchdog-cleanup.json').read_text())
        assert result['own_registered_children_only'] and result['old_processes_signalled']==0
        assert all((actor(pid) if Path('/proc',str(pid)).exists() else None)==a for pid,a in protected.items())
    finally:
        if watcher.poll() is None:watcher.terminate();watcher.wait(timeout=5)
        if child.poll() is None:child.terminate();child.wait(timeout=5)
