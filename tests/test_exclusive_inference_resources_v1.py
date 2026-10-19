import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from slm_lab import exclusive_inference_resources_v1 as r


def lease(tmp_path,monkeypatch):
    source=tmp_path/'scripts/run_exclusive_code_objective_assessments_v1.py';source.parent.mkdir();source.write_text('# own test fixture\n')
    path=tmp_path/'reports/code-objective-exclusive-assessment-v1/lease.json';path.parent.mkdir(parents=True)
    d=dict(format='slm-exclusive-inference-lease-v1',parent_pid=123,parent_start_ticks='42',
        parent_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),reserved_GPU_UUIDs=['testGPU'],
        stopped_predecessor_status_sha256={})
    path.write_text(json.dumps(d));monkeypatch.setenv(r.ENV,str(path))
    monkeypatch.setattr(r,'identity',lambda pid:dict(pid=123,ticks='42',state='S',argv=[sys.executable,str(source)]))
    monkeypatch.setattr(r,'checkpoint_writers',lambda root:[])
    monkeypatch.setattr(r,'gpu_memory_for_pid',lambda pid:{'testGPU':256})
    monkeypatch.setattr(r.shutil,'disk_usage',lambda root:SimpleNamespace(free=r.MINIMUM_FREE+r.REPORT_BUDGET+1024))
    return path,d


def test_verified_lease_uses_zero_credit_and_funds_all_reports(tmp_path,monkeypatch):
    lease(tmp_path,monkeypatch);policy=r.inference_reservation(tmp_path)
    assert policy['credited_bytes']==0 and policy['minimum_reserve_bytes']==16*1024**3


@pytest.mark.parametrize('fault',['dead','PID_reused','writer','GPU_released','GPU_at_idle_threshold','disk','source_changed'])
def test_unverified_exclusivity_or_unfunded_commit_rejects(tmp_path,monkeypatch,fault):
    path,d=lease(tmp_path,monkeypatch)
    if fault=='dead':monkeypatch.setattr(r,'identity',lambda pid:None)
    if fault=='PID_reused':d['parent_start_ticks']='43';path.write_text(json.dumps(d))
    if fault=='writer':monkeypatch.setattr(r,'checkpoint_writers',lambda root:[dict(pid=456)])
    if fault=='GPU_released':monkeypatch.setattr(r,'gpu_memory_for_pid',lambda pid:{})
    if fault=='GPU_at_idle_threshold':monkeypatch.setattr(r,'gpu_memory_for_pid',lambda pid:{'testGPU':128})
    if fault=='disk':monkeypatch.setattr(r.shutil,'disk_usage',lambda root:SimpleNamespace(free=r.MINIMUM_FREE+10))
    if fault=='source_changed':(tmp_path/'scripts/run_exclusive_code_objective_assessments_v1.py').write_text('changed')
    with pytest.raises(ValueError):r.inference_reservation(tmp_path)


def test_report_commit_keeps_remaining_reserve(tmp_path,monkeypatch):
    lease(tmp_path,monkeypatch)
    with pytest.raises(ValueError):r.validate_active_lease(tmp_path,commit_bytes=r.REPORT_BUDGET+2048)


@pytest.mark.parametrize('mode',['idle','started','PID_reused'])
def test_actual_waiter_signals_are_identity_and_idle_checked(tmp_path,mode):
    script=tmp_path/'waiter.py';script.write_text('import time\nwhile True: time.sleep(.1)\n')
    p=subprocess.Popen([sys.executable,str(script)])
    status=tmp_path/'status.json';status.write_text(json.dumps(dict(status='waiting for resources')))
    generation=tmp_path/'generations.jsonl'
    receipt=dict(pid=p.pid,start_ticks=r.identity(p.pid)['ticks'],argv=[sys.executable,str(script)])
    try:
        if mode=='started':generation.write_text('preserve completed raw output\n')
        if mode=='PID_reused':receipt['start_ticks']='wrong'
        if mode=='idle':
            result=r.retire_idle_waiter(receipt,status,[generation]);p.wait(timeout=3)
            assert result['retired'] and p.returncode==-15
        else:
            with pytest.raises(ValueError):r.retire_idle_waiter(receipt,status,[generation])
            assert p.poll() is None and r.identity(p.pid)['state'] not in {'T','t'}
    finally:
        if p.poll() is None:p.terminate();p.wait(timeout=3)
