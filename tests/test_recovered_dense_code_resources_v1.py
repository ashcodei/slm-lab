import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/train_recovered_dense_code_GPU_v1.py'


def actor(pid):
    s=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split();return dict(pid=pid,birth=s[19])


def test_exact_four_GPU_guard_rejects_unknown_context_wrong_device_and_insufficient_margin(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPT.parent));p=importlib.import_module('train_recovered_dense_code_GPU_v1')
    monkeypatch.setattr(p,'OUT',tmp_path);monkeypatch.setattr(p,'SCRATCH',tmp_path)
    owner=actor(os.getpid());plan=dict(owner=owner,expires_at_unix=time.time()+30)
    monkeypatch.setattr(p,'lease_owner',lambda:owner['pid']);gpu=[dict(index=i,uuid=u,name='NVIDIA L40',free_MiB=40000) for i,u in enumerate(p.UUIDS)]
    known={'pid':500001,'birth':'123'};(tmp_path/'children.json').write_text(json.dumps([dict(actor=known,gpu=0,profile='rate3e6')]))
    real_identity=p.identity;monkeypatch.setattr(p,'identity',lambda pid:known if pid==known['pid'] else real_identity(pid))
    apps=[dict(uuid=p.UUIDS[0],pid=known['pid'],used_MiB=100)]
    monkeypatch.setattr(p,'snapshot',lambda:(gpu,apps));p._last_check=0;p.checked(plan)
    for bad in [[dict(uuid=p.UUIDS[0],pid=999999,used_MiB=100)], [dict(uuid=p.UUIDS[1],pid=known['pid'],used_MiB=100)]]:
        monkeypatch.setattr(p,'snapshot',lambda bad=bad:(gpu,bad));p._last_check=0
        with pytest.raises(ValueError):p.checked(plan)
    gpu[0]['free_MiB']=12*1024-1;monkeypatch.setattr(p,'snapshot',lambda:(gpu,apps));p._last_check=0
    with pytest.raises(ValueError):p.checked(plan)


def test_actual_watchdog_terminates_only_registered_worker_and_leaves_unregistered_process_alive(tmp_path):
    child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)',str(SCRIPT),'--gpu-child'])
    unrelated=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])
    owner=subprocess.Popen([sys.executable,'-c','pass']);owner_actor=actor(owner.pid);owner.wait(timeout=5)
    (tmp_path/'plan.json').write_text(json.dumps(dict(owner=owner_actor,expires_at_unix=time.time()+30)))
    (tmp_path/'children.json').write_text(json.dumps([dict(actor=actor(child.pid),gpu=0,profile='rate3e6')]))
    code=('import sys;from pathlib import Path;sys.path.insert(0,'+repr(str(SCRIPT.parent))+');'
        'import train_recovered_dense_code_GPU_v1 as p;p.OUT=Path('+repr(str(tmp_path))+');p.watchdog()')
    watcher=subprocess.Popen([sys.executable,'-c',code])
    try:
        watcher.wait(timeout=15);assert watcher.returncode==0;child.wait(timeout=5)
        assert child.returncode!=0 and unrelated.poll() is None
        result=json.loads((tmp_path/'watchdog-cleanup.json').read_text())
        assert result['own_registered_children_only'] and result['external_processes_signalled']==0
    finally:
        for p in [watcher,child,unrelated]:
            if p.poll() is None:p.terminate();p.wait(timeout=5)
