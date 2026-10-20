import json
from pathlib import Path
import subprocess
import sys


SCRIPT = Path(__file__).resolve().parents[1]/'scripts/train_supplemental_public_code_GPU_v1.py'


def actor(pid):
    fields = Path('/proc', str(pid), 'stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=pid, birth=fields[19])


def test_actual_epoch_watchdog_cleans_only_its_registered_child_after_parent_death(tmp_path):
    import os
    child = subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)',str(SCRIPT),'--gpu-child'])
    child_actor = actor(child.pid)
    owner = subprocess.Popen([sys.executable,'-c','pass'])
    owner_actor = actor(owner.pid)
    owner.wait(timeout=5)
    original = {pid:actor(pid) if Path('/proc',str(pid)).exists() else None for pid in [293164,293165,293166,293167]}
    plan = dict(owner=owner_actor, old_owner=actor(os.getpid()), expires_at_unix=__import__('time').time()+30)
    (tmp_path/'plan.json').write_text(json.dumps(plan))
    (tmp_path/'children.json').write_text(json.dumps([child_actor]))
    code = ('import sys;from pathlib import Path;sys.path.insert(0,'+repr(str(SCRIPT.parent))+');'
        'import train_supplemental_public_code_GPU_v1 as p;p.OUT=Path('+repr(str(tmp_path))+');'
        'p.capacity.OUT=p.OUT;p.watchdog()')
    watcher = subprocess.Popen([sys.executable,'-c',code])
    try:
        watcher.wait(timeout=15)
        assert watcher.returncode == 0
        child.wait(timeout=5)
        assert child.returncode != 0
        result = json.loads((tmp_path/'watchdog-cleanup.json').read_text())
        assert result['own_registered_children_only'] and result['old_processes_signalled'] == 0
        assert all((actor(pid) if Path('/proc',str(pid)).exists() else None) == value for pid,value in original.items())
    finally:
        if watcher.poll() is None: watcher.terminate();watcher.wait(timeout=5)
        if child.poll() is None: child.terminate();child.wait(timeout=5)
