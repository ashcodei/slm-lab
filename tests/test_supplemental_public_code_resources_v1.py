import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/probe_supplemental_public_code_GPU_v1.py'
spec=importlib.util.spec_from_file_location('supplemental_capacity',SCRIPT)
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)


def sleeper():
    return subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)',str(SCRIPT),'--gpu-child'])


def test_cleanup_uses_exact_real_child_birth_and_cannot_signal_protected_actors():
    child=sleeper();actor=probe.identity(child.pid)
    try:
        wrong=dict(actor,birth=str(int(actor['birth'])+1))
        probe.stop_new(wrong,protected={os.getpid()},script=SCRIPT)
        assert child.poll() is None
        with pytest.raises(ValueError):probe.stop_new(actor,protected={os.getpid(),child.pid},script=SCRIPT)
        assert child.poll() is None
        probe.stop_new(actor,protected={os.getpid()},script=SCRIPT)
        child.wait(timeout=5);assert child.returncode!=0
    finally:
        if child.poll() is None:child.terminate();child.wait(timeout=5)


def test_separate_actual_watchdog_releases_only_registered_child_when_exact_parent_ends(tmp_path):
    child=sleeper();actor=probe.identity(child.pid)
    owner=subprocess.Popen([sys.executable,'-c','pass']);parent=probe.identity(owner.pid);owner.wait(timeout=5)
    original={pid:probe.identity(pid) for pid,_ in probe.OLD_WORKERS}
    plan=dict(owner=parent,old_owner=dict(pid=os.getpid(),birth=probe.identity(os.getpid())['birth']),expires_at_unix=time.time()+30)
    (tmp_path/'plan.json').write_text(json.dumps(plan));(tmp_path/'children.json').write_text(json.dumps([actor]))
    code='import importlib.util;from pathlib import Path;spec=importlib.util.spec_from_file_location("watcher",'+repr(str(SCRIPT))+');p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p);p.OUT=Path('+repr(str(tmp_path))+');p.watchdog()'
    watcher=subprocess.Popen([sys.executable,'-c',code])
    try:
        watcher.wait(timeout=15);assert watcher.returncode==0
        child.wait(timeout=5);assert child.returncode!=0
        report=json.loads((tmp_path/'watchdog-cleanup.json').read_text())
        assert report['own_registered_children_only'] and report['old_processes_signalled']==0
        assert all(probe.identity(pid)==value for pid,value in original.items())
    finally:
        if watcher.poll() is None:watcher.terminate();watcher.wait(timeout=5)
        if child.poll() is None:child.terminate();child.wait(timeout=5)
