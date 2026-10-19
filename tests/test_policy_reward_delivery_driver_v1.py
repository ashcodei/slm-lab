"""Actual source-only waiting; no weaker predecessor can deliver a model."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from slm_lab.policy_reward_delivery_recipe_v1 import prepare

ROOT=Path(__file__).resolve().parents[1]
DRIVER=ROOT/'scripts/deliver_native_policy_reward_CPU_v1.py'


def identity(pid):
    return dict(pid=pid,ticks=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19])


def test_exact_live_PPO_owner_waits_before_reading_or_constructing_a_delivery(tmp_path):
    owner=identity(os.getpid());(tmp_path/'waiter-plan.json').write_text(json.dumps(dict(owner=owner)))
    assert prepare(ROOT,PPO=tmp_path,expected_owner=owner,runtime_sources={}) is None


def test_terminal_partial_PPO_owner_cannot_be_adopted_as_completed_training(tmp_path):
    child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(.2)']);owner=identity(child.pid);child.wait(timeout=5)
    (tmp_path/'waiter-plan.json').write_text(json.dumps(dict(owner=owner)))
    with pytest.raises(ValueError,match='all original whole-source'):
        prepare(ROOT,PPO=tmp_path,expected_owner=owner,runtime_sources={})


def test_real_normal_delivery_observer_has_no_Torch_GPU_memory_query_or_worker_before_exact_source_completion(tmp_path):
    PPO=tmp_path/'PPO';PPO.mkdir();owner=identity(os.getpid())
    (PPO/'waiter-plan.json').write_text(json.dumps(dict(owner=owner)))
    workspace=tmp_path/'delivery';forbidden=tmp_path/'forbidden';log=tmp_path/'observer.log'
    wrapper='\n'.join(['import argparse,importlib.util','from pathlib import Path',
        'spec=importlib.util.spec_from_file_location("delivery_controller",'+repr(str(DRIVER))+')',
        'm=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)',
        'def forbidden(*args,**kwargs):',
        '    Path('+repr(str(forbidden))+').write_text("forbidden pre-source delivery work")',
        '    raise RuntimeError("No neural memory query or delivery worker before complete source")',
        'm.available_memory=forbidden;m.subprocess.Popen=forbidden',
        'm.parent(argparse.Namespace(workspace=Path('+repr(str(workspace))+'),PPO=Path('+repr(str(PPO))+')))'])
    with log.open('wb') as stream:
        child=subprocess.Popen([sys.executable,'-u','-c',wrapper],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
            env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
        try:
            deadline=time.monotonic()+15
            while not (workspace/'reports/waiting-progress.json').exists():
                if child.poll() is not None or time.monotonic()>deadline:pytest.fail(log.read_text())
                time.sleep(.05)
            assert child.poll() is None and not forbidden.exists()
            plan=json.loads((workspace/'reports/waiter-plan.json').read_text())
            assert plan['exact_original_PPO_owner']==owner and plan['GPU_contexts']==plan['GPU_leases']==0
            assert '/torch/' not in Path(f'/proc/{child.pid}/maps').read_text()
            assert os.getpriority(os.PRIO_PROCESS,child.pid)==0 and os.sched_getscheduler(child.pid)==os.SCHED_OTHER
            assert not (workspace/'reports/child.json').exists()
            for fd in Path(f'/proc/{child.pid}/fd').iterdir():
                try:assert 'nvidia' not in os.readlink(fd).lower()
                except FileNotFoundError:pass
        finally:
            if child.poll() is None:child.terminate()
            child.wait(timeout=5)
