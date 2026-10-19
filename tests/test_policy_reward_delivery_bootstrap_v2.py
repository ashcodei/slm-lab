"""Pin every actual standalone query dependency without importing Torch."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from slm_lab.local_source_closure_v2 import python_sources
from test_policy_reward_delivery_driver_v1 import identity

ROOT=Path(__file__).resolve().parents[1]
DRIVER=ROOT/'scripts/deliver_native_policy_reward_CPU_v2.py'
QUERY=ROOT/'src/slm_lab/own_policy_reward_query_v1.py'


def test_bootstrap_closure_includes_preserved_controller_actual_CLI_and_complete_query_dependencies():
    paths=set(python_sources(ROOT,[DRIVER]))
    for p in [DRIVER,ROOT/'scripts/deliver_native_policy_reward_CPU_v1.py',QUERY,
              ROOT/'src/slm_lab/own_policy_reward_standalone_v1.py',ROOT/'src/slm_lab/own_native_query_v1.py']:
        assert p in paths


def test_real_v2_source_observer_pins_CLI_and_defers_neural_memory_and_worker_creation(tmp_path):
    PPO=tmp_path/'PPO';PPO.mkdir();expected=identity(os.getpid())
    (PPO/'waiter-plan.json').write_text(json.dumps(dict(owner=expected)))
    work=tmp_path/'delivery-v2';flag=tmp_path/'forbidden';log=tmp_path/'observer.log'
    wrapper='\n'.join(['import argparse,importlib.util','from pathlib import Path',
        'spec=importlib.util.spec_from_file_location("delivery_bootstrap",'+repr(str(DRIVER))+')',
        'm=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)',
        'def forbidden(*args,**kwargs):',
        '    Path('+repr(str(flag))+').write_text("forbidden pre-source work")',
        '    raise RuntimeError("No resource or neural work before complete original PPO owner")',
        'm.implementation.available_memory=forbidden;m.implementation.subprocess.Popen=forbidden',
        'm.parent(argparse.Namespace(workspace=Path('+repr(str(work))+'),PPO=Path('+repr(str(PPO))+')))'])
    with log.open('wb') as stream:
        child=subprocess.Popen([sys.executable,'-u','-c',wrapper],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
            env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
        try:
            deadline=time.monotonic()+15
            while not (work/'reports/waiting-progress.json').exists():
                if child.poll() is not None or time.monotonic()>deadline:pytest.fail(log.read_text())
                time.sleep(.05)
            assert child.poll() is None and not flag.exists()
            plan=json.loads((work/'reports/waiter-plan.json').read_text())
            assert plan['exact_original_PPO_owner']==expected and plan['GPU_contexts']==plan['GPU_leases']==0
            assert plan['source_files_sha256'][str(QUERY)]==hashlib.sha256(QUERY.read_bytes()).hexdigest()
            assert str(DRIVER) in plan['source_files_sha256']
            assert str(ROOT/'scripts/deliver_native_policy_reward_CPU_v1.py') in plan['source_files_sha256']
            assert '/torch/' not in Path(f'/proc/{child.pid}/maps').read_text()
            assert not (work/'reports/child.json').exists()
            assert os.sched_getscheduler(child.pid)==os.SCHED_OTHER and os.getpriority(os.PRIO_PROCESS,child.pid)==0
        finally:
            if child.poll() is None:child.terminate()
            child.wait(timeout=5)
