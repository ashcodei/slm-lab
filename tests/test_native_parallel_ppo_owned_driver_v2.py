"""Actual non-neural source waiting and rejection of incomplete GPU evidence."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from slm_lab.native_parallel_ppo_owned_recipe_v2 import prepare,local_options,HISTORY

ROOT=Path(__file__).resolve().parents[1]
DRIVER=ROOT/'scripts/train_native_parallel_PPO_owned_GPU_v2.py'


def identity(pid):
    return dict(pid=pid,ticks=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19])


def module():
    spec=importlib.util.spec_from_file_location('owned_parallel_PPO_v2',DRIVER)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


def test_exact_live_reward_owner_waits_before_any_completed_source_or_neural_work(tmp_path,monkeypatch):
    import slm_lab.native_parallel_ppo_owned_recipe_v2 as recipe
    owner=identity(os.getpid());folder=tmp_path/'reward';folder.mkdir()
    (folder/'waiter-plan.json').write_text(json.dumps(dict(owner=owner)))
    def forbidden(*a,**k):pytest.fail('No completed joint/head readback while the exact actual reward owner lives')
    monkeypatch.setattr(recipe,'completed_joint',forbidden)
    assert prepare(ROOT,reward=folder,expected_owner=owner,joint=tmp_path/'absent-joint',
        source_recipe=tmp_path/'absent-source-recipe',runtime_sources={}) is None
    settings=local_options(ROOT)
    assert settings['max_new_tokens']==512 and settings['steps']==42893 and settings['prompts_per_update']==2
    assert HISTORY['maximum_records']==10724 and 10724*8-85786==6


def test_missing_exact_owner_never_adopts_a_partial_or_failed_head(tmp_path):
    child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(.2)'])
    owner=identity(child.pid);child.wait(timeout=5)
    (tmp_path/'waiter-plan.json').write_text(json.dumps(dict(owner=owner)))
    with pytest.raises(ValueError,match='entire original head epoch'):
        prepare(ROOT,reward=tmp_path,expected_owner=owner,joint=tmp_path/'none',source_recipe=tmp_path/'none',runtime_sources={})


def test_small_CPU_or_partial_rank_capacity_cannot_admit_full_four_L40_PPO(tmp_path):
    driver=module();driver.OUT=tmp_path;folder=tmp_path/'capacity';folder.mkdir()
    for rank in range(4):
        (folder/f'rank-{rank}.json').write_text(json.dumps(dict(complete=True,rank=rank,world_size=4,
            actual_native_values=1090146,actual_policy_reward_values=1092386,
            actual_full_horizon_capacity=dict(full_size_original_L40_horizon_capacity_passed=False))))
    with pytest.raises(ValueError,match='Every actual original L40'):
        driver.admitted_capacities(dict(reward_bundle=dict(sha256='a'*64)))


def test_actual_normal_source_observer_never_queries_GPU_memory_or_constructs_neural_runtime(tmp_path):
    reward=tmp_path/'reward';reward.mkdir();owner=identity(os.getpid())
    (reward/'waiter-plan.json').write_text(json.dumps(dict(owner=owner)))
    workspace=tmp_path/'owned';forbidden=tmp_path/'forbidden';log=tmp_path/'observer.log'
    wrapper='\n'.join([
        'import argparse,importlib.util', 'from pathlib import Path',
        'spec=importlib.util.spec_from_file_location("parallel_controller",'+repr(str(DRIVER))+')',
        'm=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)',
        'def forbidden(*args,**kwargs):',
        '    Path('+repr(str(forbidden))+').write_text("forbidden pre-source resource work")',
        '    raise RuntimeError("No GPU query/configuration/memory/neural work while exact source lives")',
        'm.configure=forbidden;m.available_memory=forbidden;m.subprocess.check_output=forbidden',
        'm.parent(argparse.Namespace(workspace=Path('+repr(str(workspace))+'),reward=Path('+repr(str(reward))+'),',
        '    joint=Path('+repr(str(tmp_path/'missing-joint'))+'),source_recipe=Path('+repr(str(tmp_path/'missing-source'))+')))'])
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
            assert plan['exact_original_reward_owner']==owner and plan['GPU_contexts']==plan['GPU_leases']==0
            assert plan['production_updates']==0
            assert '/torch/' not in Path(f'/proc/{child.pid}/maps').read_text()
            assert os.getpriority(os.PRIO_PROCESS,child.pid)==0 and os.sched_getscheduler(child.pid)==os.SCHED_OTHER
            for fd in Path(f'/proc/{child.pid}/fd').iterdir():
                try:assert 'nvidia' not in os.readlink(fd).lower()
                except FileNotFoundError:pass
        finally:
            if child.poll() is None:child.terminate()
            child.wait(timeout=5)
