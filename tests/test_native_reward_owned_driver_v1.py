"""The actual normal observer must wait without a model, CUDA or GPU lease."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT/'scripts/train_native_human_reward_owned_GPU_v1.py'


def load(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    spec = importlib.util.spec_from_file_location('native_reward_owned_driver_fixture', DRIVER)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def test_driver_import_is_non_neural():
    result = subprocess.run([sys.executable,'-c',
        "import runpy,sys;runpy.run_path('scripts/train_native_human_reward_owned_GPU_v1.py');assert 'torch' not in sys.modules;print('non-neural')"],
        cwd=ROOT,capture_output=True,text=True,timeout=30)
    assert result.returncode == 0 and result.stdout.strip() == 'non-neural'


def test_partial_capacity_cannot_admit_features_or_change_the_complete_model_scope(tmp_path,monkeypatch):
    module=load(monkeypatch);module.OUT=tmp_path
    (tmp_path/'capacity').mkdir();sources=dict(native_checkpoint_sha256='a'*64)
    for rank in range(4):(tmp_path/'capacity'/('rank-'+str(rank)+'.json')).write_text(json.dumps(dict(complete=True)))
    with pytest.raises(ValueError,match='Every actual original L40'):
        module.capacities(dict(feature_options=dict(token_budget=16384,maximum_sequences=16)),sources)


def test_actual_normal_observer_waits_for_live_exact_joint_owner_before_any_neural_or_GPU_work(tmp_path):
    code = """import importlib.util,json,os,sys
from pathlib import Path
from types import SimpleNamespace
root=Path(sys.argv[1]);joint=root/'joint';joint.mkdir()
owner={'pid':os.getpid(),'ticks':Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]}
(joint/'waiter-plan.json').write_text(json.dumps({'owner':owner}))
spec=importlib.util.spec_from_file_location('actual_reward_observer',Path('scripts/train_native_human_reward_owned_GPU_v1.py'))
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
def forbidden(*a,**kw):raise AssertionError('No model,GPU query or lease before exact full original joint completion')
module.configure=forbidden;module.subprocess.check_output=forbidden;module.memory_available=forbidden
class EndObservation(Exception):pass
def stop(seconds):raise EndObservation()
module.time.sleep=stop
try:module.parent(SimpleNamespace(workspace=root/'workspace',joint=joint))
except EndObservation:pass
else:raise AssertionError('The actual observer must stay at the exact live source wait')
assert 'torch' not in sys.modules
plan=json.loads((root/'workspace/reports/waiter-plan.json').read_text())
waiting=json.loads((root/'workspace/reports/waiting-progress.json').read_text())
assert plan['exact_original_joint_owner']==owner and plan['GPU_contexts']==plan['GPU_leases']==plan['head_updates']==0
assert not waiting['GPU_work_started'] and waiting['head_updates']==0
print('actual normal observer waited without neural or GPU work')
"""
    result=subprocess.run([sys.executable,'-c',code,str(tmp_path)],cwd=ROOT,capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
    assert result.stdout.strip()=='actual normal observer waited without neural or GPU work'
