import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from slm_lab import complete_joint_GPU_runtime_v1 as runtime
from slm_lab import complete_joint_owned_recipe_v1 as recipe

ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / 'scripts/train_complete_joint_owned_GPU_v1.py'


def driver(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    spec = importlib.util.spec_from_file_location('_joint_owned_driver_test', DRIVER)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


def test_actual_driver_import_is_non_neural_and_cannot_start_GPU_work():
    command = "import runpy,sys;runpy.run_path('scripts/train_complete_joint_owned_GPU_v1.py');assert 'torch' not in sys.modules;print('non-neural import verified')"
    result = subprocess.run([sys.executable, '-c', command], cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0 and result.stdout.strip() == 'non-neural import verified'


def test_actual_normal_CPU_observer_cannot_query_GPUs_or_construct_models_while_sources_wait(tmp_path):
    code = """import importlib.util,json,sys
from pathlib import Path
from types import SimpleNamespace
from slm_lab import complete_joint_owned_recipe_v1 as recipe
spec=importlib.util.spec_from_file_location('joint_owned_wait_fixture',Path('scripts/train_complete_joint_owned_GPU_v1.py'))
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
root=Path(sys.argv[1]);queue=root/'queue';queue.mkdir()
(queue/'plan.json').write_text(json.dumps({'owner':{'pid':99999970,'ticks':'1'}}))
recipe.prepare=lambda *a,**kw:None
def forbidden(*a,**kw):raise AssertionError('No GPU,lease or neural runtime before exact source preparation')
module.configure=forbidden;module.subprocess.check_output=forbidden
class EndObservation(Exception):pass
def stop(seconds):raise EndObservation()
module.time.sleep=stop
try:module.parent(SimpleNamespace(workspace=root/'workspace',queue=queue,initializer=root/'future',measurement=root/'future'))
except EndObservation:pass
else:raise AssertionError('The actual fixture must stop at the original-source wait')
assert 'torch' not in sys.modules
plan=json.loads((root/'workspace/reports/waiter-plan.json').read_text())
progress=json.loads((root/'workspace/reports/waiting-progress.json').read_text())
assert plan['GPU_contexts']==0 and plan['GPU_leases']==0 and not progress['GPU_work_started']
print('actual non-neural normal observer remained waiting')
"""
    result = subprocess.run([sys.executable, '-c', code, str(tmp_path)], cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'actual non-neural normal observer remained waiting'


def test_assessment_adapter_preserves_original_worker_and_restores_ownership_after_exception(tmp_path, monkeypatch):
    module = driver(monkeypatch)
    import train_native_code_suffix_aligned_GPU_v4 as original
    old = original.OUT, original.load_candidate, getattr(original, 'owner', None)
    module.OUT = tmp_path; module.owner = object()
    def controlled(endpoint, suite, physical):
        assert original.OUT == tmp_path and original.owner is module.owner
        assert original.load_candidate is module.load_candidate
        assert (endpoint, suite, physical) == ('joint', 'QA', 2)
        raise ValueError('controlled assessment failure')
    monkeypatch.setattr(original, 'assessment', controlled)
    with pytest.raises(ValueError, match='controlled assessment failure'):module.assessment('joint', 'QA', 2)
    assert original.OUT == old[0] and original.load_candidate is old[1] and getattr(original, 'owner', None) is old[2]


@pytest.mark.parametrize('kind', ['wrong_mode', 'shortened_source_budget'])
def test_invalid_full_runtime_recipe_rejects_before_any_model_or_CUDA_construction(tmp_path, monkeypatch, kind):
    plan = dict(format=recipe.FORMAT, full_original_budget=copy.deepcopy(recipe.FULL), options=recipe.OPTIONS,
        optimizer=recipe.OPTIMIZER, actor_settings=recipe.ACTOR, expected_native_values=984399690, expected_research_values=999706442)
    mode = 'capacity'
    if kind == 'wrong_mode':mode = 'partial_training'
    else:plan['full_original_budget']['updates'] = 2
    path = tmp_path / 'plan.json';path.write_text(json.dumps(plan))
    def forbidden(*a, **kw):raise AssertionError('Invalid source scope must reject before model construction')
    monkeypatch.setattr(runtime, 'load', forbidden)
    before = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match='exact owned full original'):runtime.build(ROOT, path, mode=mode)
    assert not torch.cuda.is_initialized() and torch.equal(before, torch.get_rng_state())
