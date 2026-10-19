import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]


def driver():
    sys.path.insert(0,str(ROOT/'scripts'))
    spec=importlib.util.spec_from_file_location('fast_joint_driver_v3_test',ROOT/'scripts/train_fast_joint_v3.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_capacity_inheritance_rejects_changed_training_or_neural_source(tmp_path):
    m=driver();previous=json.loads((ROOT/'reports/fast-joint-v2/plan.json').read_text())
    capacity=json.loads((ROOT/'reports/fast-joint-v2/capacity.json').read_text())
    assert m.validate_inherited_capacity(previous,previous,capacity)
    modified=copy.deepcopy(previous);modified['training_options']['maximum_microbatch']+=1
    with pytest.raises(ValueError,match='profile'):m.validate_inherited_capacity(modified,previous,capacity)
    modified=copy.deepcopy(previous)
    path=next(p for p in modified['source_files_sha256'] if p.endswith('/fast_joint_stage_v1.py'))
    modified['source_files_sha256'][path]='0'*64
    with pytest.raises(ValueError,match='graph'):m.validate_inherited_capacity(modified,previous,capacity)
    broken=copy.deepcopy(capacity);broken['proofs'][0]['actual_CPU_and_private_CUDA_RNG_restored']=False
    with pytest.raises(ValueError,match='recovery'):m.validate_inherited_capacity(previous,previous,broken)


@pytest.mark.parametrize('child_code,wall', [('import time; time.sleep(60)',.1),('raise SystemExit(7)',5)])
def test_development_timeout_or_failed_child_is_terminal_and_incomplete(tmp_path,monkeypatch,child_code,wall):
    m=driver();m.REPORTS=tmp_path;(tmp_path/'initial').mkdir();children=[]
    monkeypatch.setattr(m.owner,'owned',lambda plan:None)
    def launch(plan,args,subject,visible):
        p=subprocess.Popen([sys.executable,'-c',child_code],env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
        actor=dict(**m.owner.resources.identity(p.pid),subject=subject);children.append((p,actor))
        (tmp_path/'children.json').write_text(json.dumps([actor]));return p
    monkeypatch.setattr(m.owner,'launch',launch)
    try:
        m.isolated_pulse({'development_pulse_wall_seconds':wall},'initial',0,0)
        receipt=json.loads((tmp_path/'initial/pulse-incomplete-rank-0.json').read_text())
        assert receipt['measurement_complete'] is False
        assert receipt['full_endpoint_evaluation_still_required'] is True
        assert all(p.poll() is not None and not m.owner.resources.same(actor) for p,actor in children)
    finally:
        for p,actor in children:
            if p.poll() is None:m.owner.resources.stop_child(actor)
            p.wait(timeout=10)
