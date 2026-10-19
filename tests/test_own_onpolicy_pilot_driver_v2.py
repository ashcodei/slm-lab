import importlib.util
import json
from pathlib import Path
import sys

import pytest


def driver():
    scripts = Path(__file__).resolve().parents[1] / 'scripts'
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location('ownpolicy_pilot_driver_v2_test', scripts / 'train_own_onpolicy_pilot_v2.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_live_predecessor_or_registered_rank_rejects_before_resource_queries_or_output(monkeypatch, tmp_path):
    module = driver(); monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module, 'REPORTS', tmp_path / 'pilot')
    prior = tmp_path / 'reports/own-code-generation-repair-v1'; prior.mkdir(parents=True)
    (prior / 'plan.json').write_text(json.dumps({'owner': {'pid': 11, 'ticks': '21'}}))
    (prior / 'children.json').write_text(json.dumps([{'pid': 12, 'ticks': '22'}]))
    monkeypatch.setattr(module.subprocess, 'check_output', lambda *a, **k: pytest.fail('No GPU queries with live predecessor'))
    monkeypatch.setattr(module.owner, 'idle_original', lambda: pytest.fail('Never hold original waiter with live predecessor'))
    for live in [11, 12]:
        monkeypatch.setattr(module.owner.resources, 'same', lambda actor: actor if actor['pid'] == live else None)
        with pytest.raises(ValueError, match='actually end'):
            module.parent()
        assert not module.REPORTS.exists()


def test_active_original_waiter_rejects_before_source_preparation_or_resource_queries(monkeypatch, tmp_path):
    module = driver(); monkeypatch.setattr(module, 'REPORTS', tmp_path / 'pilot')
    monkeypatch.setattr(module, 'previous_ended', lambda: None)
    monkeypatch.setattr(module.owner.resources, 'same', lambda actor: dict(**actor, state='R'))
    monkeypatch.setattr(module.subprocess, 'check_output', lambda *a, **k: pytest.fail('No GPU queries with active waiter'))
    with pytest.raises(ValueError, match='Never preempt original work'):
        module.parent()
    assert not module.REPORTS.exists()


def test_fixed_head_factory_contract_survives_real_code_update_and_all_shared_routes(monkeypatch):
    import torch
    from slm_lab.own_assistant_v1 import OwnAssistant
    from slm_lab.frozen_base_pointer_stage_v1 import make_head
    from slm_lab.own_onpolicy_stage_v1 import OwnOnPolicyStage
    from slm_lab.own_code_training_stage_v1 import make_optimizer
    from test_own_onpolicy_stage_v1 import fixtures
    module = driver()
    model, streams, pairs, reference, options, sources = fixtures()
    head = make_head(model.base, rank=8)
    combined = OwnAssistant(model, head).eval()
    module.check_fixed_paths(combined)
    ids = torch.tensor([[1, 4, 7]]); pixels = torch.zeros(1, 3, 32, 32)
    with torch.no_grad():
        general = combined.generate(ids, pixels, route='general', max_new_tokens=2)
        grounding = combined.locate(ids, pixels)['logits'].clone()
    versions = [(id(p), p._version, p.data_ptr()) for p in head.parameters()]
    before = {n: p.clone() for n, p in head.named_parameters()}
    kernel = OwnOnPolicyStage(model, make_optimizer(model, options), streams, pairs, reference, options, sources, 'cpu')
    kernel.update(); combined.eval(); module.check_fixed_paths(combined)
    assert versions == [(id(p), p._version, p.data_ptr()) for p in head.parameters()]
    assert all(torch.equal(before[n], p) for n, p in head.named_parameters())
    assert all(p.grad is None for p in head.parameters())
    with torch.no_grad():
        assert torch.equal(general, combined.generate(ids, pixels, route='general', max_new_tokens=2))
        assert torch.equal(grounding, combined.locate(ids, pixels)['logits'])
        assert torch.isfinite(combined(ids, pixels, route='code')['logits']).all()
    head.query_projection.requires_grad_(False)
    with pytest.raises(ValueError):
        module.check_fixed_paths(combined)
