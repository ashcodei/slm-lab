"""The resource-owning pilot must reject an active original trainer first."""
import importlib.util
from pathlib import Path

import pytest


def test_active_original_trainer_rejects_before_GPU_query_signals_or_outputs(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / 'scripts/train_execution_code_pilot_v1.py'
    monkeypatch.syspath_prepend(str(script.parent))
    spec = importlib.util.spec_from_file_location('execution_pilot_driver_fixture', script)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'REPORTS', tmp_path / 'reports')
    monkeypatch.setattr(module, 'OUTPUT', tmp_path / 'checkpoints')
    monkeypatch.setattr(module.resources, 'same', lambda actor: dict(pid=actor['pid'], ticks=actor['ticks'], state='R'))
    def forbidden(*args, **kwargs):
        raise AssertionError('Active original work must be rejected before GPU queries or signals')
    monkeypatch.setattr(module.subprocess, 'run', forbidden)
    monkeypatch.setattr(module.os, 'kill', forbidden)
    with pytest.raises(ValueError, match='never preempt'):
        module.parent()
    assert not module.REPORTS.exists() and not module.OUTPUT.exists()
