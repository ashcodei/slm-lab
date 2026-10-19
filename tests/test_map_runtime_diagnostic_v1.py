from pathlib import Path
import pytest

from slm_lab import code_quality


def request(code):
    return dict(language='Python',code=code,function='transform',cases=[dict(args=[[-3,5]],expected=[3,5]),dict(args=[[]],expected=[])],stress=[])


def test_diagnostic_adds_only_map_semantics_and_preserves_original_worker_and_import_restrictions(monkeypatch):
    original=code_quality.WORKER
    payload=request('def transform(xs):\n    return list(map(abs, xs))')
    before=code_quality.execute(payload);assert not before['functional_passed'] and "'map'" in before['error']
    fork=Path(__file__).resolve().parents[1]/'scripts/code_worker_map_diagnostic_v1.py'
    monkeypatch.setattr(code_quality,'WORKER',fork);after=code_quality.execute(payload)
    assert after['functional_passed'] and after['passed_cases']==2
    bad=code_quality.execute(request('import os\ndef transform(xs):\n    return xs'))
    assert not bad['functional_passed'] and 'Only public math' in bad['error']
    bad=code_quality.execute(request('def transform(xs):\n    return list(map(eval, xs))'))
    assert not bad['functional_passed'] and 'eval' in bad['error']
    assert fork!=original
