import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
spec = importlib.util.spec_from_file_location('own_APPS_diagnostic',ROOT/'scripts/assess_apps_original_test_diagnostic_v1.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_complete_delivery_and_source_preselection():
    assert module.extract('```python\ndef f():\n return 1\n```') == 'def f():\n return 1'
    assert module.extract('```python\ndef f(): pass') is None
    assert module.extract('```python\nx=1\n```\nextra prose') is None
    assert module.extract('def f():\n return 1') == 'def f():\n return 1'
    rows = []
    for difficulty in ['introductory','interview','competition']:
        rows.extend(dict(id=difficulty+str(i),mode='stdin',difficulty=difficulty,
            supported_by_declared_local_protocol=True) for i in range(20))
    rows.append(dict(id='function',mode='function',difficulty='introductory',supported_by_declared_local_protocol=True))
    rows.append(dict(id='review',mode='function',difficulty='introductory',supported_by_declared_local_protocol=False))
    first = module.selected_rows(rows)
    assert first == module.selected_rows(list(reversed(rows)))
    assert len(first) == 49 and 'function' in [r['id'] for r in first] and 'review' not in [r['id'] for r in first]


def test_active_or_unfinished_original_assessments_reject_before_GPU(tmp_path,monkeypatch):
    monkeypatch.setattr(module.owner,'REPORTS',tmp_path)
    monkeypatch.setattr(module.owner,'ownership',lambda plan:None)
    (tmp_path/'status.json').write_text(json.dumps(dict(phase='train')))
    with pytest.raises(ValueError,match='window is not active'):
        module.idle_window({})
    (tmp_path/'status.json').write_text(json.dumps(dict(phase='candidate')))
    with pytest.raises(ValueError,match='must finish first'):
        module.idle_window({})
    (tmp_path/'APPS-validation').mkdir()
    (tmp_path/'APPS-validation/candidate-complete.json').write_text('{}')
    (tmp_path/'candidate').mkdir()
    for i in [1,2,3]:
        (tmp_path/f'candidate/suite-{i}-complete.json').write_text('{}')
    actor = dict(pid=123,ticks='456',subject='candidate-full-APPS-validation-shard-0')
    (tmp_path/'children.json').write_text(json.dumps([actor]))
    monkeypatch.setattr(module.owner.resources,'same',lambda actor:dict(state='R'))
    with pytest.raises(ValueError,match='remains live'):
        module.idle_window({})
    monkeypatch.setattr(module.owner.resources,'same',lambda actor:None)
    module.idle_window({})
