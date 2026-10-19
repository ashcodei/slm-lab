import importlib.util
import json
from pathlib import Path
import sys

import pytest

from slm_lab.corpus import file_hash


def module():
    path=Path('scripts/assess_own_code_delta_public_v1.py').resolve()
    spec=importlib.util.spec_from_file_location('checked_own_delta_benchmark',path);loaded=importlib.util.module_from_spec(spec)
    sys.path.insert(0,str(path.parent))
    try:spec.loader.exec_module(loaded)
    finally:sys.path.pop(0)
    return loaded


def test_original_raw_shard_union_is_exact_disjoint_and_complete(tmp_path):
    driver=module();data=tmp_path/'original';data.mkdir()
    rows=[dict(id=str(i),task_id=i,function='f',signature='def f(x):',specification='Return x.',cases=[dict(input=i,expected=i)]) for i in range(11)]
    source=data/'questions.jsonl';source.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (data/'manifest.json').write_text(json.dumps(dict(questions_sha256=file_hash(source),official_sanitized_test_problems=20,
        excluded_problems=9,limitations=['unit source fixture'])))
    result=driver.seal_shards(source,tmp_path/'shards');assert result==rows
    rebuilt={}
    for shard in range(4):
        folder=tmp_path/'shards'/f'shard-{shard}'
        selected=list(map(json.loads,(folder/'questions.jsonl').read_text().splitlines()))
        assert selected==rows[shard::4]
        manifest=json.loads((folder/'manifest.json').read_text())
        assert manifest['questions_sha256']==file_hash(folder/'questions.jsonl')
        for i,line in enumerate((folder/'questions.jsonl').read_bytes().splitlines(keepends=True)):
            rebuilt[shard+i*4]=line
    assert b''.join(rebuilt[i] for i in range(len(rows)))==source.read_bytes()


def test_either_live_exact_owner_rejects_before_gpu_query_signal_or_output(monkeypatch,tmp_path):
    driver=module();reports=tmp_path/'must-not-exist';monkeypatch.setattr(driver,'REPORTS',reports)
    actions=[]
    monkeypatch.setattr(driver.owner.resources,'same',lambda actor:actor if actor['pid']==12 else None)
    monkeypatch.setattr(driver.subprocess,'run',lambda *a,**k:actions.append('GPU query'))
    monkeypatch.setattr(driver.os,'kill',lambda *a,**k:actions.append('signal'))
    with pytest.raises(ValueError,match='math/grounding owners must end'):
        driver.parent({},[dict(pid=11,ticks='old'),dict(pid=12,ticks='live')])
    assert not actions and not reports.exists()
