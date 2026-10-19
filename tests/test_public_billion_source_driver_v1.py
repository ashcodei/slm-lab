import importlib.util
import io
import json
from pathlib import Path
import sys

import pytest


def driver():
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'scripts'))
    spec=importlib.util.spec_from_file_location('billion_public_source_driver_unit',root/'scripts/collect_billion_public_text_candidate_CPU_v1.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


class Response(io.BytesIO):
    def __init__(self,rows,link=''):
        super().__init__(json.dumps(rows).encode());self.headers={'Link':link}


def item(name,sha):return dict(type='file',path='data/'+name+'.parquet',size=123,lfs=dict(oid=sha,size=123))


def test_complete_pinned_source_catalog_follows_all_pages_and_preserves_original_order(monkeypatch):
    m=driver();source=dict(repo='publisher/public',revision='a'*40,family='code',path='data/first.parquet',sha256='b'*64)
    next_url='https://huggingface.co/api/datasets/publisher/public/tree/'+'a'*40+'/data?cursor=2';calls=[]
    def request(req,timeout):
        calls.append(req.full_url)
        if len(calls)==1:return Response([item('first','b'*64)],'<'+next_url+'>; rel="next"')
        return Response([item('second','c'*64)])
    monkeypatch.setattr(m,'urlopen',request);files=m.catalog(source)
    assert [x['sha256'] for x in files]==['b'*64,'c'*64] and calls[1]==next_url
    assert all(x['revision']==source['revision'] and x['repo']==source['repo'] for x in files)


def test_changed_first_file_or_unpinned_pagination_cannot_create_candidate_catalog(monkeypatch):
    m=driver();source=dict(repo='publisher/public',revision='a'*40,family='code',path='data/first.parquet',sha256='b'*64)
    monkeypatch.setattr(m,'urlopen',lambda *a,**kw:Response([item('first','0'*64)]))
    with pytest.raises(ValueError,match='independently checked'):m.catalog(source)
    monkeypatch.setattr(m,'urlopen',lambda *a,**kw:Response([item('first','b'*64)],'<https://example.org/unpinned>; rel="next"'))
    with pytest.raises(ValueError,match='pinned public repository'):m.catalog(source)
