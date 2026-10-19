import hashlib
import json

from slm_lab.public_code_benchmark import prepare
from slm_lab.verified_code import canonical_code


def test_benchmark_eligibility_preserves_ids_and_excludes_train_and_unsupported_cases(tmp_path):
    cache=tmp_path/'cache';cache.mkdir()
    rows=[{'task_id':12,'prompt':'Add one.','code':'def add_one(x):\n    return x+1',
           'test_list':['assert add_one(1)==2','assert add_one(2)==3'],'test_imports':[]},
          {'task_id':13,'prompt':'Return the same input.','code':'def same(x):\n    return x',
           'test_list':['assert same(1)==1','assert same(2)==2'],'test_imports':[]},
          {'task_id':14,'prompt':'A class-based task.','code':'class Thing:\n    pass',
           'test_list':['assert thing(1)==1','assert thing(2)==2'],'test_imports':[]},
          {'task_id':601,'prompt':'Training task stays out.','code':'def training(x):\n    return x',
           'test_list':['assert training(1)==1','assert training(2)==2'],'test_imports':[]}]
    path=cache/'sanitized-mbpp.json';path.write_text(json.dumps(rows))
    (cache/'sources.json').write_text(json.dumps({'revision':'fixture','files':[
        {'path':path.name,'url':'https://example.org/benchmark','sha256':hashlib.sha256(path.read_bytes()).hexdigest()}]}))
    training=tmp_path/'training';training.mkdir()
    digest=hashlib.sha256(canonical_code(rows[1]['code']).encode()).hexdigest()
    (training/'train.jsonl').write_text(json.dumps({'original_program_sha256':digest})+'\n')
    (training/'manifest.json').write_text(json.dumps({'train_sha256':hashlib.sha256((training/'train.jsonl').read_bytes()).hexdigest()}))
    output=tmp_path/'benchmark';manifest=prepare(cache,training,output)
    assert manifest['official_sanitized_test_problems']==3
    assert manifest['supported_problems']==1 and manifest['excluded_problems']==2
    questions=[json.loads(line) for line in (output/'questions.jsonl').read_text().splitlines()]
    assert [row['task_id'] for row in questions]==[12]
