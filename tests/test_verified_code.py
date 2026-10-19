import hashlib
import json

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

from slm_lab.code_quality import execute
from slm_lab.verified_code import assertion_cases, doctest_cases, prepare


def test_literal_examples_keep_tuple_types_without_evaluating_calls():
    name,cases=assertion_cases(['assert pair((1, 2), 3) == ((1, 2), 3)',
                               'assert pair(0, None) == (0, None)'])
    assert name=='pair'
    result=execute({'language':'Python','code':'def pair(a,b):\n    return (a,b)',
                    'function':name,'cases':cases,'stress':[]})
    assert result['functional_passed'] and result['passed_cases']==2
    with pytest.raises(ValueError):assertion_cases(['assert pair(open("private"), 1) == 0'])
    with pytest.raises(ValueError):assertion_cases(['assert pair(1, 2) == dangerous()'])


def test_doctest_selection_ignores_setup_and_other_entry_points():
    text='>>> import os\n>>> bump(1)\n2\n>>> other(1)\n2\n>>> bump(-1)\n0\n'
    cases=doctest_cases(text,'bump')
    assert len(cases)==2
    result=execute({'language':'Python','code':'def bump(x):\n    return x+1',
                    'function':'bump','cases':cases,'stress':[]})
    assert result['functional_passed']
    assert doctest_cases('>>>bump(1)\n2\n','bump')==[]


def test_preparation_keeps_official_test_ids_out_and_rejects_wrong_programs(tmp_path):
    cache=tmp_path/'cache';cache.mkdir()
    rows=[{'task_id':12,'text':'A held-out test specification.','code':'def held(x):\n    return x',
           'test_list':['assert held(1)==1','assert held(2)==2']},
          {'task_id':601,'text':'Return the ordered pair of two inputs.','code':'def pair(a,b):\n    return (a,b)',
           'test_list':['assert pair(1,2)==(1,2)','assert pair(0,-1)==(0,-1)']},
          {'task_id':602,'text':'Add one to an integer.','code':'def wrong(x):\n    return x+2',
           'test_list':['assert wrong(1)==2','assert wrong(2)==3']}]
    (cache/'mbpp.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    (cache/'repository-README.md').write_text('All datasets in this repository are released under the CC BY 4.0 International license.')
    (cache/'sources.json').write_text(json.dumps({'revision':'fixture',
        'files':[{'path':name,'url':'https://example.org/'+name,
                  'sha256':hashlib.sha256((cache/name).read_bytes()).hexdigest()}
                 for name in ['mbpp.jsonl','repository-README.md']]}))
    functions=tmp_path/'functions.jsonl';functions.write_text('')
    tokenizer=Tokenizer(models.BPE(unk_token='<unk>'));tokenizer.pre_tokenizer=pre_tokenizers.ByteLevel()
    tokenizer.train_from_iterator(['Return an ordered pair. def pair(a,b): return (a,b)'],
        trainers.BpeTrainer(vocab_size=512,initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),special_tokens=['<unk>']))
    token_path=tmp_path/'tokenizer.json';tokenizer.save(str(token_path))
    questions=tmp_path/'questions.jsonl';questions.write_text(json.dumps({'id':'reserved','question':'An unrelated reserved knowledge question.'})+'\n')
    output=tmp_path/'prepared'
    manifest=prepare(output,cache,functions,token_path,questions,variants=2)
    trained=[json.loads(line) for line in (output/'train.jsonl').read_text().splitlines()]
    assert manifest['train_examples']==2 and manifest['unique_programs']==1
    assert {row['group_id'] for row in trained}=={'mbpp:601'}
    assert {row['variant_index'] for row in trained}=={0,1}
    assert {row['function'] for row in trained} != {'pair'}
    assert json.loads((output/'reserved-mbpp.jsonl').read_text())['task_id']==12
    assert any(row['source_id']=='mbpp:602' for row in [json.loads(line) for line in (output/'excluded.jsonl').read_text().splitlines()])
