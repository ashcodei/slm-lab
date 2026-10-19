import copy
import hashlib
import json
from pathlib import Path

import pytest
from tokenizers import Tokenizer

from slm_lab.apps_execution_repair_v1 import mutations,counterexample,repair_example
from slm_lab.apps_published_cases_v2 import contract,execute


def test_exact_unicode_token_edits_and_unchanged_strings_comments():
    source = '# café 0 +\ndef f(x):\n    text="1 + True"\n    return x + 1 if x > 0 else 0\n'
    changes = mutations(source,'original-1')
    assert changes == mutations(source,'original-1') and 1<=len(changes)<=4
    for change in changes:
        patch=change['patch']
        assert source[patch['start']:patch['end']]==patch['old']
        assert change['program']==source[:patch['start']]+patch['new']+source[patch['end']:]
        assert change['program'].startswith('# café 0 +') and '"1 + True"' in change['program']
        assert hashlib.sha256(change['program'].encode()).hexdigest()==change['program_sha256']


def test_actual_complete_counterexample_and_whole_target_masks():
    source='def f(x):\n    return x * 2\n'
    cases=contract(json.dumps(dict(inputs=[[2],[4]],outputs=[4,8],fn_name='f')).encode())
    positive=execute(source,cases)
    change=next(c for c in mutations(source,'source-1') if c['patch']['old']=='*')
    negative=execute(change['program'],cases)
    witness=counterexample(positive,negative,cases)
    assert witness['case_index']==1 and witness['input']==[4] and witness['observed']==6 and witness['original_expected']==8
    assert counterexample(positive,positive,cases) is None
    timeout=execute('def f(x):\n    while True: pass\n',cases,seconds=.3)
    assert counterexample(positive,timeout,cases) is None
    broken=copy.deepcopy(negative);broken['raw_results']['results'].pop()
    with pytest.raises(ValueError,match='Every unchanged original case'):
        counterexample(positive,broken,cases)
    root=Path(__file__).resolve().parents[1]
    tokenizer=Tokenizer.from_file(str(root/'checkpoints/dialogue-retention-v1/systems-variety-code/tokenizer.json'))
    question='Return twice x.'
    original=dict(id='original-1',group_id='group-1',split='train',publisher_split='train',source_problem_id='0001',
        original_solution_index=0,original_question_sha256=hashlib.sha256(question.encode()).hexdigest(),
        original_source_body_sha256=hashlib.sha256(source.encode()).hexdigest(),original_tests_sha256=cases['original_tests_sha256'])
    row,ids,labels=repair_example(original,question,None,source,change,witness,tokenizer)
    assert row['response']==source and not row['human_preferences'] and not row['external_model_teacher_used']
    prefix=[1]+tokenizer.encode('User: '+row['prompt']+'\nAssistant:').ids
    body=tokenizer.encode(source).ids+[2]
    assert ids==prefix+body and labels==[-100]*len(prefix)+body and not row['review_reasons']
    long_question='question '*6000
    long,whole_ids,whole_labels=repair_example({**original,'original_question_sha256':hashlib.sha256(long_question.encode()).hexdigest()},
        long_question,None,source,change,witness,tokenizer)
    assert long['review_reasons']==['complete_repair_context_exceeds4096'] and len(whole_ids)>4096
    assert whole_ids[-len(body):]==body and whole_labels[-len(body):]==body
    with pytest.raises(ValueError,match='publisher-train roles'):
        repair_example(dict(split='validation',publisher_split='train'),None,None,None,None,None,None)
