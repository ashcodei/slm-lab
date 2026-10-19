import json
from pathlib import Path

import pytest

from slm_lab.held_question_registry_v2 import original_question


def test_all_actual_original_held_questions_are_strings_and_nq_metadata_is_not_serialized_as_a_question():
    root=Path(__file__).resolve().parents[1]
    expected={'GSM8K':1319,'NQ-open':3610,'reserved-MBPP':600}
    for benchmark,name in [
        ('GSM8K','data/math-reasoning-v2/test.jsonl'),
        ('NQ-open','data/nq-human-questions-v1/valid.jsonl'),
        ('reserved-MBPP','data/verified-code-expanded-v1/reserved-mbpp.jsonl')]:
        rows=[json.loads(s) for s in (root/name).open()]
        assert len(rows)==expected[benchmark]
        for row in rows:
            text=original_question(benchmark,row)
            assert isinstance(text,str) and text.strip()
            if benchmark=='NQ-open':
                assert text==row['source_original_question']['question']
                assert row['prompt']==text+'\nAnswer briefly.'


def test_invalid_or_implicit_question_schemas_fail_before_expensive_source_scan():
    for row in [{'source_original_question':'flattened text'}, {'source_original_question':{'answer':['x']}},
                {'source_original_question':{'question':{'nested':'x'}}}]:
        with pytest.raises(ValueError):original_question('NQ-open',row)
    with pytest.raises(ValueError):original_question('GSM8K',{'question':{'question':'x'}})
    with pytest.raises(ValueError):original_question('unsupported',{})
