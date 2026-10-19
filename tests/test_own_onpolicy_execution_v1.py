import copy
import json
from pathlib import Path

import pytest
from tokenizers import Tokenizer

from slm_lab.apps_published_cases_v2 import contract, execute
from slm_lab.execution_preference_v1 import validate
from slm_lab.own_onpolicy_execution_v1 import example


def test_actual_native_wrong_value_execution_admits_whole_original_and_own_EOS_responses():
    root = Path(__file__).resolve().parents[1]
    tokenizer = Tokenizer.from_file(str(root / 'checkpoints/extended-curriculum-v1/joint/tokenizer.json'))
    program = 'def f(x):\n    return x * 2\n'; wrong = 'def f(x):\n    return x + 2\n'
    cases = contract(json.dumps(dict(inputs=[[2], [4]], outputs=[4, 8], fn_name='f')).encode())
    prompt = 'Write a Python function f that doubles x.'
    prefix = [1, *tokenizer.encode('User: ' + prompt + '\nAssistant:').ids]
    source = dict(id='own-train-1', split='train', publisher_split='train', prompt=prompt, prefix_ids=prefix,
        original_problem='0001', original_human_program=program, target_tokens=len(tokenizer.encode(program).ids)+1, cases=cases)
    ids = [*tokenizer.encode(wrong).ids, 2]
    generation = dict(id=source['id'], split='train', publisher_split='train', strength=0., prefix_ids=prefix,
        response_ids=ids, raw_response=tokenizer.decode(ids, skip_special_tokens=True), code=wrong.strip(), stopped_on_all_EOS=True)
    positive = execute(program, cases); negative = execute(generation['code'], cases)
    pair, reason = example(source, generation, positive, negative, tokenizer, policy_sha256='a'*64)
    assert reason is None and pair['witness']['case_index'] == 1
    assert pair['rejected'] == [prefix+ids, [-100]*len(prefix)+ids]
    assert pair['chosen'][0][-source['target_tokens']:] == tokenizer.encode(program).ids+[2]
    assert validate([pair], True) == [[source['target_tokens'], len(ids)]]
    assert not pair['human_preferences'] and not pair['genuine_human_RLHF']
    incomplete = copy.deepcopy(generation); incomplete['stopped_on_all_EOS'] = False
    assert example(source, incomplete, positive, negative, tokenizer, policy_sha256='a'*64)[1] == 'own_response_not_complete_single_EOS_program'
    broken = copy.deepcopy(generation); broken['prefix_ids'] = [1, 4]
    with pytest.raises(ValueError, match='source identity/prefix'):
        example(source, broken, positive, negative, tokenizer, policy_sha256='a'*64)
    qualified = dict(negative, status='runtime_or_resource_qualification', all_cases_observed=False)
    assert example(source, generation, positive, qualified, tokenizer, policy_sha256='a'*64)[0] is None
    corrupt = copy.deepcopy(negative); corrupt['raw_results']['results'][1]['value'] = 8
    with pytest.raises(ValueError, match='raw source comparisons'):
        example(source, generation, positive, corrupt, tokenizer, policy_sha256='a'*64)


def test_held_source_rejection_precedes_tokenization_execution_or_program_access():
    with pytest.raises(ValueError, match='publisher-train'):
        example(dict(split='validation', publisher_split='train'), None, None, None, None, policy_sha256=None)
