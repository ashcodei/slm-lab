import copy
import hashlib
import json

import numpy as np
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from slm_lab.apps_verified_generation_v1 import generation_example


def fixture():
    tokenizer = Tokenizer(WordLevel({'<pad>': 0, '<bos>': 1, '<eos>': 2, '<unk>': 3,
        'def': 4, 'increment': 5, 'return': 6, 'x': 7, '1': 8, '+': 9}, unk_token='<unk>'))
    tokenizer.pre_tokenizer = Whitespace()
    question = 'Return one plus the integer argument.'
    program = 'def increment(x):\n    return x + 1\n'
    tests = json.dumps(dict(inputs=[[1], [9]], outputs=[2, 10], fn_name='increment')).encode()
    prompt = ('Write a Python solution to the following problem. Use the original required function or class interface.'
              ' Return only the complete code.\n\n' + question)
    prefix = [1, *tokenizer.encode('User: ' + prompt + '\nAssistant:').ids]
    body = [*tokenizer.encode(program).ids, 2]
    source = dict(id='apps:publisher-train:0001:solution:0', split='train', publisher_split='train',
        prompt=prompt, response=program, original_source_body_sha256=hashlib.sha256(program.encode()).hexdigest(),
        original_question_sha256=hashlib.sha256(question.encode()).hexdigest(),
        original_tests_sha256=hashlib.sha256(tests).hexdigest(), target_tokens=len(body),
        complete_input_ids_sha256=hashlib.sha256(np.asarray(prefix + body, dtype='<u2').tobytes()).hexdigest())
    observation = dict(id=source['id'], admitted_role='train', publisher_role='train',
        status='all_finite_published_cases_pass', all_cases_observed=True, all_finite_published_cases_pass=True,
        syntax_valid=True, mode='function', total_cases=2, passed_cases=2,
        original_tests_sha256=source['original_tests_sha256'], original_program_sha256=source['original_source_body_sha256'],
        raw_results=dict(results=[dict(value=2), dict(value=10)]), case_passes=[True, True])
    return source, question, None, program, tests, observation, tokenizer


def test_complete_generation_preserves_native_prompt_human_target_and_all_masks():
    args = fixture(); row, ids, labels = generation_example(*args)
    source = args[0]
    assert row['response'] == source['response'] and row['prompt'] == source['prompt']
    assert row['complete_original_cases'] == 2 and row['execution_verified']
    assert not row['genuine_RLHF_labels'] and not row['algorithmic_correctness_or_efficiency_proven']
    assert ids[0] == 1 and ids[-1] == 2
    start = len(ids) - source['target_tokens']
    assert labels == [-100] * start + ids[start:]


def test_held_roles_and_corrupt_raw_results_sources_or_complete_tokens_are_rejected():
    for role_key in ['split', 'publisher_split']:
        source = {role_key: 'validation'}
        with pytest.raises(ValueError, match='publisher-train'):
            generation_example(source, None, None, None, None, None, None)
    for mutate, message in [
        (lambda a: a[5]['raw_results']['results'][1].update(value=-10), 'raw result'),
        (lambda a: a[5].update(all_cases_observed=False), 'complete passing'),
        (lambda a: a.__setitem__(3, a[3] + '# changed\n'), 'Unchanged'),
        (lambda a: a[0].update(complete_input_ids_sha256='0' * 64), 'complete token'),
        (lambda a: a[0].update(prompt='fake prompt'), 'native generation prompt')]:
        args = list(fixture()); args[0] = copy.deepcopy(args[0]); args[5] = copy.deepcopy(args[5])
        mutate(args)
        with pytest.raises(ValueError, match=message):
            generation_example(*args)
