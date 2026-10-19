"""Actual pinned generators, independent answers and non-executing rewards."""
import copy
from pathlib import Path
import random
import sys

import pytest

from slm_lab.reasoning_source_loader_v1 import load,verify
from slm_lab.reasoning_source_oracles_v1 import CONFIGS,expected,expression,prompt,strict_reward

SOURCE=Path('data/reasoning-gym-pinned-source-v1')


def test_exact_pinned_sources_limited_registry_namespace_cleanup_and_replay():
    before=verify(SOURCE);rng=random.getstate()
    with load(SOURCE) as factory:
        assert set(factory.DATASETS)==set(CONFIGS)
        first=factory.create_dataset('count_bits',seed=981213,size=8,**CONFIGS['count_bits'])
        values=[first[i] for i in range(8)]
    assert not any(n=='reasoning_gym' or n.startswith('reasoning_gym.') for n in sys.modules)
    with load(SOURCE) as factory:
        second=factory.create_dataset('count_bits',seed=981213,size=8,**CONFIGS['count_bits'])
        assert values==[second[i] for i in range(8)]
    assert verify(SOURCE)==before and random.getstate()==rng
    assert not list((SOURCE/'upstream').rglob('__pycache__'))


@pytest.mark.parametrize('task',list(CONFIGS))
def test_real_question_inputs_and_original_answers_match_independent_algorithms(task):
    with load(SOURCE) as factory:
        data=factory.create_dataset(task,seed=981213,size=16,**CONFIGS[task])
        for i in range(16):
            entry=data[i]
            assert expected(entry)==entry['answer'],(task,i,entry)
            assert strict_reward(entry['answer'],entry)==1.
            assert prompt(entry).startswith(entry['question'])
            assert strict_reward(entry['answer']+' unrelated answer',entry)==0.


def test_fractional_upstream_credit_is_not_mistaken_for_strict_correctness_and_model_text_is_never_executed(tmp_path):
    with load(SOURCE) as factory:
        data=factory.create_dataset('count_bits',seed=981213,size=1,**CONFIGS['count_bits']);entry=data[0]
        wrong='1'+entry['answer']
        assert 0<data.score_answer(wrong,entry)<1 and strict_reward(wrong,entry)==0.
        matrix=factory.create_dataset('binary_matrix',seed=981213,size=1,**CONFIGS['binary_matrix'])[0]
        sentinel=tmp_path/'unexecuted-model-text'
        text="__import__('pathlib').Path("+repr(str(sentinel))+").touch() or [[0]]"
        assert strict_reward(text,matrix)==0. and not sentinel.exists()


def test_question_metadata_disagreement_and_corrupt_public_target_never_get_admitted():
    with load(SOURCE) as factory:
        entry=factory.create_dataset('count_bits',seed=981213,size=1,**CONFIGS['count_bits'])[0]
        wrong=copy.deepcopy(entry);wrong['metadata']['number']+=1
        with pytest.raises(ValueError):expected(wrong)
        wrong=copy.deepcopy(entry);wrong['answer']='999999'
        with pytest.raises(ValueError):strict_reward('999999',wrong)


def test_bounded_literal_interpreter_rejects_calls_names_and_unbounded_shifts():
    assert expression('((0x20 - 0x30) >> 1) + 4')==-4
    for value in ['__import__("os")','x + 2','1 << 1000000000','2 ** 1000']:
        with pytest.raises(ValueError):expression(value)
