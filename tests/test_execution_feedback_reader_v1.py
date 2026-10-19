"""Exercise every actual admitted complete publisher pair and source view."""
import json
from pathlib import Path

import pytest
from tokenizers import Tokenizer,normalizers

from slm_lab.execution_feedback_reader_v1 import ExecutionFeedback
from slm_lab.execution_preference_v1 import validate
from slm_lab.execution_java_training_suites_v2 import TrainingSuites

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/execution-code-feedback-v2'
TOKEN=ROOT/'checkpoints/extended-curriculum-v1/joint/tokenizer.json'


def reader():
    return ExecutionFeedback(DATA,Tokenizer.from_file(str(TOKEN)),TOKEN)


def test_every_real_admitted_pair_preserves_complete_question_programs_and_full_EOS_masks():
    corpus=reader()
    assert len(corpus.pairs)==210 and len({r['group_id'] for r in corpus.pairs})==84
    for i,row in enumerate(corpus.pairs):
        example=corpus.example(i);counts=validate([example],True)[0]
        assert counts==[row['chosen_target_tokens'],row['rejected_target_tokens']]
        for name in ['chosen','rejected']:
            ids,labels=example[name]
            start=next(j for j,v in enumerate(labels) if v!=-100)
            assert labels==[-100]*start+ids[start:] and ids[0]==1 and ids[-1]==2 and len(ids)<=4096
    corpus.guard.check(full=True)


def test_every109_original_complete_positive_is_available_after_actual_full_test_execution():
    corpus=reader();assert len(corpus.positives)==109
    for i,row in enumerate(corpus.positives):
        ids,labels=corpus.positive(i)
        assert len(ids)==row['context_tokens'] and sum(v!=-100 for v in labels)==row['target_tokens']
    assert corpus.admission['complete_original_registered_cases']==1559


def test_substituted_in_memory_source_or_invalid_index_rejects():
    corpus=reader()
    with pytest.raises(IndexError):corpus.example(-1)
    with pytest.raises(IndexError):corpus.positive(True)
    corpus.pairs[0]['rejected']+=' '
    with pytest.raises(ValueError,match='body must match'):corpus.example(0)


def test_tokenizer_object_with_changed_normalization_rejects_before_any_training():
    tokenizer=Tokenizer.from_file(str(TOKEN));tokenizer.normalizer=normalizers.Prepend('fixture-change:')
    with pytest.raises(ValueError,match='own BPE bytes, normalizer'):
        ExecutionFeedback(DATA,tokenizer,TOKEN)


def test_reviewed_helper_qualification_cannot_be_applied_to_a_held_source_role():
    suites=object.__new__(TrainingSuites)
    with pytest.raises(ValueError,match='only to its original training source'):
        suites.layout(dict(id='exercism:java:tree-building',split='validation',group_id='exercism:tree-building'))
