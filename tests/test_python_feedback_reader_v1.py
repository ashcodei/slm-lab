"""Actual independently admitted Python views through the complete DPO interface."""
from copy import deepcopy
from pathlib import Path

import pytest
from tokenizers import Tokenizer,normalizers

from slm_lab.corpus import file_hash
from slm_lab.execution_collective_v2 import feedback_identity
from slm_lab.execution_preference_v1 import validate
from slm_lab.python_feedback_reader_v1 import PythonExecutionFeedback

ROOT=Path(__file__).resolve().parents[1]
TOKEN=ROOT/'checkpoints/extended-curriculum-v1/joint/tokenizer.json'
DATA=ROOT/'data/python-execution-feedback-v1'


@pytest.fixture(scope='module')
def feedback():return PythonExecutionFeedback(DATA,Tokenizer.from_file(str(TOKEN)),TOKEN)


def test_every_actual_complete_publisher_positive_and_pair_fits_own_BPE_and_training_contract(feedback):
    identity=feedback_identity(feedback,file_hash(TOKEN))
    assert identity['positives']==82 and identity['pairs']==153
    assert feedback.admission['original_sources_accounted']==86
    assert feedback.admission['original_publisher_cases_passed']==1236
    for i in range(82):
        ids,labels=feedback.positive(i);start=next(j for j,y in enumerate(labels) if y!=-100)
        assert ids[0]==1 and ids[-1]==2 and len(ids)<=4096
        assert labels==[-100]*start+ids[start:]
    pairs=[feedback.example(i) for i in range(153)]
    assert len(validate(pairs,True))==153
    assert all(p['human_preferences'] is False and p['finite_execution_verified'] is True for p in pairs)


def test_foreign_tokenizer_normalizer_cannot_replace_original_own_encoding():
    tokenizer=Tokenizer.from_file(str(TOKEN));tokenizer.normalizer=normalizers.Lowercase()
    with pytest.raises(ValueError,match='Exact original own BPE'):
        PythonExecutionFeedback(DATA,tokenizer,TOKEN)


def test_in_memory_held_role_or_source_body_cannot_enter_the_shared_training_interface(feedback):
    original=deepcopy(feedback.pairs[0])
    try:
        feedback.pairs[0]['split']='validation'
        with pytest.raises(ValueError,match='train-only'):feedback_identity(feedback,file_hash(TOKEN))
        feedback.pairs[0]=deepcopy(original);feedback.pairs[0]['rejected']+='extra'
        with pytest.raises(ValueError,match='must match'):feedback.example(0)
    finally:feedback.pairs[0]=original
