"""Real complete Java/Python union and the actual production reader boundary."""
from copy import deepcopy
from pathlib import Path

import pytest
from tokenizers import Tokenizer,normalizers

from slm_lab.corpus import file_hash
from slm_lab.execution_collective_v2 import feedback_identity
from slm_lab.execution_feedback_union_v1 import ExecutionFeedbackUnion,parents
from slm_lab.execution_preference_v1 import validate
from slm_lab.execution_production_v6 import open_feedback

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/execution-feedback-union-v1'
TOKEN=ROOT/'checkpoints/extended-curriculum-v1/joint/tokenizer.json'


@pytest.fixture(scope='module')
def feedback():
    # This is the same factory actually called by the v6 capacity and training
    # workers, not a test replacement for its admitted-source reader.
    return open_feedback(dict(execution_feedback=str(DATA),tokenizer=str(TOKEN),
        execution_manifest_sha256=file_hash(DATA/'manifest.json'),execution_admission_sha256=file_hash(DATA/'admission.json')))


def test_every_original_parent_view_reaches_the_actual_capacity_production_reader(feedback):
    assert len(feedback.positives)==195 and len(feedback.pairs)==374
    assert feedback.positives==[r for owner in feedback.owners for r in owner.positives]
    assert feedback.pairs==[r for owner in feedback.owners for r in owner.pairs]
    identity=feedback_identity(feedback,file_hash(TOKEN));assert identity['positives']==195 and identity['pairs']==374
    assert {r['language'] for r in feedback.pairs}=={'Java','Python'}
    positive_lengths=[]
    for i in range(195):
        ids,labels=feedback.positive(i);positive_lengths.append(len(ids))
        start=next(j for j,y in enumerate(labels) if y!=-100)
        assert ids[-1]==2 and labels==[-100]*start+ids[start:]
    examples=[feedback.example(i) for i in range(374)]
    assert len(validate(examples,True))==374
    assert max(positive_lengths)==4066
    assert max(len(e['chosen'][0]) for e in examples)==4066
    assert max(len(e['rejected'][0]) for e in examples)==4067


def test_plan_and_original_parent_admission_hashes_cannot_be_replaced(feedback):
    plan=dict(execution_feedback=str(DATA),tokenizer=str(TOKEN),execution_manifest_sha256='0'*64,
        execution_admission_sha256=file_hash(DATA/'admission.json'))
    with pytest.raises(ValueError,match='evidence changed'):open_feedback(plan)
    manifest=deepcopy(feedback.manifest);manifest['parents']['Python']['admission_sha256']='0'*64
    with pytest.raises(ValueError,match='parent bytes'):parents(manifest,feedback.tokenizer,TOKEN)


def test_foreign_tokenizer_semantics_cannot_become_the_original_reference():
    tokenizer=Tokenizer.from_file(str(TOKEN));tokenizer.normalizer=normalizers.Lowercase()
    with pytest.raises(ValueError):ExecutionFeedbackUnion(DATA,tokenizer,TOKEN)


def test_in_memory_role_and_target_changes_reject_before_the_shared_kernel_interface(feedback):
    before=deepcopy(feedback.pairs[0])
    try:
        feedback.pairs[0]['split']='validation'
        with pytest.raises(ValueError,match='train-only'):feedback_identity(feedback,file_hash(TOKEN))
        feedback.pairs[0]=deepcopy(before);feedback.pairs[0]['chosen']+='extra'
        with pytest.raises(ValueError,match='must match'):feedback.example(0)
    finally:feedback.pairs[0]=before
