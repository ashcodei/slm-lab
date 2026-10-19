import gzip
import json

import numpy as np
import pytest

from slm_lab.public_pretraining_intake_v1 import candidate
from slm_lab.public_billion_candidate_v1 import CandidateWriter,QUOTAS
from test_public_pretraining_intake_v1 import fixture


def entry():
    t,q,s,r=fixture();value,_=candidate(r,s,t,q);return value


def test_actual_whole_native_binary_original_text_and_mask_descriptor_roundtrip(tmp_path):
    value=entry();writer=CandidateWriter(tmp_path/'candidate',{'code:Python':1})
    assert writer.append(value)=='whole_original_document_retained_unadmitted'
    summary=writer.summary();writer.close()
    with gzip.open(tmp_path/'candidate/whole-original-documents.jsonl.gz','rt') as stream:row=json.loads(next(stream))
    native=np.fromfile(tmp_path/'candidate'/row['native_file'],dtype='<u2').tolist()
    assert native==value['whole_native_ids'] and row['native_token_length']==len(native)
    assert row['original_publisher_row']['code']==value['original_public_text']
    assert summary['actual_train_targets']+summary['actual_valid_targets']==len(native)-1
    assert row['optimization_admitted'] is False and row['native_token_offset']==0


def test_exact_body_duplicates_and_corrupt_masks_cannot_enter_candidate(tmp_path):
    value=entry();writer=CandidateWriter(tmp_path/'candidate',{'code:Python':10**9})
    assert writer.append(value)=='whole_original_document_retained_unadmitted'
    assert writer.append(value)=='exact_original_body_duplicate_within_new_candidate'
    bad=dict(value,whole_native_labels=[-100]*len(value['whole_native_ids']))
    with pytest.raises(ValueError,match='exact whole'):writer.append(bad)
    assert sum(v for roles in writer.summary()['original_documents'].values() for v in roles.values())==1
    writer.close()


def test_full_one_billion_recipe_uses_train_targets_and_keeps_all_eight_code_languages():
    assert sum(QUOTAS.values())==10**9
    assert sum(v for k,v in QUOTAS.items() if k.startswith('code:'))==500_000_000
    assert QUOTAS['web']==250_000_000 and QUOTAS['math']==250_000_000
    assert {k.removeprefix('code:') for k in QUOTAS if k.startswith('code:')}=={'Python','C','C++','Java','JavaScript','TypeScript','Go','Rust'}
