import hashlib
from pathlib import Path

import numpy as np
import pytest
from tokenizers import Tokenizer

from slm_lab.public_reference_prefix_curriculum_v1 import view


def fixture():
    tokenizer=Tokenizer.from_file(str(Path(__file__).resolve().parents[1]/'data/tokenizer-corpus/tokenizer.json'))
    response='def sum_positive(values):\n    total = 0\n    for value in values:\n        if value > 0:\n            total += value\n    return total\n'
    prefix=[1,*tokenizer.encode('User: Implement sum_positive.\nAssistant:',add_special_tokens=False).ids]
    target=[*tokenizer.encode(response,add_special_tokens=False).ids,2];ids=prefix+target
    row=dict(id='unit-only-not-a-real-public-source',split='train',publisher_split='train',response=response,
        external_model_teacher_used=False,model_generated_examples=False,
        complete_input_ids_sha256=hashlib.sha256(np.asarray(ids,dtype='<u2').tobytes()).hexdigest())
    return tokenizer,row,ids,[-100]*len(prefix)+target


def test_original_complete_program_and_anchor_reconstruct_at_each_level_without_hint_score_promotion():
    tokenizer,row,ids,labels=fixture();views=[view(tokenizer,row,ids,labels,q) for q in [3,2,1,0]]
    assert views[0]['retained_reference_tokens']>0
    assert [v['retained_reference_tokens'] for v in views]==sorted([v['retained_reference_tokens'] for v in views],reverse=True)
    for value in views:
        assert value['native_rollout_prefix_ids']+value['original_missing_suffix_and_EOS_ids']==ids
        assert value['complete_original_anchor_labels']==labels
        assert value['complete_missing_suffix_training_budget']>=len(value['original_missing_suffix_and_EOS_ids'])
        assert not value['admitted_to_RL'] and not value['unassisted_capability_score']
    assert views[-1]['native_rollout_prefix_ids']==views[-1]['original_native_prompt_ids']
    assert views[-1]['exact_unassisted_original_source']


def test_held_or_synthetic_sources_and_tampered_native_targets_are_rejected():
    tokenizer,row,ids,labels=fixture()
    for changes in [dict(split='test'),dict(publisher_split='test'),dict(external_model_teacher_used=True),dict(model_generated_examples=True)]:
        with pytest.raises(ValueError):view(tokenizer,dict(row,**changes),ids,labels,3)
    changed=list(ids);changed[-2]=3
    with pytest.raises(ValueError):view(tokenizer,row,changed,labels,3)
    with pytest.raises(ValueError):view(tokenizer,row,ids,labels[:-1],3)
    with pytest.raises(ValueError):view(tokenizer,row,ids,labels,4)
