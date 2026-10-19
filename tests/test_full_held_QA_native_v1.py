import copy
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer

from slm_lab.dense_foundation_inference_v1 import DenseNativeView
from slm_lab.full_held_QA_native_v1 import FullHeldQA,evaluate_case,native_prompt,score,verify_record,require_shards
from slm_lab.model import ModelConfig
from slm_lab.own_dense_growth_v1 import OwnDenseAssistant,new_backbone


def example(benchmark,original):
    return dict(audit_id='unit-only:'+benchmark,benchmark=benchmark,original=original,split='held',optimization_forbidden=True,
        previously_sampled_for_local_evaluation=False,inspected_exact_match_scopes=[],potential_exact_question_exposure=False,
        fresh_MMLU_not_previously_sampled_and_no_exact_match=benchmark=='MMLU-Pro')


@pytest.fixture(scope='module')
def model_and_tokenizer():
    torch.set_num_threads(1)
    tokenizer=Tokenizer.from_file(str(Path(__file__).resolve().parents[1]/'data/tokenizer-corpus/tokenizer.json'))
    config=ModelConfig(vocab_size=32768,hidden_size=32,num_layers=1,num_heads=4,num_kv_heads=2,intermediate_size=64,max_text_length=1024,
        vision_hidden_size=32,vision_intermediate_size=64,vision_layers=1,vision_heads=4,image_size=32,patch_size=16)
    assistant=OwnDenseAssistant(new_backbone(config,964517),pointing_rank=16,reward_hidden=8,router_hidden=8,detector_hidden=8,word_temporal=8,word_expansion=1).eval()
    return DenseNativeView(assistant).eval(),tokenizer


def test_original_gold_never_enters_prompt_or_changes_real_native_generation(model_and_tokenizer):
    model,tokenizer=model_and_tokenizer
    first=example('MMLU-Pro',dict(id='unit',domain='math',question='Which number is even?',options=['Four','Five'],answer_index=0))
    second=copy.deepcopy(first);second['original']['answer_index']=1
    a=evaluate_case(model,tokenizer,first);b=evaluate_case(model,tokenizer,second)
    assert a['original_native_prompt_ids']==b['original_native_prompt_ids']
    assert a['actual_generated_ids']==b['actual_generated_ids'] and a['choice_scores']==b['choice_scores']
    assert a['candidate_token_evidence']==b['candidate_token_evidence']
    verify_record(tokenizer,first,a)
    bad=copy.deepcopy(a);bad['actual_generated_response']='A fabricated answer'
    with pytest.raises(ValueError):verify_record(tokenizer,first,bad)
    bad=copy.deepcopy(a);bad['choice_scores']['direct_letter'][0]+=1
    with pytest.raises(ValueError):verify_record(tokenizer,first,bad)
    assert all(p.grad is None for p in model.parameters())


def test_open_answers_use_numeric_or_original_alias_gold_without_candidate_hinting():
    math=example('GSM8K',dict(id='unit',question='A unit fixture asks for an amount.',answer_numerator=3,answer_denominator=2))
    assert score(math,'Computation.\n#### 1.5')['correct']
    assert not score(math,'#### 2')['correct'] and not score(math,'1.5')['answer_parseable']
    nq=example('NQ-open',dict(id='unit',source_original_question={'question':'What is the unit answer?'},
        prompt='What is the unit answer?\nAnswer briefly.',reference_answers=['Straße']))
    assert score(nq,' STRASSE ')['correct'] and not score(nq,'')['correct']
    assert 'Straße' not in native_prompt(nq)


def test_incomplete_audit_wrong_source_role_and_changed_preflight_reject_before_generation(tmp_path,model_and_tokenizer):
    with pytest.raises(ValueError,match='six-scope'):FullHeldQA(tmp_path)
    model,tokenizer=model_and_tokenizer
    case=example('MMLU-Pro',dict(id='unit',domain='math',question='Which?',options=['One','Two'],answer_index=0))
    changed=copy.deepcopy(case);changed['split']='train'
    with pytest.raises(ValueError):native_prompt(changed)
    with pytest.raises(ValueError,match='preflighted'):
        evaluate_case(model,tokenizer,case,expected_input=dict(native_input_ids_sha256='0'*64,prompt_tokens=1,fixed_generation_budget=32))


def test_missing_duplicate_or_different_checkpoint_shards_cannot_be_reconciled():
    rows=[dict(rank=i,world=4,checkpoint_sha256='a'*64) for i in range(4)]
    assert require_shards(rows)==(4,'a'*64)
    for invalid in [[],rows[:3],[rows[0],rows[0],rows[2],rows[3]],
                    [*rows[:3],dict(rows[3],checkpoint_sha256='b'*64)]]:
        with pytest.raises(ValueError):require_shards(invalid)
