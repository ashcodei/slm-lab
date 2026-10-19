import copy
import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer

from slm_lab.MBPP_all_native_evaluation_v1 import evaluate_case,grade,summary,verify_record
from slm_lab.MBPP_original_population_v2 import BUDGET,prompt
from slm_lab.deterministic_positions_v1 import install
from slm_lab.model import ModelConfig
from slm_lab.own_dense_growth_v1 import OwnDenseAssistant,new_backbone


def fixture():
    root=Path(__file__).resolve().parents[1];tokenizer=Tokenizer.from_file(str(root/'data/tokenizer-corpus/tokenizer.json'))
    row=dict(task_id=11,text='Return one more than n.',code='def answer(n):\n    return n+1\n',test_setup_code='',
        test_list=['assert answer(2) == 3','assert answer(-1) == 0'],challenge_test_list=['assert answer(7) == 8'])
    ids=[1]+tokenizer.encode(prompt(row)).ids
    expected=dict(native_input_ids=ids,native_input_ids_sha256=hashlib.sha256(np.asarray(ids,dtype='<u2').tobytes()).hexdigest(),
        generation_budget=BUDGET,original_reference_full_case_pass=True)
    return tokenizer,row,expected


def test_actual_small_neural_full512_greedy_delivery_preserved_and_empty_code_remains_failed_problem():
    torch.set_num_threads(1);tokenizer,row,expected=fixture()
    c=ModelConfig(vocab_size=tokenizer.get_vocab_size(),hidden_size=32,intermediate_size=64,num_layers=1,num_heads=4,num_kv_heads=2,
        max_text_length=4096,vision_hidden_size=8,vision_intermediate_size=16,vision_layers=1,vision_heads=2,
        image_size=16,patch_size=4,pool_size=2)
    model=install(OwnDenseAssistant(new_backbone(c,973417),reward_hidden=8,word_temporal=4,pointing_rank=8,router_hidden=8,detector_hidden=4)).eval()
    with torch.no_grad():
        for p in model.parameters():p.zero_()
    signature=[(id(p),p._version,p.data_ptr()) for p in model.parameters()]
    report=evaluate_case(model,tokenizer,row,expected)
    assert report['actual_generated_ids']==[0]*512 and report['actual_generated_response']==''
    assert not report['execution']['functional_passed'] and report['execution']['original_cases']==3
    verify_record(tokenizer,row,expected,report)
    assert signature==[(id(p),p._version,p.data_ptr()) for p in model.parameters()]
    assert summary([report])['publisher_test']['problems']==1


def test_actual_program_values_rederive_all_cases_and_changed_values_or_deliveries_reject():
    tokenizer,row,expected=fixture();code=row['code'].strip();actions=tokenizer.encode(code).ids+[2]
    # Synthetic delivery of an actually executed reference, not an NN score.
    record=dict(original_task_id=11,publisher_role='publisher_test',original_native_prompt=prompt(row),
        original_native_prompt_ids=expected['native_input_ids'],original_native_prompt_ids_sha256=expected['native_input_ids_sha256'],
        generation_budget=512,actual_generated_ids=actions,actual_generated_response=code,actual_generated_code=code,
        genuine_generated_EOS=True,original_reference_full_case_pass=True,optimization_forbidden=True,model_updates=0,
        execution=grade(code,row))
    assert record['execution']['functional_passed'];verify_record(tokenizer,row,expected,record)
    bad=copy.deepcopy(record);bad['execution']['raw_observations']['results'][0]['value']=['int','999']
    with pytest.raises(ValueError,match='verdict'):verify_record(tokenizer,row,expected,bad)
    bad=copy.deepcopy(record);bad['actual_generated_ids']=[2]
    with pytest.raises(ValueError,match='native action'):verify_record(tokenizer,row,expected,bad)
