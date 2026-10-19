import copy
import hashlib
from pathlib import Path
import struct

import pytest
from tokenizers import Tokenizer

from slm_lab.code_quality import instruction_code
from slm_lab.code_tasks import TASKS
from slm_lab.native_code_action_readback_v1 import prompt_for,validate


def fixture(language='Python',frame='instruction'):
    tokenizer=Tokenizer.from_file(str(Path(__file__).resolve().parents[1]/'data/tokenizer-corpus/tokenizer.json'))
    task=TASKS[0]
    case=dict(language=language,task=task[0],specification=task[2])
    response=(task[4] if language=='Python' else task[5]) if frame=='instruction' else ('    return 0\n' if language=='Python' else '  return 0;\n}\n')
    actions=tokenizer.encode(response).ids+[2]
    case['raw_response']=tokenizer.decode(actions)
    case['code']=instruction_code(case['raw_response']) if frame=='instruction' else prompt_for(case,frame)+case['raw_response'].split('```')[0]
    ids=[1]+tokenizer.encode(prompt_for(case,frame)).ids
    event=dict(case_position=0,original_input_token_ids=ids,actual_generated_token_ids=actions,
        complete_input_ids_sha256=hashlib.sha256(struct.pack('<'+'H'*len(ids),*ids)).hexdigest(),
        requested_native_action_budget=512,actual_native_action_count=len(actions),actual_stopped_on_EOS=True,
        greedy_native_output_tensor_returned_unchanged=True,selected_action_log_probabilities_not_retained=True)
    return tokenizer,case,event


@pytest.mark.parametrize('language',['Python','JavaScript'])
@pytest.mark.parametrize('frame',['completion','instruction'])
def test_actual_own_BPE_full_prompts_actions_and_code_reconstruct(language,frame):
    tokenizer,case,event=fixture(language,frame)
    proof=validate(case,frame,event,tokenizer,expected_case_position=0)
    assert proof['actual_stopped_on_EOS'] and proof['actual_native_action_count']==len(event['actual_generated_token_ids'])
    assert proof['actual_unchanged_program_sha256']==hashlib.sha256(case['code'].encode()).hexdigest()


@pytest.mark.parametrize('fault',['changed_prompt','changed_action','changed_code','missing_EOS','middle_EOS','wrong_count','wrong_case_position'])
def test_tampered_original_action_or_program_fields_are_rejected(fault):
    tokenizer,case,event=fixture()
    if fault=='changed_prompt':event['original_input_token_ids'][1]=4
    elif fault=='changed_action':event['actual_generated_token_ids'][0]=4
    elif fault=='changed_code':case['code']+='\n# hidden repair'
    elif fault=='missing_EOS':
        event['actual_generated_token_ids'].pop();event['actual_native_action_count']-=1;event['actual_stopped_on_EOS']=False
    elif fault=='middle_EOS':event['actual_generated_token_ids'][0]=2
    elif fault=='wrong_count':event['actual_native_action_count']+=1
    else:event['case_position']=1
    with pytest.raises(ValueError):validate(case,'instruction',event,tokenizer,expected_case_position=0)


def test_empty_EOS_delivery_is_retained_without_inventing_correctness():
    tokenizer,case,event=fixture()
    event.update(actual_generated_token_ids=[2],actual_native_action_count=1)
    case.update(raw_response='',code='')
    proof=validate(case,'instruction',event,tokenizer,expected_case_position=0)
    assert proof['actual_native_action_count']==1 and proof['actual_stopped_on_EOS']


def test_full512action_budget_stop_is_not_falsely_labelled_EOS():
    tokenizer,case,event=fixture()
    action=tokenizer.encode(' ').ids[0]
    event.update(actual_generated_token_ids=[action]*512,actual_native_action_count=512,actual_stopped_on_EOS=False)
    case['raw_response']=tokenizer.decode(event['actual_generated_token_ids']);case['code']=instruction_code(case['raw_response'])
    proof=validate(case,'instruction',event,tokenizer,expected_case_position=0)
    assert proof['actual_native_action_count']==512 and not proof['actual_stopped_on_EOS']
