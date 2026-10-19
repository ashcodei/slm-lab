"""Bind original local-code prompts, actual generated actions and raw text.

This is an input/action/code-byte readback, not a new neural generation or
certificate of source semantics. No probability or EOS information is
invented for older producers that did not retain actual action journals.
"""
import hashlib
import struct

from .code_tasks import TASKS
from .decoded_native_code_reexecution_v1 import reconstruct


def prompt_for(case,frame):
    if frame not in {'completion','instruction'} or case['language'] not in {'Python','JavaScript'}:
        raise ValueError('Exact original native frame and language required')
    task=next((t for t in TASKS if t[0]==case['task']),None)
    if task is None or case['specification']!=task[2]:raise ValueError('Every original complete task specification required')
    name,args,specification=task[:3];language=case['language']
    prefix=(f'def {name}({args}):\n    """{specification}"""\n' if language=='Python'
        else f'function {name}({args}) {{\n  // {specification}\n')
    if frame=='completion':return prefix
    signature=f'def {name}({args}):' if language=='Python' else f'function {name}({args})'
    runtime=('Use built-in Python types; only the math module is available.' if language=='Python'
        else 'Use JavaScript language built-ins; Node/browser APIs are unavailable.')
    return f'User: Write a {language} function with signature {signature}. {specification}\n{runtime}\nReturn only the complete function and any necessary helper functions.\nAssistant:'


def validate(case,frame,event,tokenizer,*,expected_case_position):
    if tokenizer.truncation is not None or tokenizer.padding is not None:
        raise ValueError('Original native tokenizer must have no hidden clipping or padding')
    if [tokenizer.token_to_id(k) for k in ['<pad>','<bos>','<eos>']]!=[0,1,2] or tokenizer.get_vocab_size()!=32768:
        raise ValueError('Exact original own vocabulary and native boundary convention required')
    expected=[1]+tokenizer.encode(prompt_for(case,frame)).ids
    if (event['case_position']!=expected_case_position or type(event['case_position']) is not int
            or event['original_input_token_ids']!=expected
            or event['requested_native_action_budget']!=512 or type(event['requested_native_action_budget']) is not int
            or event['complete_input_ids_sha256']!=hashlib.sha256(struct.pack('<'+'H'*len(expected),*expected)).hexdigest()):
        raise ValueError('Exact original case position, complete prompt ids/hash and512action budget required')
    actions=event['actual_generated_token_ids']
    if (not isinstance(actions,list) or not 1<=len(actions)<=512
            or any(type(value) is not int or not 0<=value<32768 for value in actions)
            or event['actual_native_action_count']!=len(actions) or type(event['actual_native_action_count']) is not int
            or 2 in actions[:-1] or type(event['actual_stopped_on_EOS']) is not bool
            or event['actual_stopped_on_EOS']!=(actions[-1]==2)
            or len(actions)<512 and actions[-1]!=2
            or event['greedy_native_output_tensor_returned_unchanged'] is not True
            or event['selected_action_log_probabilities_not_retained'] is not True):
        raise ValueError('Every actual unchanged bounded native action and genuine first-EOS/budget stop must reconstruct')
    if tokenizer.decode(actions)!=case['raw_response']:
        raise ValueError('Every actual original generated token must decode to the unchanged full raw response')
    _,code=reconstruct(case,frame)
    return dict(original_case_position=expected_case_position,language=case['language'],task=case['task'],frame=frame,
        complete_original_input_tokens=len(expected),complete_original_input_sha256=event['complete_input_ids_sha256'],
        actual_native_action_count=len(actions),actual_stopped_on_EOS=actions[-1]==2,
        actual_generated_ids_u16le_sha256=hashlib.sha256(struct.pack('<'+'H'*len(actions),*actions)).hexdigest(),
        raw_response_sha256=hashlib.sha256(case['raw_response'].encode()).hexdigest(),
        actual_unchanged_program_sha256=hashlib.sha256(code.encode()).hexdigest(),
        complete_prompt_tokens_actions_decoded_response_and_program_bytes_rebuilt=True,
        no_generation_repair_or_additional_neural_forward=True,action_log_probabilities_not_retained_or_certified=True)
