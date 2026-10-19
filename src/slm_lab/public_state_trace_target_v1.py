"""Lossless compact source-event targets for the existing own vocabulary."""
import base64
import hashlib
import json

from .code_contests_curriculum_candidates_v1 import native_sequence

INSTRUCTION=('Predict this Python 3 program\'s CPython source-frame trace and exact standard output for the given standard input. '
    'Return JSON with trace and stdout. Trace events are arrays: '
    '[call,frame,line,function,changes], [line,frame,line,changes,deleted], '
    '[return,frame,line,changes,deleted,value], [exception,frame,line,changes,deleted,type]. '
    'The event names are strings. Line events occur BEFORE that line executes; module entry is line0. '
    'Frame IDs count CPython call segments, including generator resumes; return includes yield/unwind events. '
    'Changes are non-dunder local bindings since that frame\'s last event. Values use tagged data '
    '[int,decimal_string], [str,text], [none], [bool,value], [list,values], [tuple,values], '
    '[dict,key_value_pairs], and analogous type tags; non-data objects use [opaque_type,type_name].')


def compact(events):
    result=[]
    for row in events:
        event=row['event'];base=[event,row['frame'],row['line']]
        if event=='call':
            if row['deleted']:raise ValueError('A fresh call segment has no prior deletions')
            result.append(base+[row['function'],row['changes']])
        else:
            value=base+[row['changes'],row['deleted']]
            if event=='return':value.append(row['return_value'])
            elif event=='exception':value.append(row['exception_type'])
            elif event!='line':raise ValueError('Explicit CPython source event required')
            result.append(value)
    return result


def expand(rows):
    names={};result=[]
    for row in rows:
        event,frame,line=row[:3]
        if event=='call':
            names[frame]=row[3];changes=row[4];deleted=[]
        else:changes,deleted=row[3:5]
        value=dict(event=event,line=line,frame=frame,function=names[frame],changes=changes,deleted=deleted)
        if event=='return':value['return_value']=row[5]
        elif event=='exception':value['exception_type']=row[5]
        result.append(value)
    return result


def target(program,public_input,observations,tokenizer):
    if observations.get('whole_trace_complete') is not True or observations.get('qualification') is not None:
        raise ValueError('No truncated or failed trace may become a supervised target')
    if '```' in program or '```' in public_input:return None,'whole_program_or_input_conflicts_with_prompt_fence'
    trace=compact(observations['events'])
    if expand(trace)!=observations['events']:raise ValueError('Every original observed state/event must round-trip losslessly')
    stdout=base64.b64decode(observations['actual_stdout_b64'],validate=True)
    try:output=stdout.decode('utf-8',errors='strict')
    except UnicodeDecodeError:return None,'whole_observed_stdout_not_UTF8'
    prompt=INSTRUCTION+'\n\nProgram:\n```python\n'+program+'\n```\n\nStandard input:\n```text\n'+public_input+'\n```'
    response=json.dumps(dict(trace=trace,stdout=output),ensure_ascii=False,separators=(',',':'),allow_nan=False)
    sequence,reason=native_sequence(tokenizer,prompt,response)
    if sequence is None:return None,reason
    ids,labels=sequence
    return dict(prompt=prompt,response=response,ids=ids,labels=labels,source_events=len(trace),
        actual_stdout_sha256=hashlib.sha256(stdout).hexdigest(),all_original_observed_events_and_state_values_preserved=True),None
