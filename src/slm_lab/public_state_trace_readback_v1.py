"""Reconstruct compact targets directly from raw observations, without formatter."""
import base64
import hashlib
import json

from .public_state_trace_target_v1 import INSTRUCTION


def rebuild(program,public_input,observation,tokenizer):
    if (observation.get('eligible_complete_trace') is not True
            or observation.get('program_sha256')!=hashlib.sha256(program.encode()).hexdigest()
            or observation.get('public_input_sha256')!=hashlib.sha256(public_input.encode()).hexdigest()
            or observation.get('expected_or_private_values_not_supplied_to_worker') is not True
            or observation.get('returncode')!=0):raise ValueError('Exact completed source/public-input observation required')
    raw=json.loads(base64.b64decode(observation['raw_observation_stdout_b64'],validate=True))
    if (raw!=observation['actual_observations'] or raw['whole_trace_complete'] is not True or raw['qualification'] is not None):
        raise ValueError('Complete raw worker transport must match the retained decoded states')
    trace=[];stack=[];names={};next_id=0
    for e in raw['events']:
        kind,frame,line=e['event'],e['frame'],e['line']
        if kind=='call':
            if frame!=next_id or e['deleted']:raise ValueError('Actual contiguous CPython call segments required')
            next_id+=1;stack.append(frame);names[frame]=e['function'];row=[kind,frame,line,e['function'],e['changes']]
        else:
            if not stack or stack[-1]!=frame or e['function']!=names[frame]:raise ValueError('Exact original complete source-frame ordering required')
            row=[kind,frame,line,e['changes'],e['deleted']]
            if kind=='return':row.append(e['return_value']);stack.pop()
            elif kind=='exception':row.append(e['exception_type'])
            elif kind!='line':raise ValueError('Every explicit source event required')
        trace.append(row)
    if stack or next_id!=raw['source_frame_activations']:raise ValueError('Complete source trace required')
    stdout=base64.b64decode(raw['actual_stdout_b64'],validate=True).decode('utf-8',errors='strict')
    response=json.dumps({'trace':trace,'stdout':stdout},ensure_ascii=False,separators=(',',':'),allow_nan=False)
    if '```' in program or '```' in public_input:return None
    prompt=INSTRUCTION+'\n\nProgram:\n```python\n'+program+'\n```\n\nStandard input:\n```text\n'+public_input+'\n```'
    prefix=[1]+tokenizer.encode('User: '+prompt+'\nAssistant:').ids;targets=tokenizer.encode(response).ids
    ids=prefix+targets+[2]
    if not targets or len(ids)>4096 or any(v<4 or v>=32768 for v in prefix[1:]+targets):return None
    return dict(prompt=prompt,response=response,ids=ids,labels=[-100]*len(prefix)+targets+[2],
        events=len(trace),actual_stdout=stdout,actual_raw_observations=raw)
