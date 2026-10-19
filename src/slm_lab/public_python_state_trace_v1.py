"""Actual bounded public source-frame traces; no golds or neural teachers."""
import base64
import hashlib
import json
import math
from pathlib import Path
import tempfile

from .corpus import file_hash
from .standalone_program_v1 import python_policy,run_bounded

ROOT=Path(__file__).resolve().parents[2]
WORKER=ROOT/'scripts/public_python_state_trace_worker_v1.py'
VALUES=ROOT/'src/slm_lab/finite_Python_values_v2.py'
FORMAT='own-public-source-frame-state-observations-v1'


def capture(program,public_input,*,maximum_events=256,maximum_trace_bytes=256*1024,seconds=4.):
    if (not isinstance(program,str) or not program or len(program.encode())>32768 or '\0' in program
            or not isinstance(public_input,str) or len(public_input.encode())>4*1024**2
            or type(maximum_events) is not int or not 1<=maximum_events<=4096
            or type(maximum_trace_bytes) is not int or not 1<=maximum_trace_bytes<=1024**2
            or not math.isfinite(seconds) or not 0<seconds<=30):raise ValueError('Whole original source/public input and explicit bounded observer required')
    with tempfile.TemporaryDirectory(prefix='slm-public-state-trace-') as temporary:
        work=Path(temporary);source=work/'program.py';source.write_text(program)
        (work/'observer.py').write_bytes(WORKER.read_bytes());(work/'values.py').write_bytes(VALUES.read_bytes())
        settings=dict(program=str(source),maximum_events=maximum_events,maximum_trace_bytes=maximum_trace_bytes)
        (work/'settings.json').write_text(json.dumps(settings))
        policy=python_policy(work);policy['command'][-1]=str(work/'observer.py');policy['command'].append(str(work/'settings.json'))
        policy['memory_bytes']=512*1024**2;policy['cpu_seconds']=max(1,math.ceil(seconds))
        actual=run_bounded(policy,public_input.encode(),seconds=seconds,output_bytes=4*1024**2,python=True)
    receipt={k:v for k,v in actual.items() if k not in {'stdout','stderr'}}
    receipt.update(program_sha256=hashlib.sha256(program.encode()).hexdigest(),public_input_sha256=hashlib.sha256(public_input.encode()).hexdigest(),
        raw_observation_stdout_b64=base64.b64encode(actual['stdout']).decode(),raw_observer_stderr_b64=base64.b64encode(actual['stderr']).decode(),
        expected_or_private_values_not_supplied_to_worker=True,model_updates=0)
    if actual['status']!='completed':return dict(receipt,status='observer_runtime_qualification',eligible_complete_trace=False)
    try:
        raw=json.loads(actual['stdout'])
        if raw['format']!=FORMAT or type(raw['whole_trace_complete']) is not bool:raise ValueError('Exact observer transport required')
        if raw['whole_trace_complete']:
            validate(raw,program,maximum_events=maximum_events,maximum_trace_bytes=maximum_trace_bytes)
    except (ValueError,TypeError,KeyError,RecursionError):
        return dict(receipt,status='invalid_observer_transport_qualification',eligible_complete_trace=False)
    return dict(receipt,status='complete_public_source_state_trace' if raw['whole_trace_complete'] else 'incomplete_or_bounded_trace_qualification',
        eligible_complete_trace=raw['whole_trace_complete'],actual_observations=raw)


def validate(raw,program,*,maximum_events,maximum_trace_bytes):
    from .finite_Python_values_v2 import decode
    def check_value(value):
        if (isinstance(value,list) and len(value)==2 and value[0]=='opaque_type'
                and isinstance(value[1],str) and value[1]):return
        decode(value,budget=[256])
    events=raw['events'];states={};stack=[];next_frame=0;byte_count=0
    if (raw['qualification'] is not None or not 1<=len(events)<=maximum_events
            or raw['non_dunder_locals_only'] is not True or raw['non_data_objects_are_opaque_types'] is not True
            or raw['source_frame_events_only'] is not True):raise ValueError('A whole declared-protocol trace is required')
    for event in events:
        kind=event['event'];frame=event['frame'];line=event['line']
        module_entry=(kind=='call' and line==0 and event['function']=='<module>' and frame==0)
        if (type(frame) is not int or type(line) is not int or not (module_entry or 1<=line<=len(program.splitlines()))
                or not isinstance(event['function'],str) or kind not in {'call','line','return','exception'}):raise ValueError('Original source event and actual line required')
        if kind=='call':
            if frame!=next_frame:raise ValueError('Contiguous activation IDs required')
            next_frame+=1;stack.append(frame);states[frame]={}
        if not stack or stack[-1]!=frame:raise ValueError('A complete stack-ordered source trace required')
        for name in event['deleted']:
            if name not in states[frame]:raise ValueError('Only an actual previous binding can be deleted')
            del states[frame][name]
        for name,value in event['changes'].items():
            if not isinstance(name,str) or name.startswith('__'):raise ValueError('Explicit observed non-dunder locals required')
            check_value(value)
            if states[frame].get(name)==value:raise ValueError('Only actual state changes belong in delta transport')
            states[frame][name]=value
        byte_count+=len(json.dumps(event,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode())
        if kind=='return':
            if 'return_value' not in event:raise ValueError('Every actual return value or opaque type required')
            check_value(event['return_value'])
            stack.pop();del states[frame]
    if (stack or states or next_frame!=raw['source_frame_activations'] or byte_count>maximum_trace_bytes
            or raw['source_lines_observed']!=sum(r['event']=='line' for r in events)):
        raise ValueError('Whole nontruncated source trace and exact denominators required')
    base64.b64decode(raw['actual_stdout_b64'],validate=True);base64.b64decode(raw['actual_stderr_b64'],validate=True)
    return True


def implementation_sources():
    from .mbpp_original_function_cases_v2 import implementation_sources as sandbox_sources
    return {**sandbox_sources(),**{str(p):file_hash(p) for p in [Path(__file__),WORKER,VALUES]}}
