"""Confined source-frame observer; receives original code and public stdin only."""
import base64
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys

FORMAT='own-public-source-frame-state-observations-v1'


def main():
    settings_path=Path(sys.argv[1]);settings=json.loads(settings_path.read_text())
    source_path=settings['program'];source=Path(source_path).read_text()
    spec=importlib.util.spec_from_file_location('trace_data_values',settings_path.with_name('values.py'))
    values=importlib.util.module_from_spec(spec);spec.loader.exec_module(values)
    events=[];frames={};next_frame=0;serialized_bytes=0;qualification=None
    active=[];maximum_events=settings['maximum_events'];maximum_bytes=settings['maximum_trace_bytes']

    class Bound(Exception):pass

    def value(item):
        try:return values.encode(item,budget=[256])
        except ValueError as error:
            if str(error).startswith('Returned type outside explicit data transport: '):
                return ['opaque_type',type(item).__name__]
            raise Bound('whole_value_exceeds_structural_bound')

    def observe(frame,event,arg):
        nonlocal next_frame,serialized_bytes
        if frame.f_code.co_filename!=source_path:return observe
        if event not in {'call','line','return','exception'}:raise Bound('unexpected_source_frame_event')
        key=id(frame)
        if event=='call':
            if key in frames:raise Bound('reentrant_or_generator_frame_requires_separate_protocol')
            frames[key]=dict(frame=next_frame,previous={});next_frame+=1;active.append(key)
        if key not in frames or not active or active[-1]!=key:raise Bound('non_stack_source_frame_requires_separate_protocol')
        info=frames[key]
        current={name:value(item) for name,item in sorted(frame.f_locals.items()) if not name.startswith('__')}
        row=dict(event=event,line=frame.f_lineno,frame=info['frame'],function=frame.f_code.co_name,
            changes={k:v for k,v in current.items() if info['previous'].get(k)!=v},
            deleted=[k for k in info['previous'] if k not in current])
        if event=='return':row['return_value']=value(arg)
        if event=='exception':row['exception_type']=arg[0].__name__
        size=len(json.dumps(row,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode())
        if len(events)>=maximum_events or serialized_bytes+size>maximum_bytes:raise Bound('whole_trace_exceeds_event_or_byte_bound')
        events.append(row);serialized_bytes+=size;info['previous']=current
        if event=='return':del frames[key];active.pop()
        return observe

    stdout_bytes=io.BytesIO();stderr_bytes=io.BytesIO()
    stdout=io.TextIOWrapper(stdout_bytes,encoding='utf-8');stderr=io.TextIOWrapper(stderr_bytes,encoding='utf-8')
    namespace=dict(__name__='__main__',__file__=source_path)
    sys.argv=[source_path]
    with contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(stderr):
        try:
            compiled=compile(source,source_path,'exec');sys.settrace(observe);exec(compiled,namespace)
        except Bound as error:qualification=str(error)
        except SystemExit as error:
            if error.code not in (None,0):qualification='original_program_nonzero_system_exit'
        except BaseException as error:qualification='original_program_or_observer_error:'+type(error).__name__
        finally:sys.settrace(None)
    stdout.flush();stderr.flush()
    if active or frames:qualification=qualification or 'incomplete_source_frame_trace'
    # A partial trace is retained for diagnosis but cannot become a target.
    result=dict(format=FORMAT,whole_trace_complete=qualification is None,qualification=qualification,events=events,
        actual_stdout_b64=base64.b64encode(stdout_bytes.getvalue()).decode(),actual_stderr_b64=base64.b64encode(stderr_bytes.getvalue()).decode(),
        source_frame_activations=next_frame,source_lines_observed=sum(r['event']=='line' for r in events),
        non_dunder_locals_only=True,non_data_objects_are_opaque_types=True,source_frame_events_only=True)
    print(json.dumps(result,ensure_ascii=False,separators=(',',':'),allow_nan=False),flush=True)


if __name__=='__main__':main()
