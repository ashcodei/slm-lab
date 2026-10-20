"""Fresh-process keyword-call checks of whole original CodeI/O cached cases.

Stdlib-only OS confinement is reused unchanged. All expected values stay in
the parent. Runtime/schema limitations abstain; two observed seed contexts
are not a proof of purity, mathematical correctness or general determinism.
"""
import hashlib
import json
import math
from pathlib import Path
import tempfile
import time

from .corpus import file_hash
from .standalone_program_v1 import python_policy,run_bounded

ROOT=Path(__file__).resolve().parents[2]
HARNESS=ROOT/'scripts/codeio_return_value_harness_v1.py'
MAX_PAYLOAD=4*1024**2
MAX_PROGRAM=512*1024


def strict_loads(raw):
    def invalid(value):raise ValueError('Nonfinite original JSON requires qualification')
    def pairs(items):
        result={}
        for key,value in items:
            if key in result:raise ValueError('Ambiguous duplicate original JSON key requires qualification')
            result[key]=value
        return result
    value=json.loads(raw,parse_constant=invalid,object_pairs_hook=pairs)
    json.dumps(value,allow_nan=False)
    return value


def contract(row):
    code=row.get('refcode');name=row.get('funcname');cases=row.get('ios')
    if not isinstance(code,str) or not code or len(code.encode())>MAX_PROGRAM or '\0' in code:
        raise ValueError('Whole original bounded Python reference code required')
    if not isinstance(name,str) or not name.isidentifier() or name.startswith('__'):
        raise ValueError('Original explicit Python function entrypoint required')
    if not isinstance(cases,list) or not 1<=len(cases)<=1000:
        raise ValueError('Every nonempty complete original cached-case population required')
    if any(not isinstance(c,dict) or set(c)!={'input','output'} or not isinstance(c['input'],dict)
            or any(not isinstance(k,str) for k in c['input']) for c in cases):
        raise ValueError('Every original keyword dictionary and output schema is required')
    raw=json.dumps(cases,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()
    if len(raw)>MAX_PAYLOAD:raise ValueError('The complete cached-case payload exceeds the declared bound')
    return dict(program=code,function=name,cases=cases,total_cases=len(cases),
        original_program_sha256=hashlib.sha256(code.encode()).hexdigest(),original_cases_sha256=hashlib.sha256(raw).hexdigest())


def equal_json(a,b):
    # JSON booleans must never alias 0/1. JSON numeric values compare exactly,
    # without tolerance; singleton wrappers and strings are never substituted.
    if type(a) is bool or type(b) is bool:return type(a) is type(b) and a==b
    if type(a) in (int,float) and type(b) in (int,float):
        return all(type(v) is int or math.isfinite(v) for v in (a,b)) and a==b
    if type(a) is not type(b):return False
    if isinstance(a,list):return len(a)==len(b) and all(equal_json(x,y) for x,y in zip(a,b))
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal_json(a[k],b[k]) for k in a)
    return a==b


def execute(cases,*,seed,seconds=20,program=None):
    if type(seed) is not int or seed not in (0,1) or not 0<seconds<=120:
        raise ValueError('An explicit declared seed context and whole-program wall budget are required')
    validated=contract(dict(refcode=cases['program'],funcname=cases['function'],ios=cases['cases']))
    if validated!=cases:raise ValueError('Every original complete case/source identity must remain exact')
    code=cases['program'] if program is None else program
    if not isinstance(code,str) or not code or len(code.encode())>MAX_PROGRAM:raise ValueError('A complete bounded explicit program is required')
    started=time.monotonic();records=[];reason=None
    with tempfile.TemporaryDirectory(prefix='slm-CodeIO-original-cases-v1-') as temporary:
        work=Path(temporary);(work/'program.py').write_text(code);(work/'harness.py').write_bytes(HARNESS.read_bytes())
        for position,case in enumerate(cases['cases']):
            remaining=seconds-(time.monotonic()-started)
            if remaining<=0:reason='whole_source_wall_budget_qualification';break
            (work/'settings.json').write_text(json.dumps(dict(program=str(work/'program.py'),function=cases['function'],
                kwargs=case['input'],seed=seed,case_position=position),ensure_ascii=False,allow_nan=False))
            policy=python_policy(work);policy['command'][-1]=str(work/'harness.py');policy['command'].append(str(work/'settings.json'))
            policy['memory_bytes']=768*1024**2;policy['cpu_seconds']=max(1,math.ceil(remaining))
            actual=run_bounded(policy,b'',seconds=remaining,output_bytes=MAX_PAYLOAD,python=True)
            execution={k:v for k,v in actual.items() if k not in ['stdout','stderr']}
            execution.update(stdout_sha256=hashlib.sha256(actual['stdout']).hexdigest(),stderr=actual['stderr'].decode(errors='replace'))
            result=None
            if actual['status']=='completed':
                try:
                    result=strict_loads(actual['stdout'])
                    if (set(result)!={'format','case_position','value','seconds'} or result['format']!='own-CodeIO-single-return-value-v1'
                            or result['case_position']!=position or type(result['seconds']) not in (int,float)
                            or not math.isfinite(result['seconds']) or result['seconds']<0):raise ValueError('Invalid complete return record')
                except (ValueError,TypeError,KeyError):reason='returned_value_schema_qualification'
            else:reason='source_runtime_or_resource_qualification'
            records.append(dict(original_case_position=position,execution=execution,returned=result))
            if reason:break
    whole=reason is None and len(records)==cases['total_cases']
    flags=[equal_json(r['returned']['value'],c['output']) for r,c in zip(records,cases['cases'])] if whole else None
    return dict(seed=seed,original_program_sha256=hashlib.sha256(code.encode()).hexdigest(),original_cases_sha256=cases['original_cases_sha256'],
        total_original_cases=cases['total_cases'],all_original_cases_observed=whole,all_original_cached_values_match=whole and all(flags),
        status=reason or ('all_original_cached_values_match' if all(flags) else 'completed_cached_value_mismatch'),
        case_passes=flags,records=records,fresh_process_per_case=True,keyword_invocation_preserved=True,
        expected_outputs_not_sent_to_children=True,publisher_full_runtime_reproduced=False,
        independent_semantic_correctness_certified=False,training_or_RL_admitted=False)


def implementation_sources():
    return {str(p):file_hash(p) for p in [Path(__file__).resolve(),HARNESS,ROOT/'src/slm_lab/standalone_program_v1.py',
        ROOT/'src/slm_lab/native_code.py',ROOT/'scripts/python_program_sandbox_v1.py',ROOT/'scripts/native_sandbox.py']}
