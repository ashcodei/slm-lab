"""Fresh finite-case executions of unchanged decoded native code deliveries.

Original GPU reports lack token-action journals. Reconstructing code from their
stored raw decoded response therefore does not certify original token IDs or
EOS stopping. It does certify that the reexecuted source is exactly the source
that the producer reported, without a program repair.
"""
import hashlib
import re

from .code_tasks import TASKS,request
from .code_quality import instruction_code,execute as native_execute
from .ordinary_python_function_cases_v1 import execute as ordinary_execute


def reconstruct(case,frame):
    if frame not in {'completion','instruction'} or case['language'] not in {'Python','JavaScript'}:
        raise ValueError('Exact original local code frame and language required')
    task=next(t for t in TASKS if t[0]==case['task'])
    name,args,specification=task[:3]
    if case.get('specification')!=specification:raise ValueError('Every original task specification required')
    prefix=(f'def {name}({args}):\n    """{specification}"""\n' if case['language']=='Python'
        else f'function {name}({args}) {{\n  // {specification}\n')
    code=instruction_code(case['raw_response']) if frame=='instruction' else prefix+case['raw_response'].split('```')[0]
    if case['language']=='Python':code=re.split(r'\n(?=File:)',code)[0]
    if case['code']!=code:raise ValueError('Original decoded response must reconstruct the identical code bytes without repair')
    return task,code


def replay(case,frame):
    task,code=reconstruct(case,frame);payload=request(task,case['language'],code)
    actual=native_execute(payload)
    if (bool(actual.get('syntax_valid'))!=bool(case['candidate'].get('syntax_valid'))
            or bool(actual.get('functional_passed'))!=bool(case['candidate'].get('functional_passed'))):
        raise ValueError('Fresh exact original native syntax/functional verdict differs; preserve and investigate the discrepancy')
    ordinary=ordinary_execute(code,task[0],payload['cases']) if case['language']=='Python' else None
    return dict(language=case['language'],task=task[0],frame=frame,actual_unchanged_code=code,
        actual_code_sha256=hashlib.sha256(code.encode()).hexdigest(),actual_original_finite_cases=len(payload['cases']),
        fresh_native_execution=actual,fresh_ordinary_Python_execution=ordinary,
        producer_empirical_efficiency_pass=case['efficiency_stress_passed'],
        empirical_efficiency_not_recompared_to_new_reference=True,original_generated_token_IDs_or_EOS_stop_not_retained_by_producer=True,
        no_program_repair_or_new_neural_generation=True)
