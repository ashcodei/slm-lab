"""Whole cached-case observations, with a new namespace for each input.

One confined interpreter per source/seed amortizes sandbox startup costs.
This is a separately named local context, not full publisher-environment
reproduction, a general determinism proof, or an admitted RL reward oracle.
"""
import hashlib
import math
from pathlib import Path
import tempfile

from .codeio_published_cases_v1 import MAX_PAYLOAD,MAX_PROGRAM,contract,equal_json,strict_loads
from .corpus import file_hash
from .standalone_program_v1 import python_policy,run_bounded

ROOT=Path(__file__).resolve().parents[2]
HARNESS=ROOT/'scripts/codeio_whole_case_harness_v2.py'


def execute(cases,*,seed,seconds=20,program=None):
    if type(seed) is not int or seed not in (0,1) or not 0<seconds<=120:
        raise ValueError('Explicit seed and complete-source wall budget required')
    validated=contract(dict(refcode=cases['program'],funcname=cases['function'],ios=cases['cases']))
    if validated!=cases:raise ValueError('Every original source and cached case must remain exact')
    code=cases['program'] if program is None else program
    if not isinstance(code,str) or not code or len(code.encode())>MAX_PROGRAM:
        raise ValueError('A complete bounded program is required')
    import json
    with tempfile.TemporaryDirectory(prefix='slm-CodeIO-whole-cases-v2-') as temporary:
        work=Path(temporary);(work/'program.py').write_text(code)
        (work/'harness.py').write_bytes(HARNESS.read_bytes())
        (work/'settings.json').write_text(json.dumps(dict(program=str(work/'program.py'),function=cases['function'],
            inputs=[c['input'] for c in cases['cases']],seed=seed),ensure_ascii=False,allow_nan=False))
        policy=python_policy(work);policy['command'][-1]=str(work/'harness.py')
        policy['command'].append(str(work/'settings.json'));policy['memory_bytes']=768*1024**2
        policy['cpu_seconds']=max(1,math.ceil(seconds))
        actual=run_bounded(policy,b'',seconds=seconds,output_bytes=MAX_PAYLOAD,python=True)
    execution={k:v for k,v in actual.items() if k not in ['stdout','stderr']}
    execution.update(stdout_sha256=hashlib.sha256(actual['stdout']).hexdigest(),stderr=actual['stderr'].decode(errors='replace'))
    reason=None;records=[]
    if actual['status']!='completed':reason='source_runtime_or_resource_qualification'
    else:
        try:
            parsed=strict_loads(actual['stdout'])
            if (not isinstance(parsed,dict) or set(parsed)!={'format','records'}
                    or parsed['format']!='own-CodeIO-whole-return-values-v2'
                    or not isinstance(parsed['records'],list) or len(parsed['records'])!=cases['total_cases']):
                raise ValueError('A complete original returned-case population is required')
            for position,row in enumerate(parsed['records']):
                if (not isinstance(row,dict) or set(row)!={'case_position','value','seconds'}
                        or type(row['case_position']) is not int or row['case_position']!=position
                        or type(row['seconds']) not in (int,float) or not math.isfinite(row['seconds']) or row['seconds']<0):
                    raise ValueError('Complete ordered returned values and finite timing required')
            records=parsed['records']
        except (ValueError,TypeError,KeyError):reason='returned_value_schema_qualification'
    whole=reason is None
    flags=[equal_json(r['value'],c['output']) for r,c in zip(records,cases['cases'])] if whole else None
    return dict(seed=seed,original_program_sha256=hashlib.sha256(code.encode()).hexdigest(),
        original_cases_sha256=cases['original_cases_sha256'],total_original_cases=cases['total_cases'],
        execution=execution,records=records,all_original_cases_observed=whole,
        all_original_cached_values_match=whole and all(flags),case_passes=flags,
        status=reason or ('all_original_cached_values_match' if all(flags) else 'completed_cached_value_mismatch'),
        fresh_module_namespace_per_case=True,fresh_interpreter_per_case=False,keyword_invocation_preserved=True,
        expected_outputs_not_sent_to_children=True,publisher_full_runtime_reproduced=False,
        independent_semantic_correctness_certified=False,training_or_RL_admitted=False)


def implementation_sources():
    from .codeio_published_cases_v1 import implementation_sources as original_sources
    return {**original_sources(),str(Path(__file__).resolve()):file_hash(__file__),str(HARNESS):file_hash(HARNESS)}
