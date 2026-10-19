"""Rebuild all96actual native deliveries, source bytes and completed outcomes.

This reader does not perform a new neural generation or rerun programs whose
native and ordinary executions the completed producer already retained.
"""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np
from tokenizers import Tokenizer

from slm_lab.code_quality import instruction_code,request
from slm_lab.code_tasks import TASKS
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.local_source_closure_v2 import python_sources
from slm_lab.source_jsonl_records_v1 import records

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'reports/main-foundation-instruction14336-code-CPU-v2'
OUT=ROOT/'reports/main-matched-code-complete-v1-readback'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists():raise ValueError('Fresh CPU-hidden completed source/outcome readback required')
    complete_path=SOURCE/'complete.json';complete=json.loads(complete_path.read_text())
    if (not complete['all96_original_native_deliveries_and_real_token_actions_retained']
            or not complete['all96_actual_native_programs_freshly_reexecuted']
            or not complete['all48_actual_Python_programs_also_executed_under_secondary_protocol']):
        raise ValueError('Every actual native and secondary delivery/execution must already be complete')
    plan_path=SOURCE/'plan.json';plan=json.loads(plan_path.read_text())
    files={str(p):file_hash(p) for p in [complete_path,plan_path,*python_sources(ROOT,[Path(__file__)])]}
    files.update(complete['source_files_sha256']);guard=FileBindings(files)
    tokenizer_path=ROOT/'data/tokenizer-corpus/tokenizer.json';tokenizer=Tokenizer.from_file(str(tokenizer_path))
    if file_hash(tokenizer_path)!=json.loads((ROOT/'reports/phase-joint-v3/plan.json').read_text())['tokenizer_sha256']:
        raise ValueError('Exact original own native tokenizer required')
    profiles=[];outcomes={};ledger=[]
    for endpoint in ['foundation12853','instruction14336']:
        folder=SOURCE/endpoint;endpoint_receipt=json.loads((folder/'complete.json').read_text())
        outcomes[endpoint]={}
        for frame in ['completion','instruction']:
            receipt=json.loads((folder/(frame+'-complete.json')).read_text());FileBindings(receipt['source_files_sha256']).check(full=True)
            files.update(receipt['source_files_sha256']);files[str(folder/(frame+'-complete.json'))]=file_hash(folder/(frame+'-complete.json'))
            report=json.loads((folder/('code-'+frame+'.json')).read_text())
            actions=list(records(folder/(frame+'-all24-actual-generation-actions.jsonl')))
            native=json.loads((folder/(frame+'-all24-fresh-native-program-reexecutions.json')).read_text())
            ordinary=json.loads((folder/(frame+'-all12-ordinary-Python-program-executions.json')).read_text())
            expected=[(language,t[0]) for language in ['Python','JavaScript'] for t in TASKS]
            if (len(actions)!=24 or len(native)!=24 or len(ordinary)!=12
                    or [(c['language'],c['task']) for c in report['results']]!=expected):
                raise ValueError('Every original task/frame/language/delivery must remain present')
            profiles.append({k:report[k] for k in ['generation_budget','decoding','suite_sha256','tokenizer_sha256','inference_device_type','inference_precision']})
            syntax=functional=efficiency=ordinary_correct=0;by_language={}
            for position,(case,event,actual) in enumerate(zip(report['results'],actions,native)):
                task=next(t for t in TASKS if t[0]==case['task']);name,args,specification=task[:3];language=case['language']
                prefix=(f'def {name}({args}):\n    """{specification}"""\n' if language=='Python' else f'function {name}({args}) {{\n  // {specification}\n')
                if frame=='instruction':
                    signature=f'def {name}({args}):' if language=='Python' else f'function {name}({args})'
                    runtime=('Use built-in Python types; only the math module is available.' if language=='Python' else 'Use JavaScript language built-ins; Node/browser APIs are unavailable.')
                    prompt=f'User: Write a {language} function with signature {signature}. {specification}\n{runtime}\nReturn only the complete function and any necessary helper functions.\nAssistant:'
                else:prompt=prefix
                ids=[1]+tokenizer.encode(prompt).ids;generated=event['actual_generated_token_ids'];raw=tokenizer.decode(generated)
                code=instruction_code(raw) if frame=='instruction' else prefix+raw.split('```')[0]
                if language=='Python':code=re.split(r'\n(?=File:)',code)[0]
                if (event['case_position']!=position or event['original_input_token_ids']!=ids
                        or event['complete_input_ids_sha256']!=hashlib.sha256(np.asarray(ids,dtype='<u2').tobytes()).hexdigest()
                        or event['requested_native_action_budget']!=512 or not 1<=len(generated)<=512
                        or any(type(v) is not int or not 0<=v<32768 for v in generated) or 2 in generated[:-1]
                        or (len(generated)<512 and generated[-1]!=2)
                        or event['actual_native_action_count']!=len(generated) or raw!=case['raw_response'] or code!=case['code']
                        or actual['task']!=name or actual['language']!=language
                        or actual['actual_generated_code_sha256']!=hashlib.sha256(code.encode()).hexdigest()):
                    raise ValueError('Every actual original prompt/action/code byte must reconstruct without repair')
                executed=actual['actual_execution'];payload=request(task,language,code)
                if (executed.get('total_cases')!=len(payload['cases'])
                        or bool(executed.get('syntax_valid'))!=bool(case['candidate'].get('syntax_valid'))
                        or bool(executed.get('functional_passed'))!=bool(case['candidate'].get('functional_passed'))):
                    raise ValueError('Every complete original finite-case denominator and fresh execution verdict must reconstruct')
                syntax+=bool(executed.get('syntax_valid'));functional+=bool(executed.get('functional_passed'));efficiency+=bool(case['efficiency_stress_passed'])
                group=by_language.setdefault(language,dict(tasks=0,syntax=0,functional=0));group['tasks']+=1
                group['syntax']+=bool(executed.get('syntax_valid'));group['functional']+=bool(executed.get('functional_passed'))
                if language=='Python':
                    second=ordinary[position]
                    if second['task']!=name or second['actual_code']!=code:raise ValueError('Every actual secondary program must be the unchanged original generated Python')
                    ordinary_correct+=bool(second['ordinary_execution']['functional_passed'])
                ledger.append(dict(endpoint=endpoint,frame=frame,language=language,task=name,actual_actions=len(generated),
                    actual_code_sha256=hashlib.sha256(code.encode()).hexdigest(),actual_original_finite_cases=len(payload['cases']),
                    syntax=bool(executed.get('syntax_valid')),functional=bool(executed.get('functional_passed')),
                    diagnostic=executed.get('error'),actual_program_not_reexecuted_by_this_reader=True))
            r=receipt['actual_results']
            if (syntax,functional,efficiency,ordinary_correct)!=(r['syntax_valid'],r['native_functionally_correct'],r['producer_empirical_efficiency_passes'],r['ordinary_Python_correct']):
                raise ValueError('Every complete actual frame outcome must reconstruct')
            outcomes[endpoint][frame]=dict(tasks=24,syntax=syntax,functional=functional,producer_efficiency=efficiency,
                ordinary_Python_correct=ordinary_correct,by_language=by_language)
        if endpoint_receipt['exact_completed_main_updates']!=(12853 if endpoint=='foundation12853' else 14336):
            raise ValueError('Exact fixed own endpoint updates required')
    if any(p!=profiles[0] for p in profiles) or len(ledger)!=96 or profiles[0]['generation_budget']!=512:
        raise ValueError('All96original cases must share the unchanged native runtime and complete action budget')
    totals={name:dict(tasks=48,syntax=sum(r['syntax'] for r in frames.values()),functional=sum(r['functional'] for r in frames.values()),
        producer_efficiency=sum(r['producer_efficiency'] for r in frames.values()),ordinary_Python_correct=sum(r['ordinary_Python_correct'] for r in frames.values())) for name,frames in outcomes.items()}
    guard.check(full=True);OUT.mkdir();atomic_json(OUT/'all96-delivery-source-outcomes.json',ledger)
    atomic_json(OUT/'complete.json',dict(at=datetime.now(timezone.utc).isoformat(),complete=True,actual_frame_outcomes=outcomes,
        actual_matched_endpoint_totals=totals,all96actual_inputs_actions_code_and_fresh_native_execution_records_rederived=True,
        all48actual_secondary_Python_source_and_outcomes_rederived=True,
        original_cases_frames_budgets_and_protocol_unchanged=True,source_files_sha256=files,
        no_new_neural_forward_or_program_execution=True,no_asymptotic_efficiency_or_final_independence_claim=True,
        model_updates=0,candidate_promoted=False,goal_achieved=False))
    print(json.dumps(dict(complete=str(OUT/'complete.json'),actual_matched_endpoint_totals=totals)),flush=True)


if __name__=='__main__':main()
