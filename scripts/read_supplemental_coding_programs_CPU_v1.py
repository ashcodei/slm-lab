"""Freshly execute all three matched endpoints as their code frames close.

Retains all48 original programs and actual native action/EOS deliveries per
endpoint. Ordinary Python and fresh positive controls are separate from the
restricted native runtime. No generation, repair or optimization is added.
"""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
RUN=Path('/tmp/slm-supplemental-public-code-full-GPU-v1')
OUT=ROOT/'reports/supplemental-coding-program-readback-CPU-v1'
ENDPOINTS=['baseline','plain_tokens','eos_weight4']
FRAMES=['completion','instruction']
GIB=1024**3


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text())


def live(actor):
    try:s=Path('/proc',str(actor['pid']),'stat').read_text().rsplit(')',1)[1].split()
    except FileNotFoundError:return False
    return s[19]==str(actor['birth']) and s[0] not in ['Z','X']


def write(name,value):
    path=OUT/name;pending=path.with_suffix(path.suffix+'.pending')
    with pending.open('x') as f:
        f.write(json.dumps(value,indent=2,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
    os.replace(pending,path)


def main():
    if OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('Fresh explicitly CPU-only whole-program readback required')
    os.sched_setaffinity(0,{22});os.nice(10)
    from tokenizers import Tokenizer
    from slm_lab.code_tasks import TASKS,request
    from slm_lab.code_quality import execute as native_execute,efficiency
    from slm_lab.decoded_native_code_reexecution_v1 import reconstruct
    from slm_lab.native_code_action_readback_v1 import validate
    from slm_lab.ordinary_python_function_cases_v1 import execute as ordinary_execute,implementation_sources
    from slm_lab.local_source_closure_v2 import python_sources
    run=read(RUN/'plan.json');predecessor=run['owner']
    files={str(p):sha(p) for p in [RUN/'plan.json',*python_sources(ROOT,[Path(__file__).resolve()])]}
    files.update(implementation_sources())
    if (run['full_code_programs_per_endpoint']!=48 or run['code_actions_per_program']!=512
            or run['updates_per_arm']!=230 or len(TASKS)!=12):
        raise ValueError('The original complete three-endpoint/48-program/512-action experiment is required')
    if shutil.disk_usage(ROOT).free<48*GIB+64*1024**2:
        raise ValueError('Complete source/program receipts must fit above the persistent48GiB reserve')
    OUT.mkdir();owner=dict(pid=os.getpid(),birth=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19])
    write('plan.json',dict(at=datetime.now(timezone.utc).isoformat(),owner=owner,exact_original_owner=predecessor,
        source_files_sha256=files,endpoints=ENDPOINTS,programs_per_endpoint=48,original_action_budget=512,
        own_nice=10,CPU_affinity=[22],CPU_threads=1,no_GPU_or_neural_generation_or_model_weights_loaded=True,
        complete_original_broad_evaluations_and_final_benchmarks_not_replaced=True,candidate_promoted=False,goal_achieved=False))
    try:
        references={};ordinary_references={}
        for task in TASKS:
            for language,position in [('Python',4),('JavaScript',5)]:
                result=native_execute(request(task,language,task[position],True))
                if not result['functional_passed'] or not result['efficiency_completed']:
                    raise ValueError('Every full fresh native reference/efficiency control must pass')
                references[language+':'+task[0]]=result
            py=ordinary_execute(task[4],task[0],request(task,'Python',task[4])['cases'])
            if not py['functional_passed'] or not py['all_cases_observed']:
                raise ValueError('Every complete ordinary-Python positive control must pass')
            ordinary_references[task[0]]=py
        write('reference-controls.json',dict(native=references,ordinary_Python=ordinary_references,
            controls_reused_across_all_endpoints=True))
        summaries={};per_frame={}
        for endpoint in ENDPOINTS:
            folder=RUN/endpoint;assessment=folder/'assessment'
            while not all((assessment/('code-'+f+'.json')).exists() for f in FRAMES):
                if not live(predecessor):
                    raise ValueError('The exact original owner ended before every original code frame closed')
                if any(sha(p)!=h for p,h in files.items()):raise ValueError('A bound original/runtime source changed while waiting')
                write('progress.json',dict(phase='wait_exact_original_code_frames',endpoint=endpoint,
                    exact_original_owner_confirmed_live=True,completed_endpoint_summaries=summaries))
                time.sleep(10)
            paths=[folder/'selected-model.pt',folder/'tokenizer.json']
            paths += [assessment/('code-'+f+suffix) for f in FRAMES for suffix in ['.json','-actual-actions.jsonl']]
            if endpoint!='baseline':paths.append(folder/'training-complete.json')
            files.update({str(p):sha(p) for p in paths})
            tokenizer=Tokenizer.from_file(str(folder/'tokenizer.json'))
            if sha(folder/'tokenizer.json')!=run['native']['tokenizer_sha256']:
                raise ValueError('Every endpoint must use the same original own tokenizer')
            endpoint_records=[]
            for frame in FRAMES:
                report=read(assessment/('code-'+frame+'.json'))
                actions=[json.loads(l) for l in (assessment/('code-'+frame+'-actual-actions.jsonl')).read_text().splitlines()]
                if (report['task_count']!=24 or len(report['results'])!=24 or len(actions)!=24
                        or report['generation_budget']!=512 or report['prompt_mode']!=frame
                        or report['inference_device_type']!='cuda' or report['inference_precision']!='bf16'
                        or report['checkpoint_sha256']!=sha(folder/'selected-model.pt')
                        or report['tokenizer_sha256']!=run['native']['tokenizer_sha256']):
                    raise ValueError('Every complete native BF16 report, actual action and selected checkpoint is required')
                for rel,digest in report['evaluation_provenance']['files_sha256'].items():
                    path=ROOT/rel
                    if sha(path)!=digest:raise ValueError('An original producer runtime source changed')
                    files[str(path)]=digest
                observations=[]
                for position,(case,action) in enumerate(zip(report['results'],actions)):
                    task=TASKS[position%12];language=['Python','JavaScript'][position//12]
                    if case['language']!=language or case['task']!=task[0]:
                        raise ValueError('Every original task/language/order must remain unchanged')
                    delivery=validate(case,frame,action,tokenizer,expected_case_position=position)
                    actual_task,code=reconstruct(case,frame)
                    if actual_task!=task:raise ValueError('Exact original public task required')
                    actual=native_execute(request(task,language,code))
                    if any(actual[k]!=case['candidate'][k] for k in ['syntax_valid','functional_passed']):
                        raise ValueError('Fresh native syntax/functional verdict differs; preserve the discrepancy')
                    efficient,measurement=efficiency(actual,references[language+':'+task[0]],task[3])
                    py=ordinary_execute(code,task[0],request(task,language,code)['cases']) if language=='Python' else None
                    record=dict(endpoint=endpoint,frame=frame,language=language,task=task[0],delivery_proof=delivery,
                        actual_actions=action,actual_unchanged_code=code,fresh_native=actual,fresh_ordinary_Python=py,
                        producer_empirical_efficiency=case['efficiency_stress_passed'],fresh_empirical_efficiency=efficient,
                        fresh_efficiency_measurement=measurement,shared_host_timing_not_asymptotic_proof=True)
                    write(endpoint+'-'+frame+'-'+language+'-'+task[0]+'.json',record)
                    observations.append(record);endpoint_records.append(record)
                summary=dict(programs=24,native_syntax=sum(r['fresh_native']['syntax_valid'] for r in observations),
                    native_functional=sum(r['fresh_native']['functional_passed'] for r in observations),
                    ordinary_Python_syntax=sum(r['fresh_ordinary_Python']['syntax_valid'] for r in observations if r['fresh_ordinary_Python'] is not None),
                    ordinary_Python_functional=sum(r['fresh_ordinary_Python']['functional_passed'] for r in observations if r['fresh_ordinary_Python'] is not None),
                    producer_efficiency=sum(r['producer_empirical_efficiency'] for r in observations),
                    fresh_empirical_efficiency=sum(r['fresh_empirical_efficiency'] for r in observations),
                    EOS=sum(r['actual_actions']['actual_stopped_on_EOS'] for r in observations),
                    budget_ends=sum(not r['actual_actions']['actual_stopped_on_EOS'] for r in observations))
                if (report['syntax_pass_rate']!=summary['native_syntax']/24
                        or report['functional_pass_at_1']!=summary['native_functional']/24
                        or report['efficiency_stress_pass_rate']!=summary['producer_efficiency']/24):
                    raise ValueError('All original producer code aggregates must reconstruct')
                per_frame[endpoint+'-'+frame]=summary
            summaries[endpoint]={k:sum(per_frame[endpoint+'-'+f][k] for f in FRAMES) for k in per_frame[endpoint+'-completion']}
            if any(sha(p)!=h for p,h in files.items()):raise ValueError('An original/runtime/model/delivery byte changed')
            write(endpoint+'-complete.json',dict(complete=True,endpoint=endpoint,summary=summaries[endpoint],
                source_files_sha256=files,actual_original_actions_and_unchanged_programs_reconstructed=True,
                fresh_native_reexecutions=48,fresh_ordinary_Python_reexecutions=24,
                every_syntax_functional_verdict_agrees=True,no_neural_likelihood_or_model_inference_replayed=True,
                original_broad_and_final_evaluations_still_required=True,candidate_promoted=False,goal_achieved=False))
            print(json.dumps(dict(endpoint=endpoint,complete=True,summary=summaries[endpoint])),flush=True)
        write('complete.json',dict(complete=True,at=datetime.now(timezone.utc).isoformat(),source_files_sha256=files,
            summary=summaries,per_frame=per_frame,actual_native_reexecutions=144,actual_ordinary_Python_reexecutions=72,
            fresh_native_reference_controls=24,fresh_ordinary_Python_reference_controls=12,
            every_original_action_EOS_program_and_syntax_functional_verdict_rebuilt=True,
            reused_local_development_suite_not_independent_final_benchmarks=True,
            no_new_model_generation_training_or_GPUwork=True,candidate_promoted=False,goal_achieved=False))
    except BaseException as error:
        write('failure.json',dict(error=repr(error),traceback=traceback.format_exc(),partial_outputs_retained=True,
            original_inputs_and_processes_unchanged=True,candidate_promoted=False,goal_achieved=False));raise


if __name__=='__main__':main()
