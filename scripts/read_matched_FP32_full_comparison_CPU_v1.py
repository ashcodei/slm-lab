"""Independently rebuild/reexecute the whole closed96-program FP32 comparison.

Wait for the exact recovery owner without neural work. Then reconstruct every
prompt/action/delivery, all native and ordinary Python verdicts, and fresh
native/ordinary reference controls. No model or optimizer is loaded.
"""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path('/tmp/slm-broad-code-step512-code-evaluation-CPU-v2')
OUT=ROOT/'reports/matched-FP32-full-comparison-readback-v1'
SNAPSHOT=Path('/tmp/slm-broad-code-step512-snapshot-CPU-v1')


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def live(owner):
    try:fields=(Path('/proc')/str(owner['pid'])/'stat').read_text().rsplit(')',1)[1].split()
    except FileNotFoundError:return False
    return fields[19]==str(owner['birth']) and fields[0] not in ('Z','X')


def prompt_for(task,language,frame):
    if language not in ('Python','JavaScript') or frame not in ('completion','instruction'):
        raise ValueError('Only the original language/frame contracts are supported')
    if frame=='completion':
        return (f'def {task[0]}({task[1]}):\n    """{task[2]}"""\n' if language=='Python'
            else f'function {task[0]}({task[1]}) {{\n  // {task[2]}\n')
    signature=f'def {task[0]}({task[1]}):' if language=='Python' else f'function {task[0]}({task[1]})'
    runtime=('Use built-in Python types; only the math module is available.' if language=='Python'
        else 'Use JavaScript language built-ins; Node/browser APIs are unavailable.')
    return f'User: Write a {language} function with signature {signature}. {task[2]}\n{runtime}\nReturn only the complete function and any necessary helper functions.\nAssistant:'


def main():
    if (OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.nice(0)!=0 or os.sched_getscheduler(0)!=os.SCHED_OTHER
            or any(os.environ.get(k)!='1' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'])):
        raise ValueError('Fresh normal one-thread CUDA-hidden complete native readback required')
    source_plan=json.loads((SOURCE/'owner-plan.json').read_text());predecessor=source_plan['owner']
    files={str(SOURCE/'owner-plan.json'):sha(SOURCE/'owner-plan.json'),str(Path(__file__).resolve()):sha(__file__)}
    OUT.mkdir();owner=dict(pid=os.getpid(),birth=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19])
    def emit(name,value):
        with (OUT/name).open('x') as stream:stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')
    emit('owner-plan.json',dict(owner=owner,at=datetime.now(timezone.utc).isoformat(),exact_recovery_owner=predecessor,
        source_files_sha256=files,original_total_programs=96,original_precision='CPU FP32',generation_budget=512,
        new_neural_models=0,new_optimizer_updates=0,original_workers_unchanged=True,goal_achieved=False))
    try:
        while live(predecessor):
            if any(sha(path)!=wanted for path,wanted in files.items()):raise ValueError('Exact waiting owner/source bytes changed')
            print(json.dumps(dict(phase='waiting_exact_closed96program_recovery',predecessor=predecessor)),flush=True)
            time.sleep(15)
        if (SOURCE/'failure.json').exists() or not (SOURCE/'complete.json').is_file():
            raise ValueError('Exact recovery ended without the full original96-program success; preserve all partial evidence')
        complete=json.loads((SOURCE/'complete.json').read_text())
        if (complete.get('complete') is not True or complete.get('owner')!=predecessor
                or complete.get('actual_delivered_programs')!=96 or complete.get('retained_original_neural_generations_replayed')!=84
                or complete.get('new_missing_suffix_neural_generations')!=12):
            raise ValueError('All84original and12new complete source deliveries required')
        from tokenizers import Tokenizer
        from slm_lab.code_tasks import TASKS,request
        from slm_lab.code_quality import execute as native_execute,efficiency
        from slm_lab.decoded_native_code_reexecution_v1 import reconstruct
        from slm_lab.ordinary_python_function_cases_v1 import execute as ordinary_execute,implementation_sources
        from slm_lab.local_source_closure_v2 import python_sources
        reports={(endpoint,frame):json.loads((SOURCE/(endpoint+'-'+frame+'.json')).read_text())
            for endpoint in ['initial','step512'] for frame in ['completion','instruction']}
        ordinary={(endpoint,frame):json.loads((SOURCE/(endpoint+'-'+frame+'-ordinary-Python.json')).read_text()) for endpoint,frame in reports}
        paths=[SOURCE/'complete.json',SOURCE/'actual-native-actions.jsonl',SOURCE/'interim-source-state-readback.json',SNAPSHOT/'tokenizer.json']
        paths += [SOURCE/(endpoint+'-'+frame+suffix+'.json') for endpoint,frame in reports for suffix in ['','-ordinary-Python']]
        paths += python_sources(ROOT,[Path(__file__).resolve()])
        files.update({str(path):sha(path) for path in paths});files.update(implementation_sources())
        tokenizer=Tokenizer.from_file(str(SNAPSHOT/'tokenizer.json'));vocab=tokenizer.get_vocab_size()
        actions=[json.loads(line) for line in (SOURCE/'actual-native-actions.jsonl').read_text().splitlines()]
        assert len(actions)==96 and sum(r['new_neural_generation'] for r in actions)==12
        refs={};ordinary_refs={}
        for task in TASKS:
            for language,index in [('Python',4),('JavaScript',5)]:
                ref=native_execute(request(task,language,task[index],True))
                assert ref['functional_passed'] and ref['efficiency_completed']
                refs[language+':'+task[0]]=ref
            ref=ordinary_execute(task[4],task[0],request(task,'Python',task[4])['cases'])
            assert ref['functional_passed'] and ref['all_cases_observed'];ordinary_refs[task[0]]=ref
        per_frame={};all_records=[]
        for endpoint_index,endpoint in enumerate(['initial','step512']):
            for frame_index,frame in enumerate(['completion','instruction']):
                report=reports[(endpoint,frame)];observations=[]
                assert report['task_count']==len(report['results'])==24 and report['generation_budget']==512
                assert report['decoding']=='greedy, cached FP32' and report['prompt_mode']==frame
                assert report['inference_precision']=='fp32' and report['inference_device_type']=='cpu'
                assert report['tokenizer_sha256']==sha(SNAPSHOT/'tokenizer.json')
                assert report['suite_sha256']==reports[('initial','completion')]['suite_sha256']
                saved_ordinary={r['task']:r for r in ordinary[(endpoint,frame)]['results']}
                for i,case in enumerate(report['results']):
                    task=TASKS[i%12];language=['Python','JavaScript'][i//12];action=actions[endpoint_index*48+frame_index*24+i]
                    assert tuple(action[k] for k in ['endpoint','frame','case','language','task'])==(endpoint,frame,i+1,language,task[0])
                    assert case['language']==language and case['task']==task[0]
                    prompt=prompt_for(task,language,frame)
                    assert action['complete_prompt_ids']==[1]+tokenizer.encode(prompt).ids
                    ids=action['actual_response_ids'];assert 0<len(ids)<=512 and all(type(t) is int and 0<=t<vocab for t in ids)
                    assert 2 not in ids[:-1] and action['stopped_on_EOS'] is (ids[-1]==2)
                    assert action['ended_on_budget'] is (len(ids)==512 and ids[-1]!=2)
                    assert tokenizer.decode(ids)==case['raw_response']
                    actual_task,code=reconstruct(case,frame);assert actual_task==task
                    actual=native_execute(request(task,language,code))
                    assert all(actual[key]==case['candidate'][key] for key in ['syntax_valid','functional_passed'])
                    fresh_efficiency,measure=efficiency(actual,refs[language+':'+task[0]],task[3])
                    py=None
                    if language=='Python':
                        py=ordinary_execute(code,task[0],request(task,language,code)['cases']);saved=saved_ordinary[task[0]]
                        assert saved['actual_unchanged_code_sha256']==hashlib.sha256(code.encode()).hexdigest()
                        assert all(py[key]==saved['execution'][key] for key in ['syntax_valid','functional_passed','original_cases_sha256','total_cases'])
                    row=dict(endpoint=endpoint,frame=frame,language=language,task=task[0],actual_actions=action,
                        actual_unchanged_code=code,fresh_native=actual,fresh_ordinary_Python=py,
                        producer_efficiency=case['efficiency_stress_passed'],fresh_empirical_efficiency=fresh_efficiency,
                        fresh_efficiency_measurement=measure,shared_host_timing_not_asymptotic_proof=True)
                    emit(endpoint+'-'+frame+'-'+language+'-'+task[0]+'.json',row);observations.append(row);all_records.append(row)
                summary=dict(programs=24,native_syntax=sum(r['fresh_native']['syntax_valid'] for r in observations),
                    native_functional=sum(r['fresh_native']['functional_passed'] for r in observations),
                    ordinary_Python_syntax=sum(r['fresh_ordinary_Python']['syntax_valid'] for r in observations if r['fresh_ordinary_Python'] is not None),
                    ordinary_Python_functional=sum(r['fresh_ordinary_Python']['functional_passed'] for r in observations if r['fresh_ordinary_Python'] is not None),
                    producer_efficiency=sum(r['producer_efficiency'] for r in observations),fresh_empirical_efficiency=sum(r['fresh_empirical_efficiency'] for r in observations),
                    EOS=sum(r['actual_actions']['stopped_on_EOS'] for r in observations),budget_ends=sum(r['actual_actions']['ended_on_budget'] for r in observations))
                assert report['syntax_pass_rate']==summary['native_syntax']/24 and report['functional_pass_at_1']==summary['native_functional']/24
                assert report['efficiency_stress_pass_rate']==summary['producer_efficiency']/24
                per_frame[endpoint+'-'+frame]=summary
                print(json.dumps(dict(endpoint=endpoint,frame=frame,summary=summary)),flush=True)
        aggregate={endpoint:{key:sum(per_frame[endpoint+'-'+frame][key] for frame in ['completion','instruction']) for key in per_frame[endpoint+'-completion']}
            for endpoint in ['initial','step512']}
        for endpoint,summary in aggregate.items():
            producer=complete['summary'][endpoint]
            assert (summary['programs'],summary['native_syntax'],summary['native_functional'],summary['producer_efficiency'],summary['ordinary_Python_functional'])==(
                producer['programs'],producer['syntax'],producer['functional'],producer['empirical_efficiency'],producer['ordinary_Python'])
        assert all(sha(path)==wanted for path,wanted in files.items())
        emit('reference-controls.json',dict(native=refs,ordinary_Python=ordinary_refs))
        results={str(path):sha(path) for path in OUT.glob('*.json') if path.name!='owner-plan.json'}
        emit('complete.json',dict(complete=True,owner=owner,at=datetime.now(timezone.utc).isoformat(),source_files_sha256=files,
            closed_result_files_sha256=results,per_frame=per_frame,summary=aggregate,actual_native_reexecutions=96,
            actual_ordinary_Python_reexecutions=48,fresh_native_reference_controls=24,fresh_ordinary_Python_reference_controls=12,
            fresh_ordinary_Python_reference_cases=sum(r['total_cases'] for r in ordinary_refs.values()),
            every_original_prompt_action_and_unchanged_code_delivery_rebuilt=True,every_syntax_functional_verdict_agrees=True,
            fresh_empirical_efficiency_timing_recorded_separately_from_producer=True,
            full96local_comparison_not_independent_final_benchmark=True,no_new_generation_model_load_training_or_GPUwork=True,
            checkpoint512_not_final3721=True,all_original_broad_final_requirements_preserved=True,candidate_promoted=False,goal_achieved=False))
    except BaseException:
        emit('failure.json',dict(owner=owner,traceback=traceback.format_exc(),partial_outputs_retained=True,goal_achieved=False));raise


if __name__=='__main__':main()
