"""Matched full local coding assessment of the first code/state continuation.

Fixes the complete own starting model and step2048 before seeing the latter
code outputs. Every original local task/frame/budget is retained. The ordinary
Python protocol is secondary; neither protocol is an independent final score.
"""
from datetime import datetime,timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.local_source_closure_v2 import python_sources
from slm_lab.source_jsonl_records_v1 import records

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/code-first-step2048-native-code-CPU-v1'
SOURCE=ROOT/'reports/code-first-continuation-v3'
CHECKPOINTS=ROOT/'checkpoints/code-first-continuation-v3'


def alive(owner):
    path=Path('/proc')/str(owner['pid'])/'stat'
    if not path.exists():return False
    fields=path.read_text().split(') ',1)[1].split()
    return fields[19]==owner['ticks'] and fields[0] not in {'Z','X'}


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists():raise ValueError('Fresh CPU-hidden matched interim assessment required')
    import torch
    from tokenizers import Tokenizer
    from slm_lab.code_tasks import TASKS
    from slm_lab.code_quality import request,execute as native_execute
    from slm_lab.evaluation_provenance import capture
    from slm_lab.fast_joint_inference_v2 import load
    from slm_lab.inference_rotary_cache_v1 import inference_cache
    from slm_lab.native_code_shards_v1 import evaluate_shard,reconcile
    from slm_lab.native_greedy_observer_v1 import observe,signature
    from slm_lab.ordinary_python_function_cases_v1 import execute as ordinary_execute,implementation_sources
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    source_plan_path=SOURCE/'plan.json';source_plan=json.loads(source_plan_path.read_text())
    checkpoints=[dict(name='initial',arm='initial',updates=0,
        checkpoint=str(CHECKPOINTS/'initial/model.pt'),expected_SHA256=file_hash(CHECKPOINTS/'initial/model.pt')),
        dict(name='step2048',arm='step-2048',updates=2048,
             checkpoint=str(CHECKPOINTS/'step-2048/model.pt'))]
    tests=ROOT/'reports/native-greedy-observer-v1-final-tests.log'
    if '2 passed' not in tests.read_text():raise ValueError('Actual small neural observation transparency and failure restoration tests required')
    files={str(p):file_hash(p) for p in [source_plan_path,tests,
        ROOT/'tests/test_native_greedy_observer_v1.py',*python_sources(ROOT,[Path(__file__),ROOT/'tests/test_native_greedy_observer_v1.py'])]}
    files.update({str(ROOT/p):h for p,h in capture()['files_sha256'].items()})
    files.update(implementation_sources())
    guard=FileBindings(files);OUT.mkdir();owner=dict(pid=os.getpid(),ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19])
    atomic_json(OUT/'plan.json',dict(at=datetime.now(timezone.utc).isoformat(),owner=owner,exact_original_code_first_owner=source_plan['owner'],
        source_files_sha256=files,fixed_endpoints=checkpoints,original_native_tasks_each_frame=24,
        original_native_frames=['completion','instruction'],native_endpoints=2,total_native_deliveries=96,
        original_action_budget=512,inference_precision='CPU_FP32',CPU_threads=2,
        all_actual_input_and_generated_token_IDs_retained=True,action_log_probabilities_not_retained=True,
        original_native_cases_prompts_extractors_and_budgets_unchanged=True,
        secondary_ordinary_Python_original_program_bytes_not_repaired=True,
        secondary_runtime_is_not_the_original_math_only_prompt_contract=True,
        fixed_development_assessment_not_independent_final_benchmark=True,
        no_checkpoint_selected_as_winner_or_promoted=True,model_updates=0,goal_achieved=False))
    deadline=time.monotonic()+24*3600;endpoint_results={};started=time.monotonic()
    try:
        native_controls=[]
        for language,index in [('Python',4),('JavaScript',5)]:
            for task in TASKS:
                control=native_execute(request(task,language,task[index],True))
                if not control.get('functional_passed') or not control.get('efficiency_completed'):
                    raise ValueError('Every canonical original native function must pass before model generation')
                native_controls.append(dict(language=language,task=task[0],actual_native_reference_execution=control))
        atomic_json(OUT/'all24-original-native-reference-controls.json',native_controls)
        controls=[]
        for task in TASKS:
            payload=request(task,'Python',task[4]);control=ordinary_execute(task[4]+'\nif __name__ == "__main__":\n    pass\n',task[0],payload['cases'])
            if not control['functional_passed']:raise ValueError('Every canonical ordinary-Python main-guard reference must pass')
            controls.append(dict(task=task[0],actual_ordinary_module_control=control))
        atomic_json(OUT/'all12-original-ordinary-module-reference-controls.json',controls)
        for endpoint in checkpoints:
            checkpoint=Path(endpoint['checkpoint'])
            while not checkpoint.exists():
                guard.check(full=False)
                if (SOURCE/'failure.json').exists():raise ValueError('Original code/state training failed')
                if not alive(source_plan['owner']):raise ValueError('Exact original code/state owner terminated before the fixed checkpoint')
                if time.monotonic()>deadline:raise ValueError('Observer deadline; no main-model restart or checkpoint substitution attempted')
                time.sleep(20)
            digest=file_hash(checkpoint)
            if 'expected_SHA256' in endpoint and endpoint['expected_SHA256']!=digest:raise ValueError('Original complete own starting checkpoint bytes changed')
            source_guard=FileBindings({str(checkpoint):digest,str(source_plan_path):file_hash(source_plan_path)})
            model,metadata=load(checkpoint,source_plan_path,endpoint['arm'],expected_checkpoint_sha256=digest,
                expected_plan_sha256=file_hash(source_plan_path),device='cpu')
            if metadata['actual_completed_updates']!=endpoint['updates'] or model.counts()['total_unique']!=984399690 or torch.cuda.is_initialized():
                raise ValueError('Exact actual984399690-value own code/state checkpoint on CPU required')
            tokenizer=Tokenizer.from_file(str(checkpoint.parent/'tokenizer.json'))
            frozen=signature(model);folder=OUT/endpoint['name'];folder.mkdir();frames={}
            for frame in ['completion','instruction']:
                actions_path=folder/(frame+'-all24-actual-generation-actions.jsonl')
                with actions_path.open('x') as actions:
                    observed_count=0
                    def record(event):
                        nonlocal observed_count
                        observed_count+=1
                        atomic_json(OUT/'progress.json',dict(endpoint=endpoint['name'],frame=frame,
                            completed_original_deliveries_in_frame=observed_count,planned_original_deliveries_in_frame=24,
                            latest_actual_action_count=event['actual_native_action_count'],elapsed_seconds=time.monotonic()-started,
                            inference_precision='CPU_FP32',model_updates=0,goal_achieved=False))
                        actions.write(json.dumps(event,separators=(',',':'))+'\n');actions.flush()
                    with inference_cache(),observe(model,record) as delivered:
                        envelope=evaluate_shard(model,metadata,checkpoint,folder,rank=0,world=1,frame=frame,device='cpu',budget=512)
                report=reconcile([envelope],folder/('code-'+frame+'.json'))
                if len(delivered)!=24:raise ValueError('Every native case must retain its actual native generation actions')
                fresh=[];ordinary=[]
                for case,event in zip(report['results'],delivered):
                    task=next(t for t in TASKS if t[0]==case['task'])
                    if (tokenizer.decode(event['actual_generated_token_ids'])!=case['raw_response']
                            or event['requested_native_action_budget']!=512 or event['actual_native_action_count']>512):
                        raise ValueError('Every actual delivered response must reconstruct from retained real token actions')
                    actual=native_execute(request(task,case['language'],case['code']))
                    if (bool(actual.get('syntax_valid'))!=bool(case['candidate'].get('syntax_valid'))
                            or bool(actual.get('functional_passed'))!=bool(case['candidate'].get('functional_passed'))):
                        raise ValueError('Fresh actual native program execution must reproduce syntax and functional verdicts')
                    fresh.append(dict(language=case['language'],task=case['task'],
                        actual_generated_code_sha256=hashlib.sha256(case['code'].encode()).hexdigest(),actual_execution=actual))
                    if case['language']=='Python':
                        result=ordinary_execute(case['code'],case['task'],request(task,'Python',case['code'])['cases'])
                        ordinary.append(dict(task=case['task'],actual_code=case['code'],ordinary_execution=result,
                            original_native_functional_pass=bool(case['candidate'].get('functional_passed'))))
                if signature(model)!=frozen:raise ValueError('Interim inference or observations mutated model parameters')
                atomic_json(folder/(frame+'-all24-fresh-native-program-reexecutions.json'),fresh)
                atomic_json(folder/(frame+'-all12-ordinary-Python-program-executions.json'),ordinary)
                frames[frame]=dict(original_native_tasks=24,syntax_valid=sum(bool(c['actual_execution'].get('syntax_valid')) for c in fresh),
                    native_functionally_correct=sum(bool(c['actual_execution'].get('functional_passed')) for c in fresh),
                    producer_empirical_efficiency_passes=sum(bool(c['efficiency_stress_passed']) for c in report['results']),
                    actual_ordinary_Python_functions=12,ordinary_Python_correct=sum(bool(c['ordinary_execution']['functional_passed']) for c in ordinary),
                    actual_generated_actions_retained=sum(c['actual_native_action_count'] for c in delivered),
                    no_ordinary_Python_efficiency_claim=True,empirical_efficiency_not_asymptotic_proof=True)
                atomic_json(folder/(frame+'-complete.json'),dict(at=datetime.now(timezone.utc).isoformat(),frame=frame,actual_results=frames[frame],
                    checkpoint_sha256=digest,source_files_sha256={str(p):file_hash(p) for p in [actions_path,folder/('code-'+frame+'.json'),
                        folder/(frame+'-all24-fresh-native-program-reexecutions.json'),folder/(frame+'-all12-ordinary-Python-program-executions.json')]},
                    model_updates=0,candidate_promoted=False,goal_achieved=False))
            source_guard.check(full=True);guard.check(full=True);endpoint_results[endpoint['name']]=frames
            atomic_json(folder/'complete.json',dict(at=datetime.now(timezone.utc).isoformat(),complete=True,actual_code_first_updates=endpoint['updates'],
                actual_results=frames,checkpoint_sha256=digest,every_original_local_case_and_action_preserved=True,
                all_actual_native_programs_freshly_reexecuted=True,model_parameters_unchanged=True,
                not_independent_final_benchmark_or_frontier_equivalence=True,model_updates=0,candidate_promoted=False,goal_achieved=False))
            del model;gc.collect()
        atomic_json(OUT/'complete.json',dict(at=datetime.now(timezone.utc).isoformat(),complete=True,actual_results=endpoint_results,
            all96_original_native_deliveries_and_real_token_actions_retained=True,
            all96_actual_native_programs_freshly_reexecuted=True,all48_actual_Python_programs_also_executed_under_secondary_protocol=True,
            fixed_original_frames_budgets_and_case_populations_unchanged=True,
            source_files_sha256={str(p):file_hash(p) for p in [OUT/'plan.json',OUT/'all24-original-native-reference-controls.json',OUT/'all12-original-ordinary-module-reference-controls.json',
                *(OUT/e['name']/'complete.json' for e in checkpoints)]},
            developmental_observation_not_final_independent_model_selection=True,model_updates=0,candidate_promoted=False,goal_achieved=False))
        print(json.dumps(dict(complete=str(OUT/'complete.json'),actual_results=endpoint_results,goal_achieved=False)),flush=True)
    except BaseException as error:
        atomic_json(OUT/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),completed_endpoints=list(endpoint_results),
            model_updates=0,candidate_promoted=False,goal_achieved=False));raise


if __name__=='__main__':main()
