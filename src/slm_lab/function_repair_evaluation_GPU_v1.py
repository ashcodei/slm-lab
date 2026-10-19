"""Unchanged full local suites and all15fixed source-validation functions.

GPU scheduling/shards change only which worker performs a case. All raw
generations, native actions, original scorers and failure outcomes remain.
The final broad nine-endpoint assessment pipeline is independent of this.
"""
from contextlib import contextmanager
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from . import code_quality,knowledge_evaluation,math_evaluation,ocr_evaluation,object_evaluation,vision_evaluation
from .corpus import file_hash
from .diverse_images import atomic_json
from .inference_rotary_cache_v1 import inference_cache
from .native_generation_observer_device_v1 import observe
from .native_code_shards_v1 import evaluate_shard as code_shard,reconcile as code_reconcile
from .public_function_validation_observation_v1 import observe as execute_validation


def source_positions(rank,world):
    if type(world) is not int or not 1<=world<=4 or type(rank) is not int or not 0<=rank<world:
        raise ValueError('Explicit complete original source-validation shard required')
    return list(range(rank,15,world))


def validate_source_cohort(path):
    cases=json.loads(Path(path).read_text())
    if (len(cases)!=15 or [c['position'] for c in cases]!=list(range(15))
            or any(c['source_role']!='validation' or c['optimization_forbidden'] is not True
                or c['native_generation_budget']!=1024 or len(c['original_prompt_ids'])+1024>4096 for c in cases)):
        raise ValueError('Every original validation role, whole prefix and fixed1024native-action budget required')
    return cases


def source_shard(model,checkpoint,cohort,folder,*,rank,world=4,device='cuda:0'):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);positions=source_positions(rank,world)
    cases=validate_source_cohort(cohort);tokenizer_path=Path(checkpoint).parent/'tokenizer.json'
    tok=Tokenizer.from_file(str(tokenizer_path));results=[];events=[];model.eval();device=torch.device(device)
    with (folder/('source-validation-actions-'+str(rank)+'.jsonl')).open('x') as actions,(folder/('source-validation-programs-'+str(rank)+'.jsonl')).open('x') as journal:
        def record(event):events.append(event);actions.write(json.dumps(event)+'\n');actions.flush()
        with torch.no_grad(),torch.autocast(device.type,dtype=torch.bfloat16),inference_cache(),observe(model,record,device=device):
            for position in positions:
                case=cases[position];ids=torch.tensor([case['original_prompt_ids']],device=device,dtype=torch.long)
                output=model.generate(ids,max_new_tokens=1024,eos_token_id=2,use_cache=True);event=events[-1]
                if event['actual_generated_token_ids']!=output[0,ids.shape[1]:].cpu().tolist():raise ValueError('Every actual unchanged native delivery must reconstruct')
                raw=tok.decode(event['actual_generated_token_ids']);program=code_quality.instruction_code(raw)
                result=dict(position=position,original_source_id=case['original_source_id'],source_role='validation',
                    actual_event=event,raw_response=raw,program=program,public_execution=execute_validation(case['original_source_row'],program))
                results.append(result);journal.write(json.dumps(result)+'\n');journal.flush()
    report=dict(complete=True,rank=rank,world=world,original_positions=positions,functions=len(results),results=results,
        checkpoint_sha256=file_hash(checkpoint),cohort_sha256=file_hash(cohort),tokenizer_sha256=file_hash(tokenizer_path),
        wrapper_sha256=file_hash(__file__),observer_sha256=file_hash(Path(__file__).with_name('native_generation_observer_device_v1.py')),
        inference_device_type=device.type,inference_precision='bf16',native_budget=1024,
        source_pack_validation_not_all_history_independent=True,goal_achieved=False)
    atomic_json(folder/('source-validation-shard-'+str(rank)+'.json'),report);return report


def reconcile_source(paths,cohort,checkpoint,output):
    paths=list(map(Path,paths));cases=validate_source_cohort(cohort);world=len(paths);by_position={};identity=None
    tok=Tokenizer.from_file(str(Path(checkpoint).parent/'tokenizer.json'))
    for rank,path in enumerate(paths):
        report=json.loads(path.read_text());shared={k:report[k] for k in ['checkpoint_sha256','cohort_sha256','tokenizer_sha256','wrapper_sha256','observer_sha256','inference_device_type','inference_precision','native_budget']}
        if identity is None:identity=shared
        if (shared!=identity or report['rank']!=rank or report['world']!=world or report['original_positions']!=source_positions(rank,world)
                or [r['position'] for r in report['results']]!=source_positions(rank,world) or report['functions']!=len(report['results'])
                or shared['cohort_sha256']!=file_hash(cohort) or shared['checkpoint_sha256']!=file_hash(checkpoint)
                or shared['tokenizer_sha256']!=file_hash(Path(checkpoint).parent/'tokenizer.json')
                or shared['wrapper_sha256']!=file_hash(__file__) or shared['observer_sha256']!=file_hash(Path(__file__).with_name('native_generation_observer_device_v1.py'))
                or shared['inference_device_type']!='cuda' or shared['inference_precision']!='bf16' or shared['native_budget']!=1024):
            raise ValueError('Every original source shard and native profile must match the complete frozen cohort')
        for j,row in enumerate(report['results']):
            p=row['position'];event=row['actual_event'];ids=event['actual_generated_token_ids']
            if (p in by_position or row['original_source_id']!=cases[p]['original_source_id'] or row['source_role']!='validation'
                    or event['case_position']!=j or event['original_input_token_ids']!=cases[p]['original_prompt_ids']
                    or event['requested_native_action_budget']!=1024 or event['actual_native_action_count']!=len(ids) or not 1<=len(ids)<=1024
                    or event['actual_stopped_on_EOS']!=(ids[-1]==2) or row['raw_response']!=tok.decode(ids)
                    or row['program']!=code_quality.instruction_code(row['raw_response']) or event['generated_program_not_repaired'] is not True):
                raise ValueError('Every actual complete native token delivery, EOS boundary and unmodified generated program must reconstruct')
            by_position[p]=row
    if set(by_position)!=set(range(15)):raise ValueError('Every original source-validation function must be assessed exactly once')
    results=[by_position[i] for i in range(15)]
    report=dict(complete=True,**identity,functions=15,results=results,
        syntax_valid=sum(r['public_execution']['syntax_valid'] for r in results),
        all_original_examples_passed=sum(r['public_execution']['all_original_examples_passed'] for r in results),
        true_EOS_stops=sum(r['actual_event']['actual_stopped_on_EOS'] for r in results),
        source_pack_validation_not_all_history_independent=True,original_public_answers_visible_in_prompt=True,
        source_files_sha256={str(p):file_hash(p) for p in paths},goal_achieved=False)
    atomic_json(output,report);return report


@contextmanager
def legacy_loader(model,metadata,checkpoint,device):
    modules=[code_quality,knowledge_evaluation,math_evaluation,ocr_evaluation,object_evaluation,vision_evaluation]
    previous=[];threads=torch.set_num_threads;checkpoint=Path(checkpoint).resolve();digest=file_hash(checkpoint)
    def loader(path,actual,training_state=False):
        if Path(path).resolve()!=checkpoint or str(actual)!=str(device) or training_state or file_hash(checkpoint)!=digest:
            raise ValueError('Exact own immutable declared inference endpoint required')
        return model,metadata
    try:
        for module in modules:previous.append((module,module.load_checkpoint));module.load_checkpoint=loader
        torch.set_num_threads=lambda n:None
        with inference_cache():yield
    finally:
        for module,value in previous:module.load_checkpoint=value
        torch.set_num_threads=threads


def native_local(model,metadata,checkpoint,folder,root,pulses,*,rank,suite,device='cuda:0'):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);root=Path(root);model.eval()
    with legacy_loader(model,metadata,checkpoint,device):
        if suite=='code':
            for frame in ['completion','instruction']:
                with (folder/('code-'+frame+'-actions-'+str(rank)+'.jsonl')).open('x') as journal:
                    def record(event):journal.write(json.dumps(event)+'\n');journal.flush()
                    with observe(model,record,device=device):code_shard(model,metadata,checkpoint,folder,rank=rank,world=4,frame=frame,device=device)
        elif suite=='broad':
            jobs={0:[('knowledge',knowledge_evaluation,dict(data=pulses['knowledge'],max_examples=56,max_new_tokens=32))],
                1:[('math',math_evaluation,dict(data=root/'data/math-reasoning-v2/test.jsonl',max_examples=16,max_new_tokens=256))],
                2:[('ocr',ocr_evaluation,dict(data=pulses['ocr'],max_examples=32,max_new_tokens=128,image_size=448)),
                   ('objects',object_evaluation,dict(data=pulses['objects'],max_examples=32,max_new_tokens=32))],
                3:[('vision',vision_evaluation,dict(data=pulses['vision'],max_examples=32,image_size=448))]}[rank]
            for name,module,options in jobs:
                with (folder/(name+'-actions.jsonl')).open('x') as journal:
                    def record(event):journal.write(json.dumps(event)+'\n');journal.flush()
                    with observe(model,record,device=device):module.evaluate(checkpoint,output=folder/(name+'.json'),device=device,**options)
        else:raise ValueError('Declared complete local suite required')


def reconcile_local(folder):
    folder=Path(folder);reports={}
    for frame in ['completion','instruction']:
        reports['code-'+frame]=code_reconcile([folder/f'code-{frame}-shard-{r}.json' for r in range(4)],folder/('code-'+frame+'.json'))
    for name in ['knowledge','math','ocr','objects','vision']:reports[name]=json.loads((folder/(name+'.json')).read_text())
    if (any(reports['code-'+f]['task_count']!=24 for f in ['completion','instruction'])
            or reports['knowledge']['overall']['questions']!=56 or reports['knowledge']['skipped']
            or reports['math']['evaluated_examples']!=16 or reports['ocr']['overall']['examples']!=20
            or reports['objects']['overall']['examples']!=15 or reports['vision']['images']!=32
            or any(v.get('inference_device_type')!='cuda' or v.get('inference_precision')!='bf16' for v in reports.values())):
        raise ValueError('Every fixed original local text/image/code population and actual GPU precision must remain')
    atomic_json(folder/'complete.json',dict(complete=True,full_local_code_tasks=48,local_development_checks_not_final_broad_benchmarks=True,
        source_files_sha256={str(folder/(name+'.json')):file_hash(folder/(name+'.json')) for name in reports},goal_achieved=False))
    return reports
