"""Fixed native validation and unchanged local suites using explicit CPU BF16.

The legacy report sealer assumes CPU FP32. This wrapper checks its original
source/data seals, then records the actual outer CPU autocast before writing.
It retains all original tasks, outputs, scoring and source-role limitations.
"""
from contextlib import contextmanager,ExitStack
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from . import code_quality,knowledge_evaluation,math_evaluation,ocr_evaluation,object_evaluation,vision_evaluation,evaluation_provenance
from .corpus import file_hash
from .diverse_images import atomic_json
from .inference_rotary_cache_v1 import inference_cache
from .native_greedy_observer_v1 import signature
from .public_function_validation_observation_v1 import observe as execute_validation

MODULES=(code_quality,knowledge_evaluation,math_evaluation,ocr_evaluation,object_evaluation,vision_evaluation)


@contextmanager
def native_actions(model,callback):
    """Observe unchanged actual text/image generator tensors, including EOS."""
    if model.training or not callable(callback) or getattr(model,'_repair_observer_active_v1',False):
        raise ValueError('One eval-mode native generator and an observation sink required')
    original=model.generate;before=signature(model);calls=[]
    def generate(input_ids,*args,**kwargs):
        if (input_ids.ndim!=2 or input_ids.shape[0]!=1 or len(args)>1
                or kwargs.get('temperature',0)!=0 or kwargs.get('return_rollout',False)
                or kwargs.get('use_cache',True) is not True or type(kwargs.get('max_new_tokens')) is not int
                or kwargs['max_new_tokens']<1):
            raise ValueError('Complete single-example cached greedy text/image generation required')
        pixels=args[0] if args else kwargs.get('pixel_values')
        output=original(input_ids,*args,**kwargs)
        count=output.shape[1]-input_ids.shape[1] if isinstance(output,torch.Tensor) and output.ndim==2 else -1
        if (not isinstance(output,torch.Tensor) or output.ndim!=2 or output.shape[0]!=1
                or not torch.equal(output[:,:input_ids.shape[1]],input_ids)
                or not 1<=count<=kwargs['max_new_tokens'] or before!=signature(model)):
            raise ValueError('Unmodified generator output, original complete prefix and bounded actions required')
        prefix=input_ids[0].cpu().tolist();actions=output[0,len(prefix):].cpu().tolist();eos=kwargs.get('eos_token_id')
        event=dict(case_position=len(calls),original_input_token_ids=prefix,actual_generated_token_ids=actions,
            original_input_ids_sha256=hashlib.sha256(np.asarray(prefix,dtype='<u2').tobytes()).hexdigest(),
            input_pixels_sha256=None if pixels is None else evaluation_provenance.pixels_sha256(pixels),
            requested_native_action_budget=kwargs['max_new_tokens'],actual_native_action_count=count,
            actual_stopped_on_EOS=bool(eos is not None and actions[-1]==eos),
            inference_device_type='cpu',inference_precision='bf16',
            greedy_native_output_tensor_returned_unchanged=True,generated_program_not_repaired=True)
        if not torch.is_autocast_enabled('cpu') or torch.get_autocast_dtype('cpu')!=torch.bfloat16:
            raise ValueError('The recorded CPU BF16 generation context must actually be active')
        calls.append(event);callback(event);return output
    model.generate=generate;model._repair_observer_active_v1=True
    try:yield calls
    finally:model.generate=original;model._repair_observer_active_v1=False


@contextmanager
def legacy_runtime(model,checkpoint,metadata):
    """In-process loader/precision adapter; no source files or scoring edited."""
    checkpoint=Path(checkpoint).resolve();digest=file_hash(checkpoint);threads=torch.get_num_threads()
    previous_threads=torch.set_num_threads;previous_seal=evaluation_provenance.seal;previous=[]
    wrapper_sha=file_hash(__file__)
    def loader(path,device,training_state=False):
        if (Path(path).resolve()!=checkpoint or str(device)!='cpu' or training_state
                or file_hash(checkpoint)!=digest):raise ValueError('Exact immutable own CPU inference endpoint required')
        return model,metadata
    def seal(report,provenance,*args,**kwargs):
        if (not torch.is_autocast_enabled('cpu') or torch.get_autocast_dtype('cpu')!=torch.bfloat16
                or torch.get_num_threads()!=threads or kwargs.get('device')!='cpu'):
            raise ValueError('Actual fixed CPU BF16 inference context required before report serialization')
        previous_seal(report,provenance,*args,**kwargs)
        report.update(inference_device_type='cpu',inference_precision='bf16',actual_CPU_threads=threads,
            CPU_execution_wrapper_sha256=wrapper_sha,complete_original_legacy_scoring_preserved=True)
        if 'decoding' in report:report['decoding']='greedy, cached CPU BF16'
        return report
    def retain_threads(requested):
        if type(requested) is not int or requested<1:raise ValueError('Valid legacy thread request required')
    try:
        for module in MODULES:
            for name,value in [('load_checkpoint',loader),('seal',seal)]:
                if hasattr(module,name):previous.append((module,name,getattr(module,name)));setattr(module,name,value)
        evaluation_provenance.seal=seal;torch.set_num_threads=retain_threads
        with torch.no_grad(),torch.autocast('cpu',dtype=torch.bfloat16),inference_cache():yield
    finally:
        torch.set_num_threads=previous_threads;evaluation_provenance.seal=previous_seal
        for module,name,value in reversed(previous):setattr(module,name,value)
        if file_hash(__file__)!=wrapper_sha:raise ValueError('Execution wrapper changed during inference')


def source_validation(model,checkpoint,folder,root):
    folder=Path(folder);folder.mkdir();root=Path(root)
    fixed=root/'reports/source-function-validation-runtime-CPU-v1/all15-fixed-native-validation-cases.json'
    cases=json.loads(fixed.read_text());tokenizer=Tokenizer.from_file(str(Path(checkpoint).parent/'tokenizer.json'))
    if len(cases)!=15 or [r['position'] for r in cases]!=list(range(15)):
        raise ValueError('Every15original source validation case in fixed order required')
    results=[];events=[];model.eval()
    with (folder/'native-actions.jsonl').open('x') as actions,(folder/'programs-and-executions.jsonl').open('x') as journal:
        def record(event):events.append(event);actions.write(json.dumps(event)+'\n');actions.flush()
        with torch.no_grad(),torch.autocast('cpu',dtype=torch.bfloat16),inference_cache(),native_actions(model,record):
            for case in cases:
                if (case['source_role']!='validation' or case['optimization_forbidden'] is not True
                        or case['native_generation_budget']!=1024):raise ValueError('Original validation role and complete fixed budget required')
                ids=torch.tensor([case['original_prompt_ids']],dtype=torch.long)
                generated=model.generate(ids,max_new_tokens=1024,eos_token_id=2,use_cache=True)
                event=events[-1]
                if (event['original_input_token_ids']!=case['original_prompt_ids']
                        or event['actual_generated_token_ids']!=generated[0,ids.shape[1]:].tolist()):
                    raise ValueError('Every actual validation action must reconstruct')
                raw=tokenizer.decode(event['actual_generated_token_ids']);program=code_quality.instruction_code(raw)
                result=dict(position=case['position'],original_source_id=case['original_source_id'],source_role='validation',
                    actual_native_event_position=event['case_position'],raw_response=raw,program=program,
                    public_execution=execute_validation(case['original_source_row'],program))
                results.append(result);journal.write(json.dumps(result)+'\n');journal.flush()
                print(json.dumps(dict(validation_completed=len(results),correct=sum(r['public_execution']['all_original_examples_passed'] for r in results))),flush=True)
    report=dict(complete=True,functions=15,syntax_valid=sum(r['public_execution']['syntax_valid'] for r in results),
        all_original_examples_passed=sum(r['public_execution']['all_original_examples_passed'] for r in results),
        true_EOS_stops=sum(e['actual_stopped_on_EOS'] for e in events),fixed_native_action_budget=1024,
        original_validation_cohort_sha256=file_hash(fixed),checkpoint_sha256=file_hash(checkpoint),
        tokenizer_sha256=file_hash(Path(checkpoint).parent/'tokenizer.json'),
        inference_device_type='cpu',inference_precision='bf16',actual_CPU_threads=torch.get_num_threads(),
        execution_wrapper_sha256=file_hash(__file__),results=results,
        original_public_example_answers_visible_in_model_prompt=True,
        source_pack_validation_not_all_history_independent_benchmark=True,generated_program_not_repaired=True,goal_achieved=False)
    atomic_json(folder/'complete.json',report);return report


def local_suites(model,checkpoint,metadata,folder,root,pulses):
    """Full unchanged local48code plus fixed broad development checks."""
    root=Path(root);folder=Path(folder);folder.mkdir();reports={};model.eval()
    jobs=[('code-completion',code_quality,dict(max_new_tokens=512,prompt_mode='completion')),
          ('code-instruction',code_quality,dict(max_new_tokens=512,prompt_mode='instruction')),
          ('knowledge',knowledge_evaluation,dict(data=pulses['knowledge'],max_examples=56,max_new_tokens=32)),
          ('math',math_evaluation,dict(data=root/'data/math-reasoning-v2/test.jsonl',max_examples=16,max_new_tokens=256)),
          ('ocr',ocr_evaluation,dict(data=pulses['ocr'],max_examples=32,max_new_tokens=128,image_size=448)),
          ('objects',object_evaluation,dict(data=pulses['objects'],max_examples=32,max_new_tokens=32)),
          ('vision',vision_evaluation,dict(data=pulses['vision'],max_examples=32,image_size=448))]
    with legacy_runtime(model,checkpoint,metadata):
        for name,module,options in jobs:
            with (folder/(name+'-native-actions.jsonl')).open('x') as journal:
                def record(event):journal.write(json.dumps(event)+'\n');journal.flush()
                with native_actions(model,record):module.evaluate(str(checkpoint),output=folder/(name+'.json'),device='cpu',**options)
            report=json.loads((folder/(name+'.json')).read_text())
            if report.get('inference_precision')!='bf16' or report.get('inference_device_type')!='cpu':raise ValueError('Actual local precision must be sealed')
            reports[name]=report
    if (any(reports['code-'+name]['task_count']!=24 for name in ['completion','instruction'])
            or reports['knowledge']['overall']['questions']!=56 or reports['knowledge']['skipped']
            or reports['math']['evaluated_examples']!=16 or reports['ocr']['overall']['examples']!=20
            or reports['objects']['overall']['examples']!=15 or reports['vision']['images']!=32):
        raise ValueError('All original local text/image task populations must remain')
    atomic_json(folder/'complete.json',dict(complete=True,report_files_sha256={str(folder/(n+'.json')):file_hash(folder/(n+'.json')) for n in reports},
        full_local_code_tasks=48,full_final_pipeline_unchanged=True,local_development_checks_not_final_broad_benchmarks=True,
        source_model_checkpoint_sha256=file_hash(checkpoint),inference_precision='bf16',goal_achieved=False))
    return reports
