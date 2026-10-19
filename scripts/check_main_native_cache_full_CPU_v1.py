"""Read-only real984Mcache/full-forward diagnostics at two fixed own endpoints.

Tests the actual complete4096code context, original native code prefix and a
real whole448training photo. Does not preempt a GPU or alter existing scoring,
checkpoint selection, training data or optimizer state.
"""
from datetime import datetime,timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
from PIL import Image,ImageOps
import torch
from tokenizers import Tokenizer

from slm_lab.code_contests_CE_view_v1 import CodeContestsCEView
from slm_lab.codenet_checkpoint_v1 import FileBindings,capture_rng
from slm_lab.corpus import file_hash
from slm_lab.dense_foundation_inference_v1 import DenseNativeView
from slm_lab.diverse_images import atomic_json
from slm_lab.fast_joint_inference_v2 import load
from slm_lab.inference_rotary_cache_v1 import inference_cache
from slm_lab.local_source_closure_v2 import python_sources
from slm_lab.native_cache_comparison_v1 import compare,chunks,signature
from slm_lab.source_jsonl_records_v1 import records

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/main-native-cache-full-CPU-v1'


def equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a) is type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists():raise ValueError('Fresh CPU-only native cache diagnostic required')
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    tests=ROOT/'reports/native-cache-comparison-v1-tests.log'
    if '2 passed' not in tests.read_text():raise ValueError('Actual text/image/chunk/single parity and deliberate decoder-fault detection controls required')
    source_plan_path=ROOT/'reports/phase-joint-v3/plan.json'
    reference_path=source_plan_path.with_name('train-reference-binding.json');reference=json.loads(reference_path.read_text())
    endpoints=[dict(name='foundation12853',arm='train-foundation-reference',updates=12853,path=reference['path'],sha256=reference['checkpoint_sha256']),
        dict(name='instruction14336',arm='step-14336',updates=14336,
            path=str(ROOT/'checkpoints/phase-joint-v3/step-14336/model.pt'))]
    endpoints[1]['sha256']=file_hash(endpoints[1]['path'])
    view=CodeContestsCEView(ROOT);lengths=view.pack.index[view.positions]['length']
    position=int(np.argmax(lengths));long_ids,_=view.sequence(position)
    if len(long_ids)!=4096:raise ValueError('Actual full original4096code context required')
    native_action_path=ROOT/'reports/main-foundation-instruction14336-code-CPU-v2/foundation12853/completion-all24-actual-generation-actions.jsonl'
    native=next(records(native_action_path));short_prefix=native['original_input_token_ids']
    actual_old_actions=native['actual_generated_token_ids'][:16]
    short_ids=short_prefix+actual_old_actions
    photo_folder=ROOT/'data/alignment-curriculum-v1';photo_path=photo_folder/'contrastive.train.jsonl'
    photo=next(r for r in records(photo_path) if r.get('annotation_origin')=='Original human spoken image description, manually transcribed.' and r['split']=='train')
    image_path=Path(photo['image'])
    if file_hash(image_path)!=photo['stored_image_sha256']:raise ValueError('Original whole public training photo must remain exact')
    with Image.open(image_path) as source:
        array=np.asarray(ImageOps.pad(source.convert('RGB'),(448,448),method=Image.Resampling.BICUBIC,color=(128,128,128)),dtype=np.uint8).copy()
    if hashlib.sha256(array.tobytes()).hexdigest()!=photo['model_pixel_sha256_448']:raise ValueError('Actual original native448pixels must reconstruct')
    pixels=torch.from_numpy(array.astype(np.float32)/127.5-1).permute(2,0,1).unsqueeze(0)
    tokenizer=Tokenizer.from_file(str(view.tokenizer_path))
    photo_ids=[1]+tokenizer.encode('User: '+photo['prompt']+'\nAssistant:').ids+tokenizer.encode(photo['response']).ids+[2]
    inputs=[dict(name='original_native_code_prefix_and16actual_old_actions',ids=short_ids,pixels=None,paths=[
        dict(prefix=len(short_prefix),chunks=[1]*len(actual_old_actions),rotary_tables=False),
        dict(prefix=len(short_prefix),chunks=[1]*len(actual_old_actions),rotary_tables=True),
        dict(prefix=1,chunks=chunks(len(short_ids)-1,16),rotary_tables=True)]),
        dict(name='whole4096original_public_code',ids=long_ids,pixels=None,paths=[
            dict(prefix=1,chunks=[31,224,1792,2032,16],rotary_tables=False),
            dict(prefix=1,chunks=[31,224,1792,2032,16],rotary_tables=True),
            dict(prefix=4080,chunks=[1]*16,rotary_tables=True)]),
        dict(name='whole_native448real_human_training_photo',ids=photo_ids,pixels=pixels,paths=[
            dict(prefix=1,chunks=chunks(len(photo_ids)-1,16),rotary_tables=False),
            dict(prefix=1,chunks=chunks(len(photo_ids)-1,16),rotary_tables=True)])]
    files={str(p):file_hash(p) for p in [tests,ROOT/'tests/test_native_cache_comparison_v1.py',source_plan_path,reference_path,
        native_action_path,photo_path,photo_folder/'manifest.json',image_path,view.folder/'manifest.json',view.folder/'admission.json',
        view.tokenizer_path,*python_sources(ROOT,[Path(__file__),ROOT/'tests/test_native_cache_comparison_v1.py'])]}
    files.update({str(p):h for p,h in view.guard.files.items()});guard=FileBindings(files);OUT.mkdir();started=time.monotonic()
    owner=dict(pid=os.getpid(),ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19])
    atomic_json(OUT/'plan.json',dict(at=datetime.now(timezone.utc).isoformat(),owner=owner,source_files_sha256=files,
        fixed_own_endpoints=endpoints,actual_unique_student_values=984399690,CPU_threads=2,
        complete_original_code_view_position=position,whole_code_context=4096,photo_id=photo['id'],photo_native448=True,
        measured_precision='CPU_FP32',maximum_absolute_logit_bound=.002,
        original_rotary_full_forward_oracle_versus_original_and_accelerated_cached_paths=True,
        greedy_cached_and_full_recomputation_budget=16,
        actual_data_inputs_and_paths=[{k:v for k,v in r.items() if k!='pixels'} for r in inputs],
        existing_native512evaluation_protocol_and_current_training_unchanged=True,
        no_optimizer_or_gradient_or_GPU_initialization=True,
        no_all_contexts_devices_BF16_or_model_quality_claim=True,model_updates=0,candidate_promoted=False,goal_achieved=False))
    results={}
    try:
        for endpoint in endpoints:
            source_guard=FileBindings({endpoint['path']:endpoint['sha256'],str(source_plan_path):file_hash(source_plan_path)})
            assistant,metadata=load(endpoint['path'],source_plan_path,endpoint['arm'],expected_checkpoint_sha256=endpoint['sha256'],
                expected_plan_sha256=file_hash(source_plan_path),device='cpu')
            if assistant.counts()['total_unique']!=984399690 or metadata['actual_completed_updates']!=endpoint['updates'] or torch.cuda.is_initialized():
                raise ValueError('Exact own984MFP32CPUmodel endpoint required')
            model=DenseNativeView(assistant).eval();frozen=signature(model);rng=capture_rng('cpu');observations=[]
            folder=OUT/endpoint['name'];folder.mkdir()
            for row in inputs:
                atomic_json(OUT/'progress.json',dict(owner=owner,endpoint=endpoint['name'],phase='actual_cache_vs_full_forward',
                    actual_native_input=row['name'],elapsed_seconds=time.monotonic()-started,model_updates=0,goal_achieved=False))
                def record(path_result):atomic_json(folder/(row['name']+'-latest-actual-path.json'),path_result)
                actual=compare(model,torch.tensor([row['ids']]),pixels=row['pixels'],paths=row['paths'],record=record)
                observations.append(dict(input=row['name'],**actual));atomic_json(folder/(row['name']+'-complete.json'),actual)
            prefix=torch.tensor([short_prefix]);deliveries={}
            with torch.inference_mode():
                uncached=model.generate(prefix,max_new_tokens=16,eos_token_id=2,use_cache=False)
                cached=model.generate(prefix,max_new_tokens=16,eos_token_id=2,use_cache=True)
                with inference_cache():accelerated=model.generate(prefix,max_new_tokens=16,eos_token_id=2,use_cache=True)
            for name,tokens in [('full_recomputation',uncached),('original_cached',cached),('accelerated_cached',accelerated)]:
                deliveries[name]=tokens[0,len(short_prefix):].tolist()
            if signature(model)!=frozen or not equal(rng,capture_rng('cpu')):raise ValueError('Read-only cache diagnostics changed parameters or actual Python/NumPy/Torch RNG')
            if torch.cuda.is_initialized():raise ValueError('CPU diagnosis must never initialize CUDA')
            source_guard.check(full=True);guard.check(full=True)
            outcome=dict(actual_completed_updates=endpoint['updates'],actual_unique_student_values=984399690,
                actual_cache_full_forward_comparisons=observations,actual_16_budget_greedy_deliveries=deliveries,
                actual_greedy_tokens_all_three_paths_equal=torch.equal(uncached,cached) and torch.equal(uncached,accelerated),
                actual_parameters_and_all_private_CPU_RNG_unchanged=True,
                all_observed_logit_paths_within_predeclared_bound=all(r['all_paths_within_bound'] for r in observations),
                not_a_functional_program_score_or_GPU_BF16_parity=True,model_updates=0,goal_achieved=False)
            atomic_json(folder/'complete.json',outcome);results[endpoint['name']]=outcome
            del model,assistant,uncached,cached,accelerated;gc.collect()
        atomic_json(OUT/'complete.json',dict(at=datetime.now(timezone.utc).isoformat(),complete=True,actual_fixed_endpoint_results=results,
            every_actual_requested_full_context_and_full_vocabulary_comparison_retained=True,
            neural_logit_values_not_archived_only_per_position_errors_argmax_flags_and_greedy_actions=True,
            CPU_diagnostic_not_all_devices_contexts_or_generated_code_accuracy=True,
            source_files_sha256={str(p):file_hash(p) for p in [OUT/'plan.json',*[OUT/e['name']/'complete.json' for e in endpoints]]},
            model_updates=0,candidate_promoted=False,goal_achieved=False))
        print(json.dumps(dict(complete=str(OUT/'complete.json'),results={name:dict(
            logits_within_bound=r['all_observed_logit_paths_within_predeclared_bound'],greedy_equal=r['actual_greedy_tokens_all_three_paths_equal']) for name,r in results.items()})),flush=True)
    except BaseException as error:
        atomic_json(OUT/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),completed_fixed_endpoints=list(results),
            no_current_training_or_original_observer_modified=True,model_updates=0,goal_achieved=False));raise


if __name__=='__main__':main()
