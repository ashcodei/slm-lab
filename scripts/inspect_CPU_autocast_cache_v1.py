"""Small actual native-path control for the suspected repeated-weight-cast cost.

The active evaluator source encloses its suites in one no_grad CPU BF16 AMP
context. This recreates that context and a deliberately short-lived control.
No current full-size weights or GPU context are loaded or modified.
"""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from slm_lab.dense_native_view_v2 import CompleteDenseNativeView
from slm_lab.local_source_closure_v2 import python_sources
from slm_lab.model import ModelConfig
from slm_lab.own_dense_growth_v1 import OwnDenseAssistant,new_backbone

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/CPU-autocast-cache-inspection-v1'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    with path.open('x') as stream:stream.write(json.dumps(value,indent=2)+'\n')


class Casts(TorchDispatchMode):
    def __init__(self,weights):super().__init__();self.weights=weights;self.events=[]

    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        options=kwargs or {}
        if (str(func)=='aten._to_copy.default' and options.get('dtype')==torch.bfloat16
                and args and isinstance(args[0],torch.Tensor) and args[0].data_ptr() in self.weights):
            self.events.append(self.weights[args[0].data_ptr()])
        return func(*args,**options)


def recorded(function,weights):
    with Casts(weights) as observer:result=function()
    return result,list(observer.events)


def main():
    if (os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.nice(0)<19 or OUT.exists()
            or any(os.environ.get(k)!='1' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'])):
        raise ValueError('Fresh bounded single-thread low-priority CPU control required')
    paths=python_sources(ROOT,[Path(__file__),ROOT/'scripts/assess_public_pretraining_interim_CPU_v2.py',
        ROOT/'src/slm_lab/function_repair_evaluation_CPU_v1.py'])
    files={str(p):sha(p) for p in paths};OUT.mkdir();torch.set_num_threads(1)
    write(OUT/'plan.json',dict(source_files_sha256=files,scope='Explicit small native own CPU dense model only',
        actual_current_evaluator_source_inspected=True,no_current_weights_or_live_jobs_modified=True,goal_achieved=False))
    config=ModelConfig(vocab_size=128,hidden_size=16,intermediate_size=32,num_layers=2,num_heads=4,num_kv_heads=2,
        max_text_length=32,vision_hidden_size=16,vision_intermediate_size=32,vision_layers=2,vision_heads=4,
        image_size=16,patch_size=4,pool_size=2)
    own=OwnDenseAssistant(new_backbone(config,986117),pointing_rank=4,reward_hidden=4,word_temporal=4,
        router_hidden=4,detector_hidden=4,word_expansion=1).eval()
    view=CompleteDenseNativeView(own).eval();weights={p.data_ptr():n for n,p in view.named_parameters() if p.ndim>=2}
    ids=torch.tensor([[1,4,7,8]]);next_id=torch.tensor([[9]])
    torch.clear_autocast_cache()
    with torch.no_grad(),torch.autocast('cpu',dtype=torch.bfloat16):
        state,prefill=recorded(lambda:view.prefill(ids),weights)
        first,decode1=recorded(lambda:view.decode(next_id,state['cache']),weights)
        second,decode2=recorded(lambda:view.decode(next_id,first['cache']),weights)
    with torch.no_grad(),torch.autocast('cpu',dtype=torch.bfloat16):
        separate1,short1=recorded(lambda:view.decode(next_id,first['cache']),weights)
    with torch.no_grad(),torch.autocast('cpu',dtype=torch.bfloat16):
        separate2,short2=recorded(lambda:view.decode(next_id,first['cache']),weights)
    same=torch.equal(second['logits'],separate1['logits']) and torch.equal(second['logits'],separate2['logits'])
    instrument_valid=bool(prefill) and bool(short1) and bool(short2)
    if not instrument_valid or not same:raise ValueError('Actual cast events and equal unchanged native outputs are required')
    if any(sha(p)!=digest for p,digest in files.items()):raise ValueError('All original and measured runtime source bytes must stay exact')
    proof=dict(complete=True,at=datetime.now(timezone.utc).isoformat(),source_files_sha256=files,
        actual_small_shared_context_weight_casts=dict(prefill=prefill,first_decode=decode1,second_decode=decode2),
        actual_small_separate_context_weight_casts=dict(first_decode=short1,second_decode=short2),
        actual_shared_context_cast_counts=[len(prefill),len(decode1),len(decode2)],
        actual_separate_context_cast_counts=[len(short1),len(short2)],
        same_actual_greedy_logits_bit_exact=True,shared_context_cached_weight_reuse_observed=not decode1 and not decode2,
        original_live_evaluator_uses_shared_context=True,source_weights_already_leaf_FP32_parameters=True,
        original_existing_KV_and_rotary_caches_already_enabled=True,
        new_conversion_optimization_not_justified_by_these_controls=True,
        full_size_live_model_not_profiled=True,full_size_throughput_not_measured=True,
        new_production_optimizer_updates=0,production_model_modified=False,goal_achieved=False)
    write(OUT/'complete.json',proof)
    print(json.dumps(dict(complete=True,shared_context_cast_counts=proof['actual_shared_context_cast_counts'],
        separate_context_cast_counts=proof['actual_separate_context_cast_counts'],live_evaluator_modified=False)))


if __name__=='__main__':main()
