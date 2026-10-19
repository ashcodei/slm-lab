"""Measure actual small-graph fold parity and operator calls, with no GPU use.

This is not a full-size capacity/throughput experiment or a capability score.
"""
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from slm_lab.local_source_closure_v2 import python_sources
from slm_lab.model import ModelConfig
from slm_lab.own_dense_actor_v1 import OwnDenseActor
from slm_lab.own_dense_actor_fold_v1 import fold_actor,signature
from slm_lab.own_dense_growth_v1 import OwnDenseAssistant,new_backbone

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/own-dense-actor-fold-CPU-v1'
TESTS=ROOT/'reports/own-dense-actor-fold-v1-final-tests.log'
INVENTORY=ROOT/'reports/full-function-repair-comparison-CPU-v1/plan.json'
CONFIG=ROOT/'data/public-reasoning-curriculum-v1/model-config.json'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    with path.open('x') as stream:stream.write(json.dumps(value,indent=2)+'\n')


class Operations(TorchDispatchMode):
    def __init__(self):super().__init__();self.counts=Counter()

    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        self.counts[str(func)]+=1
        return func(*args,**(kwargs or {}))


def measured(function):
    with torch.no_grad(),Operations() as operations:out=function()
    return out,dict(operations.counts)


def linear_calls(counts):return counts.get('aten.mm.default',0)+counts.get('aten.addmm.default',0)


def main():
    if (os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.nice(0)<19 or OUT.exists()
            or any(os.environ.get(k)!='1' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'])
            or '7 passed' not in TESTS.read_text()):raise ValueError('Fresh tested single-thread low-priority CPU control required')
    paths=[TESTS,INVENTORY,CONFIG,*python_sources(ROOT,[Path(__file__),ROOT/'tests/test_own_dense_actor_fold_v1.py'])]
    files={str(p):sha(p) for p in paths};OUT.mkdir();torch.set_num_threads(1)
    write(OUT/'plan.json',dict(source_files_sha256=files,scope='Explicit own small two-layer16-hidden learned fixture only',
        source_initializer_seed=985317,actual_fixture_training_updates=3,production_models_loaded=False,
        full_size_capacity_quality_throughput_not_established=True,candidate_promoted=False,goal_achieved=False))
    try:
        config=ModelConfig(vocab_size=128,hidden_size=16,intermediate_size=32,num_layers=2,num_heads=4,num_kv_heads=2,
            max_text_length=32,vision_hidden_size=16,vision_intermediate_size=32,vision_layers=2,vision_heads=4,
            image_size=16,patch_size=4,pool_size=2)
        base=OwnDenseAssistant(new_backbone(config,985317),pointing_rank=4,reward_hidden=4,word_temporal=4,
            router_hidden=4,detector_hidden=4,word_expansion=1)
        actor=OwnDenseActor(base,rank=2);ids=torch.tensor([[1,4,5,6,7,8],[1,8,7,6,5,4]])
        optimizer=torch.optim.AdamW([p for _,p in actor.named_selected(('actor','critic'))],lr=.004,foreach=False)
        for _ in range(3):
            actor.train();optimizer.zero_grad(set_to_none=True);out=actor(ids,labels=ids)
            (out['loss']+(out['values']-.5).square().mean()).backward();optimizer.step()
        optimizer.zero_grad(set_to_none=True);actor.eval();before=signature(actor)
        folded,fold=fold_actor(actor,sources=files,working_budget_bytes=256*1024**2)
        observations=[]
        for side in [None,16,32]:
            pixels=None if side is None else torch.linspace(-1,1,2*3*side*side).reshape(2,3,side,side)
            with torch.no_grad():
                a=actor(ids,pixel_values=pixels);b=folded(ids,pixel_values=pixels)
                with torch.autocast('cpu',dtype=torch.bfloat16):
                    low_a=actor(ids,pixel_values=pixels);low_b=folded(ids,pixel_values=pixels)
            for key in ['logits','values']:torch.testing.assert_close(a[key],b[key],atol=5e-7,rtol=2e-5)
            observations.append(dict(image_side=side,FP32_max_abs={k:float((a[k]-b[k]).abs().max()) for k in ['logits','values']},
                BF16_max_abs={k:float((low_a[k].float()-low_b[k].float()).abs().max()) for k in ['logits','values']},
                FP32_greedy_token_mismatches=int((a['logits'].argmax(-1)!=b['logits'].argmax(-1)).sum()),
                BF16_greedy_token_mismatches=int((low_a['logits'].argmax(-1)!=low_b['logits'].argmax(-1)).sum())))
        _,source_ops=measured(lambda:actor(ids));_,folded_ops=measured(lambda:folded(ids))
        with torch.no_grad():
            a_cache=actor.policy.prefill(ids[:1])['cache'];b_cache=folded.network.prefill(ids[:1])['cache']
        _,source_decode=measured(lambda:actor.policy.decode(torch.tensor([[9]]),a_cache))
        _,folded_decode=measured(lambda:folded.network.decode(torch.tensor([[9]]),b_cache))
        expected_removed=2*7*config.num_layers
        if (linear_calls(source_ops)-linear_calls(folded_ops)!=expected_removed
                or linear_calls(source_decode)-linear_calls(folded_decode)!=expected_removed
                or before!=signature(actor)):
            raise ValueError('Actual forward/decode projection-call reduction and untouched source must match the algebra')
        if any(sha(p)!=digest for p,digest in files.items()):raise ValueError('Every frozen control/source must remain exact')
        inventory=json.loads(INVENTORY.read_text())['actual_student_counts'];production_config=json.loads(CONFIG.read_text())
        write(OUT/'complete.json',dict(complete=True,at=datetime.now(timezone.utc).isoformat(),source_files_sha256=files,
            actual_small_fixture_fold=fold,actual_text_and_image_parity=observations,
            actual_forward_operator_counts=dict(actor=source_ops,folded=folded_ops),
            actual_cached_decode_operator_counts=dict(actor=source_decode,folded=folded_decode),
            actual_forward_2D_linear_calls=dict(actor=linear_calls(source_ops),folded=linear_calls(folded_ops)),
            actual_cached_decode_2D_linear_calls=dict(actor=linear_calls(source_decode),folded=linear_calls(folded_decode)),
            actual_mechanics_tests_passed=7,
            existing_full_size_inventory=inventory,
            shape_derived_full_size_removed_linear_calls_per_forward_or_decode=2*7*production_config['num_layers'],
            shape_derived_full_size_folded_parameter_count=inventory['frozen_dense_and_all_heads'],
            shape_derived_full_size_removed_parameter_count=inventory['actor']+inventory['separate_policy_critic'],
            derived_full_size_counts_are_not_actual_full_size_folding=True,
            production_models_loaded_or_modified=False,production_training_or_reset_job_started=False,
            full_size_capacity_quality_throughput_not_established=True,
            teacher_data_or_external_weights_used=False,candidate_promoted=False,goal_achieved=False))
        print(json.dumps(dict(complete=True,small_forward_linear_calls=[linear_calls(source_ops),linear_calls(folded_ops)],
            small_decode_linear_calls=[linear_calls(source_decode),linear_calls(folded_decode)],production_models_modified=False)))
    except BaseException as error:
        write(OUT/'failure.json',dict(error=repr(error),goal_achieved=False));raise


if __name__=='__main__':main()
