"""Measure actual small-model losses/gradients/Adam changes before any adoption.

This is a CPU-only mechanical comparison, not a model-quality experiment or
full-size capacity run. Failed parity tests and their exact sources are bound.
"""
from contextlib import nullcontext
import copy
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path

import torch

from slm_lab.model import ModelConfig
from slm_lab.own_dense_growth_v1 import OwnDenseAssistant,new_backbone
from slm_lab.own_dense_actor_v1 import OwnDenseActor
from slm_lab.own_actor_vision_alignment_v1 import OwnActorVisionAlignment,pair_positives
from slm_lab.own_actor_vision_prefix_reuse_v1 import OwnActorVisionPrefixReuse
from slm_lab.own_actor_vision_prefix_split_backward_v1 import OwnActorVisionPrefixSplitBackward
from slm_lab.local_source_closure_v2 import python_sources

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/visual-prefix-gradient-rounding-CPU-v1'
FAILED = ROOT/'reports/own-actor-vision-prefix-reuse-v1-tests.log'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path,value):
    pending = path.with_suffix(path.suffix+'.pending')
    with pending.open('x') as stream:
        stream.write(json.dumps(value,indent=2)+'\n');stream.flush();os.fsync(stream.fileno())
    pending.replace(path)


def setup():
    c = ModelConfig(vocab_size=64,hidden_size=32,intermediate_size=64,num_layers=2,num_heads=4,num_kv_heads=2,
        max_text_length=64,vision_hidden_size=32,vision_intermediate_size=64,vision_layers=1,vision_heads=4,
        image_size=32,patch_size=16)
    model = OwnDenseActor(OwnDenseAssistant(new_backbone(c,989117),reward_hidden=8,word_temporal=4,
        pointing_rank=8,router_hidden=8,detector_hidden=4,word_expansion=1),rank=2,seed=989217)
    ids = torch.tensor([[1,5,6,7,8,2],[1,9,10,11,2,0],[1,12,13,14,15,2]])
    mask = ids != 0; labels = ids.clone();labels[:,:2] = -100;labels[~mask] = -100
    pixels = torch.randn(3,3,64,64,generator=torch.Generator().manual_seed(989317))
    positive = pair_positives(['a','a','b'],['cat','generic','generic'])
    return model,dict(input_ids=ids,attention_mask=mask,labels=labels,pixel_values=pixels),positive


def differences(left,right):
    squared = reference = 0.;maximum = 0.;worst = None;nonexact = signs = count = 0
    for name,a in left.items():
        b = right[name];delta = (a-b).double();square = float(delta.square().sum())
        squared += square;reference += float(a.double().square().sum());count += a.numel()
        nonexact += int((a != b).sum());signs += int((a*b < 0).sum())
        current = float(delta.abs().max())
        if current > maximum:maximum,worst = current,name
    return dict(all_values=count,nonexact_values=nonexact,opposite_nonzero_sign_values=signs,
        maximum_absolute_difference=maximum,worst_tensor=worst,
        relative_L2_difference=(squared/reference)**.5 if reference else None)


def main():
    if (OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19
            or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'])
            or '4 failed, 3 passed' not in FAILED.read_text()):
        raise ValueError('Original failed optimizer parity and a fresh one-thread low-priority CPU diagnostic required')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    paths = [FAILED,*python_sources(ROOT,[Path(__file__),ROOT/'tests/test_own_actor_vision_prefix_reuse_v1.py'])]
    files = {str(p):sha(p) for p in paths};OUT.mkdir()
    base,batch,positive = setup()
    atomic(OUT/'plan.json',dict(at=datetime.now(timezone.utc).isoformat(),source_files_sha256=files,
        actual_small_fixture_parameters=base.counts()['actual_unique'],complete_cases=4,
        arms=['two_encoder_forwards_joint_backward','one_encoder_forward_joint_backward','one_encoder_forward_split_backward'],
        precisions=['CPU_FP32','CPU_BF16_autocast_FP32_weights'],checkpointing=[False,True],
        actual_optimizer_steps_per_case_per_arm=1,production_optimizer_steps=0,training_admitted=False,goal_achieved=False))
    try:
        results = []
        for bf16 in [False,True]:
            for checkpointing in [False,True]:
                amp = (lambda:torch.autocast('cpu',dtype=torch.bfloat16)) if bf16 else nullcontext
                outcomes = []
                for cls in [OwnActorVisionAlignment,OwnActorVisionPrefixReuse,OwnActorVisionPrefixSplitBackward]:
                    model = copy.deepcopy(base);view = cls(model).train()
                    view.native.gradient_checkpointing = checkpointing;view.native.vision.train()
                    selected = [p for _,p in view.named_selected()];selected_ids = {id(p) for p in selected}
                    fixed = {n:p.detach().clone() for n,p in model.named_parameters() if id(p) not in selected_ids}
                    signature = [(id(p),p.data_ptr(),p.requires_grad) for p in model.parameters()]
                    with torch.no_grad(),amp():text_before = model(batch['input_ids'],attention_mask=batch['attention_mask'])['logits'].clone()
                    calls = [];hook = view.native.vision.register_forward_hook(lambda *args:calls.append(1))
                    opt = torch.optim.AdamW(selected,lr=1e-3,weight_decay=.01,foreach=False)
                    with amp():objective = view.objective(batch,batch['input_ids'],batch['attention_mask'],positive)
                    hook.remove();view.backward(objective)
                    grads = {n:p.grad.detach().clone() for n,p in view.named_selected()}
                    if any(not bool(torch.isfinite(g).all()) for g in grads.values()):raise ValueError('Finite selected gradients required')
                    if any(p.grad is not None for p in model.parameters() if id(p) not in selected_ids):raise ValueError('Fixed leaves acquired gradients')
                    norm = float(torch.nn.utils.clip_grad_norm_(selected,1.,error_if_nonfinite=True))
                    opt.step();opt.zero_grad(set_to_none=True)
                    if (signature != [(id(p),p.data_ptr(),p.requires_grad) for p in model.parameters()]
                            or any(not torch.equal(saved,model.get_parameter(n)) for n,saved in fixed.items())):
                        raise ValueError('All fixed weights and trainability metadata must remain unchanged')
                    with torch.no_grad(),amp():text_after = model(batch['input_ids'],attention_mask=batch['attention_mask'])['logits']
                    if not torch.equal(text_before,text_after):raise ValueError('Text computation changed during the vision-only fixture update')
                    outcomes.append(dict(name=cls.__name__,vision_encoder_forward_calls=len(calls),
                        loss=float(objective['loss'].detach()),caption_CE=float(objective['caption_CE'].detach()),
                        matching_loss=float(objective['matching_loss'].detach()),response_EOS_targets=objective['response_EOS_targets'],
                        gradient_norm_before_clip=norm,gradients=grads,parameters={n:p.detach().clone() for n,p in view.named_selected()},
                        Adam_state_leaves=len(opt.state),all_fixed_weights_and_text_logits_unchanged=True))
                reference_gradients = outcomes[0]['gradients'];reference_parameters = outcomes[0]['parameters'];compared = []
                for result in outcomes:
                    gd = differences(reference_gradients,result.pop('gradients'))
                    wd = differences(reference_parameters,result.pop('parameters'))
                    compared.append(dict(**result,gradient_difference_from_two_forward=gd,Adam_parameter_difference_from_two_forward=wd))
                results.append(dict(BF16_autocast=bf16,checkpointing=checkpointing,arms=compared))
                atomic(OUT/'progress.json',dict(complete_cases=len(results),total_cases=4,training_admitted=False))
        if any(sha(p) != digest for p,digest in files.items()):raise ValueError('Every failed-source and diagnostic byte must remain unchanged')
        atomic(OUT/'complete.json',dict(complete=True,at=datetime.now(timezone.utc).isoformat(),source_files_sha256=files,
            actual_small_fixture_parameters=base.counts()['actual_unique'],actual_cases=results,
            failed_strict_optimizer_parity_not_waived=True,production_candidate_promoted=False,
            GPU_and_full_size_and_quality_unverified=True,training_admitted=False,new_production_optimizer_updates=0,goal_achieved=False))
        print(json.dumps(dict(complete=True,actual_cases=len(results),training_admitted=False)),flush=True)
    except BaseException as error:
        atomic(OUT/'failure.json',dict(error=repr(error),automatic_retry=False,training_admitted=False,goal_achieved=False));raise


if __name__ == '__main__': main()
