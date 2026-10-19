"""Actual857M/BF16 UL-only backward; no optimizer updates or checkpoints."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from tokenizers import Tokenizer
import torch

from slm_lab.code_repetition_objective_v1 import repetition_negative_positions,sequence_unlikelihood
from slm_lab.code_residual_inference_v1 import load_residual_policy
from slm_lab.code_source_generation_diagnostic_v1 import select_controls,OriginalBaseView
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.codenet_token_pack_v1 import CompleteCodePack
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.recurrent_ablation import state_digest

ROOT=Path(__file__).resolve().parents[1]


def main():
    reports=ROOT/'reports/code-repetition-objective-cuda-v1'
    if reports.exists():raise ValueError('Preserve previous objective backward evidence')
    # Fund the main pilot's two complete1.75GiB states, its96MiB overhead, and
    # this additional32MiB report above its unchanged16GiB remaining reserve.
    if shutil.disk_usage(ROOT).free<16*1024**3+2*(7*1024**3//4)+128*1024**2:
        raise ValueError('Both ongoing writer artifacts and diagnostic report must stay funded')
    device=torch.device('cuda:2');torch.cuda.set_device(device);torch.set_num_threads(1)
    if not torch.cuda.is_bf16_supported() or torch.cuda.mem_get_info(device)[0]<20*1024**3:
        raise ValueError('Initially20GiB free on a BF16 diagnostic GPU required')
    torch.cuda.set_per_process_memory_fraction(.35,device)
    pilot=ROOT/'reports/code-residual-lr-pilot-v1';prior=json.loads((pilot/'results.json').read_text())
    plan=json.loads((pilot/'plan.json').read_text());selection_path=pilot/'selected-original-rows.json'
    row=next(r for r in select_controls(json.loads(selection_path.read_text()),budget=1024)
        if r['role']=='train' and r['language']=='C++')
    pack=CompleteCodePack(ROOT/'data/codenet-durable-pack-v2','train',purpose='training')
    ids,labels=pack.sequence(row['local_position']);start=next(i for i,y in enumerate(labels) if y!=-100)
    assert hashlib.sha256(np.asarray(ids,dtype='<u2').tobytes()).hexdigest()==row['full_input_ids_sha256']
    checkpoint=Path(prior['best_complete_checkpoint']);base=Path(plan['source_checkpoint']);token_path=checkpoint.parent/'tokenizer.json'
    paths=[Path(__file__),selection_path,pilot/'plan.json',pilot/'results.json',checkpoint,base,token_path,
        ROOT/'src/slm_lab/code_repetition_objective_v1.py',ROOT/'src/slm_lab/code_residual_inference_v1.py',
        ROOT/'src/slm_lab/code_residual_branch_v1.py',ROOT/'src/slm_lab/code_source_generation_diagnostic_v1.py',
        ROOT/'src/slm_lab/model.py',ROOT/'src/slm_lab/training.py',pack.folder/'manifest.json',pack.folder/'admission.json']
    bound={str(p):file_hash(p) for p in paths}
    bound.update({str(pack.folder/n):h for n,h in pack.manifest['files_sha256'].items()})
    guard=FileBindings(bound);reports.mkdir()
    model,metadata=load_residual_policy(checkpoint,base,token_path,
        expected_checkpoint_sha256=prior['best_complete_checkpoint_sha256'],
        expected_base_sha256=plan['source_checkpoint_sha256'],expected_plan_sha256=prior['run_plan_sha256'],device=str(device))
    assert metadata['model_counts']==dict(total_unique=857007616,frozen_base=706012672,trainable_code=150994944)
    original=OriginalBaseView(model);before=state_digest(model.state_dict());base_before=state_digest(original.state_dict())
    prompt=torch.tensor([ids[:start]],device=device)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        generated=model.generate(prompt,max_new_tokens=128,eos_token_id=2,use_cache=True)[0,start:].tolist()
    negatives=repetition_negative_positions(generated,ids[start:],n=16,eos=2)
    if not negatives:raise ValueError('This original control exposes no selected repetition; inspect before another diagnostic')
    model.train();model.gradient_checkpointing=True
    full=torch.tensor([ids[:start]+generated],device=device)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        logits=model(full)['logits'];loss=sequence_unlikelihood(logits,full,start,negatives)
    loss.backward()
    trainable=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    gradients=[dict(name=n,numel=p.numel(),present=p.grad is not None,
        finite=p.grad is not None and bool(torch.isfinite(p.grad).all()),
        nonzero=p.grad is not None and bool(torch.any(p.grad!=0)),
        L2_norm=float(p.grad.norm()) if p.grad is not None else None) for n,p in trainable]
    assert len(gradients)==48 and sum(g['numel'] for g in gradients)==150994944
    assert all(g['present'] and g['finite'] and g['nonzero'] for g in gradients)
    assert all(p.grad is None for p in original.parameters())
    assert state_digest(model.state_dict())==before and state_digest(original.state_dict())==base_before
    guard.check(full=True)
    token=Tokenizer.from_file(str(token_path))
    atomic_json(reports/'results.json',dict(completed_at=datetime.now(timezone.utc).isoformat(),source_files_sha256=bound,
        original_training_source=row,generation_budget=128,generated_token_ids=generated,
        raw_generated_text=token.decode(generated[:-1] if generated[-1]==2 else generated,skip_special_tokens=False),
        negative_generated_token_positions=negatives,mean_UL=float(loss.detach()),gradients=gradients,
        actual_model_counts=metadata['model_counts'],full_policy_state_sha256_before=before,
        full_policy_state_sha256_after=state_digest(model.state_dict()),frozen_base_state_sha256=base_before,
        base_gradients_absent=True,optimizer_updates=0,new_model_checkpoints=0,primary_benchmark_modified=False,
        candidate_promoted=False,goal_achieved=False,
        scope='One actual857M BF16/checkpointed UL-only backward on one original training prompt and its own generated repetition. All48 code gradients are finite/nonzero, base gradients absent and weights unchanged. This is new-objective CUDA mechanics, not learning or functional correctness.'))
    print(json.dumps(dict(UL_only_backward_verified=True,trainable_gradients=48,mean_UL=float(loss.detach()),
        negative_positions=len(negatives),optimizer_updates=0,original_base_unchanged=True)))


if __name__=='__main__':main()
