"""Disposable full own-model CPU mechanics probe, using synthetic inputs only.

This is not a GPU capacity proof, corpus training, or a capability evaluation.
Only JSON evidence is retained; the original checkpoint is never rewritten.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import os
if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
    raise ValueError('Explicitly disable GPU visibility before starting this CPU-only probe')

from tokenizers import Tokenizer
import torch

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.model import ModelConfig, TinyMultimodalLM
from slm_lab.point_grounding_head_v1 import PointGroundingHead

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'checkpoints/extended-curriculum-v1/joint/model.pt'
BASE_SHA='2dacd490e4a6ed20527a8393629d560e7230c1b6fdec54c1a49dc2eedf7f4e49'
TOKEN=BASE.parent/'tokenizer.json'
REPORTS=ROOT/'reports/point-grounding-full-CPU-v2'


def digest(state, vision):
    result=hashlib.sha256()
    for name,tensor in sorted(state.items()):
        if name.startswith('vision.') != vision:
            continue
        result.update(json.dumps([name,list(tensor.shape),str(tensor.dtype)],separators=(',',':')).encode())
        result.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def main():
    if REPORTS.exists():
        raise ValueError('Preserve previous full-model CPU probe evidence')
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.manual_seed(348107)
    if file_hash(BASE)!=BASE_SHA:
        raise ValueError('Exact original own joint checkpoint required')
    for name in ['point-grounding-head-v1-tests.log','point-grounding-data-v1-tests.log','point-grounding-protocol-v1-tests.log']:
        if '2 passed' not in (ROOT/'reports'/name).read_text():
            raise ValueError('Actual small-model/objective/input/control tests required')
    paths=[Path(__file__).resolve(),ROOT/'src/slm_lab/point_grounding_head_v1.py',
        ROOT/'src/slm_lab/model.py',TOKEN,BASE]
    bound={str(p):file_hash(p) for p in paths}
    REPORTS.mkdir()
    atomic_json(REPORTS/'plan.json',dict(created_at=datetime.now(timezone.utc).isoformat(),
        source_files_sha256=bound,device='CPU',precision='FP32',batch_size=1,query_token_budget=34,
        image_size=448,head_rank=256,disposable_updates=1,synthetic_inputs_and_attention_counts=True,
        data_candidate_used=False,production_weights_will_be_saved=False,goal_achieved=False))
    started=time.monotonic()
    print('loading_actual_original_own706M_checkpoint_CPU',flush=True)
    state=torch.load(BASE,map_location='cpu',weights_only=True,mmap=True)
    tokenizer=Tokenizer.from_file(str(TOKEN))
    if state['tokenizer_sha256']!=file_hash(TOKEN):
        raise ValueError('Actual checkpoint own BPE must match')
    base=TinyMultimodalLM(ModelConfig(**state['config']))
    base.load_state_dict(state['model']);del state
    wrapped=PointGroundingHead(base,rank=256)
    inventory=wrapped.trainable_inventory()
    if (inventory['total_unique_parameters']!=706799104 or inventory['actual_trainable_parameters']!=52410880
            or inventory['actual_additional_head_parameters']!=786432):
        raise ValueError('Actual complete own-model parameter inventory differs')
    print('actual_full_706799104_parameter_inventory_verified',flush=True)
    frozen_before=digest(base.state_dict(),False); vision_before=digest(base.state_dict(),True)
    ids=torch.tensor([[1,*tokenizer.encode('red flowers in the garden').ids,2]])
    if ids.shape[1]>34:
        raise ValueError('Probe query unexpectedly exceeds whole-query budget')
    pixels=torch.linspace(-1,1,3*448*448,dtype=torch.float32).reshape(1,3,448,448)
    targets=torch.zeros(1,196); targets[0,57]=3; targets[0,58]=1
    valid=torch.ones(1,196,dtype=torch.bool)
    base.eval()
    with torch.no_grad():
        text_before=base.generate(ids,max_new_tokens=4,eos_token_id=2)
    query_before=wrapped.query_features(ids).clone()
    wrapped.train()
    trainable=[p for p in wrapped.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(trainable,lr=1e-5,weight_decay=.01)
    print('full_448px_forward_backward_and_disposable_Adam_step_starting',flush=True)
    result=wrapped(ids,pixels,targets=targets,valid_cells=valid)
    loss=result['loss']; loss.backward()
    if not torch.isfinite(loss) or any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in wrapped.parameters()):
        raise ValueError('Actual full-model loss and every gradient must be finite')
    if (not any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in base.vision.parameters())
            or any(p.grad is not None for n,p in base.named_parameters() if not n.startswith('vision.'))
            or not all(p.grad is not None and bool(p.grad.abs().sum()>0) for p in [wrapped.query_projection.weight,wrapped.patch_projection.weight])):
        raise ValueError('Only actual own vision/head must receive nonzero gradients')
    norm=torch.nn.utils.clip_grad_norm_(trainable,1.)
    optimizer.step();optimizer.zero_grad(set_to_none=True)
    if (len(optimizer.state)!=len(trainable) or any(float(s['step'])!=1 for s in optimizer.state.values())
            or any(not torch.isfinite(s[key]).all() for s in optimizer.state.values() for key in ('exp_avg','exp_avg_sq'))):
        raise ValueError('Every actual trainable parameter needs finite complete disposable Adam state')
    frozen_after=digest(base.state_dict(),False); vision_after=digest(base.state_dict(),True)
    base.eval()
    with torch.no_grad():
        text_after=base.generate(ids,max_new_tokens=4,eos_token_id=2)
    query_after=wrapped.query_features(ids)
    if (frozen_before!=frozen_after or vision_before==vision_after or not torch.equal(text_before,text_after)
            or not torch.equal(query_before,query_after)):
        raise ValueError('Actual full-model frozen language/critic/query features/text-only generation changed')
    if any(file_hash(p)!=sha for p,sha in bound.items()) or torch.cuda.is_initialized():
        raise ValueError('Original source weights must be unchanged and no CUDA context may be initialized')
    atomic_json(REPORTS/'complete.json',dict(completed_at=datetime.now(timezone.utc).isoformat(),
        actual_inventory=inventory,device='CPU',precision='FP32',image_size=448,actual_image_cells=196,
        actual_query_tokens=ids.shape[1],actual_synthetic_attention_samples=4,actual_loss=float(loss.detach()),
        actual_gradient_norm_before_clipping=float(norm),actual_disposable_optimizer_steps=1,
        actual_finite_complete_Adam_states=len(optimizer.state),frozen_language_critic_SHA_before=frozen_before,
        frozen_language_critic_SHA_after=frozen_after,vision_SHA_before=vision_before,vision_SHA_after=vision_after,
        original_query_features_and_text_only_generated_tokens_bit_identical=True,
        actual_text_only_probe_generated_ids=text_before.tolist(),source_files_sha256=bound,
        CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),CUDA_initialized=torch.cuda.is_initialized(),
        elapsed_seconds=time.monotonic()-started,production_model_updates=0,model_weights_saved=False,
        corpus_training=False,GPU_capacity_established=False,capability_gain_established=False,goal_achieved=False))
    print('actual_full_model_CPU_vision_gradients_Adam_and_frozen_text_preservation_verified',flush=True)


if __name__=='__main__':
    main()
