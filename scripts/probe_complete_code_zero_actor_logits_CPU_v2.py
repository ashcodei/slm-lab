"""Measure zero-adapter versus dense CPU arithmetic before a new experiment."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path

import torch

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.own_dense_standalone_v1 import load
from slm_lab.own_dense_actor_v1 import OwnDenseActor
from slm_lab.own_function_trajectory_preferences_v1 import OwnFunctionTrajectoryPreferences

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/complete-code-zero-actor-logits-CPU-v2'
BASE=ROOT/'checkpoints/code-first-continuation-v3/standalone/model.pt'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists():raise ValueError('Fresh GPU-hidden exact arithmetic diagnostic required')
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    plan=json.loads((ROOT/'reports/completed-code-public-function-rollouts-CPU-v1/plan.json').read_text())
    data=OwnFunctionTrajectoryPreferences(ROOT/'data/own-function-trajectory-preferences-v1','audit')
    base,_=load(BASE,expected_checkpoint_sha256=file_hash(BASE),expected_sources=plan['expected_standalone_sources']);model=OwnDenseActor(base).eval()
    values=[]
    for pair in data.rows:
        ids=torch.tensor([pair['original_native_prompt_ids']]);seen={};hooks=[]
        
        with torch.no_grad():before=base(input_ids=ids)['logits'].clone()
        for h in hooks:h.remove()
        layers={}
        
        with torch.no_grad():after=model(input_ids=ids)['logits']
        for h in hooks:h.remove()
        values.append(dict(id=pair['id'],full_prompt_logits_exact=torch.equal(before,after),max_absolute_logit_error=float((before-after).abs().max()),
            last_prompt_argmax_same=bool(torch.equal(before[:,-1].argmax(-1),after[:,-1].argmax(-1))),
            all_prompt_argmax_same_fraction=float((before.argmax(-1)==after.argmax(-1)).float().mean()),
            every_added_up_factor_exactly_zero=all(int(torch.count_nonzero(layer.up))==0 for layer in model.parts['linears']),per_layer_errors=layers,original_prompt_length=len(pair['original_native_prompt_ids'])))
        print(json.dumps(values[-1]),flush=True)
    OUT.mkdir();atomic_json(OUT/'complete.json',dict(at=datetime.now(timezone.utc).isoformat(),complete=True,actual_values=values,
        source_checkpoint_sha256=file_hash(BASE),actual_counts=model.counts(),neural_optimizer_updates=0,goal_achieved=False))
    print(json.dumps(dict(complete=str(OUT/'complete.json'),actual_values=[{k:v for k,v in row.items() if k!='per_layer_errors'} for row in values])),flush=True)


if __name__=='__main__':main()
