"""Matched full-size own-failure preference/control research, with train controls.

Both arms start from the same complete own code epoch and actor parameters.
Only10hash-sealed source-train functions provide five full supervised epochs;
four other source-train functions remain unoptimized controls. This is neither
human RLHF nor an independent benchmark nor production model promotion.
"""
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import traceback

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.local_source_closure_v2 import python_sources

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/complete-code-zero-actor-full-setup-CPU-v1'
DATA=ROOT/'data/own-function-trajectory-preferences-v1'
BASE=ROOT/'checkpoints/code-first-continuation-v3/standalone/model.pt'
SOURCE=ROOT/'reports/completed-code-public-function-rollouts-CPU-v1'
MAX_STATE=512*1024**2
RESERVE=26*1024**3


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists():raise ValueError('Fresh GPU-hidden actual full-model matched research required')
    import numpy as np
    import torch
    from tokenizers import Tokenizer
    from slm_lab.own_dense_standalone_v1 import load
    from slm_lab.own_dense_actor_v1 import OwnDenseActor
    from slm_lab.frozen_own_actor_reference_v1 import FrozenOwnActorReference,signature
    from slm_lab.own_dense_actor_checkpoint_v1 import make_state,restore_state
    from slm_lab.own_function_trajectory_preferences_v1 import OwnFunctionTrajectoryPreferences
    from slm_lab.own_function_trajectory_preference_stage_v1 import gradients,information
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    from slm_lab.codenet_checkpoint_v1 import atomic_checkpoint,capture_rng
    from slm_lab.native_greedy_observer_v1 import observe
    from slm_lab.inference_rotary_cache_v1 import inference_cache
    from slm_lab.public_function_rollout_observation_v1 import observe as execute
    from slm_lab.code_quality import instruction_code
    import copy
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    tests=[('own-function-trajectory-preferences-v1-final-tests',3),('own-function-trajectory-preference-stage-v1-tests',2)]
    for name,count in tests:
        if str(count)+' passed' not in (ROOT/'reports'/(name+'.log')).read_text():raise ValueError('Actual complete masks, censored-action/reference gradients and control protections must pass')
    data=OwnFunctionTrajectoryPreferences(DATA,'optimization')
    if len(data.optimization_positions)!=10 or len(data.unoptimized_control_positions)!=4:raise ValueError('All10optimization and4unoptimized source-train controls must remain')
    source_plan=json.loads((SOURCE/'plan.json').read_text())
    paths=[BASE,BASE.parent/'tokenizer.json',SOURCE/'plan.json',SOURCE/'complete.json',
        ROOT/'reports/own-function-trajectory-preferences-v1-readback-admission/complete.json',
        *[ROOT/'reports'/(name+'.log') for name,_ in tests],
        *python_sources(ROOT,[Path(__file__),ROOT/'tests/test_own_function_trajectory_preference_stage_v1.py'])]
    files={**data.source_files_sha256,**{str(p):file_hash(p) for p in paths}}
    guard=FileBindings(files)
    base,metadata=load(BASE,expected_checkpoint_sha256=files[str(BASE)],expected_sources=source_plan['expected_standalone_sources'])
    model=OwnDenseActor(base).eval()
    if (base.counts()['total_unique']!=984399690 or model.counts()['actual_unique']!=999706442
            or metadata['completed_updates']!=7558 or torch.cuda.is_initialized()):raise ValueError('Only actual full complete own under1B code models on CPU are eligible')
    initial=copy.deepcopy(model.parts.state_dict());initial_digest=tensor_digest(model.parts.named_parameters())
    base_digest=tensor_digest(base.named_parameters());critic=model.parts['critic'].weight.detach().clone()
    source_rows=json.loads((SOURCE/'all14-original-source-train-records.json').read_text())
    if [r['row']['id'] for r in source_rows]!=[r['id'] for r in data.rows]:raise ValueError('Every original source-train diagnostic row must match its exact trajectory pair')
    # Verify the initial zero-factor policy supplies the same full actual
    # original prompt logits as the source model; no initial adapter learning.
    zero_errors=[]
    with torch.no_grad():
        for pair in data.rows:
            ids=torch.tensor([pair['original_native_prompt_ids']])
            original=base(input_ids=ids)['logits'];policy=model(input_ids=ids)['logits']
            error=float((original-policy).abs().max())
            zero_errors.append(dict(id=pair['id'],max_absolute_error=error,exact=torch.equal(original,policy),last_argmax_same=bool(torch.equal(original[:,-1].argmax(-1),policy[:,-1].argmax(-1)))))
            print(json.dumps(zero_errors[-1]),flush=True)
    OUT.mkdir()
    atomic_json(OUT/'complete.json',dict(complete=True,actual_full_setup_prompt_calibration=zero_errors,
        actual_counts=model.counts(),source_files_sha256=files,neural_optimizer_updates=0,goal_achieved=False))


if __name__=='__main__':main()
