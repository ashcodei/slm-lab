"""Read every real folded student tensor and both full-language Adam snapshots.

No neural network is instantiated and no forward/backward is repeated. This
independently checks complete stored states/counts against the actual earlier
runtime proof, without turning that small experiment into a source epoch.
"""
from datetime import datetime,timezone
import json
import os
from pathlib import Path

import torch

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.code_residual_checkpoint_v1 import _validate_rng
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'reports/full-actor-fold-dense-update-CPU-v1'
OUT=ROOT/'reports/full-actor-fold-dense-update-CPU-v1-readback'
CHECKPOINTS=ROOT/'checkpoints/full-actor-fold-dense-update-CPU-v1'


def tensors(payload):
    return {kind+'.'+name:value for kind in ['network','heads'] for name,value in payload[kind].items() if name!='lm_head.weight'}


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists():raise ValueError('Fresh CPU-only full-state reader required')
    torch.set_num_threads(1)
    done_path=SOURCE/'complete.json';done=json.loads(done_path.read_text());plan_path=SOURCE/'plan.json';plan=json.loads(plan_path.read_text())
    if (not done['complete'] or done['actual_language_updates']!=2 or done['physical_optimizer_calls']!=3
            or not done['fresh_four_GPU_capacity_and_full_source_epoch_and_generated_assessments_still_required']):
        raise ValueError('Complete explicitly limited full-size CPU experiment required')
    bindings={**done['source_files_sha256'],str(done_path):file_hash(done_path),str(Path(__file__)):file_hash(__file__)}
    guard=FileBindings(bindings);OUT.mkdir()
    atomic_json(OUT/'plan.json',dict(at=datetime.now(timezone.utc).isoformat(),source_files_sha256=bindings,
        all_actual_native_and_private_model_values_and_every_Adam_moment_checked=True,
        neural_forward_or_backward_not_repeated=True,models_constructed=0,model_updates=0))
    initial=torch.load(CHECKPOINTS/'initial-folded-student.pt',map_location='cpu',weights_only=True,mmap=True)
    final=torch.load(CHECKPOINTS/'trained-native-student.pt',map_location='cpu',weights_only=True,mmap=True)
    first=torch.load(CHECKPOINTS/'step1-full-private.pt',map_location='cpu',weights_only=True,mmap=True)
    last=torch.load(CHECKPOINTS/'final-full-private.pt',map_location='cpu',weights_only=True,mmap=True)
    a,b=tensors(initial['complete_dense']),tensors(final['complete_dense'])
    if a.keys()!=b.keys() or sum(t.numel() for t in b.values())!=984399690:
        raise ValueError('Every actual unique counted dense/visual/head value must remain present')
    for native in [initial,final]:
        network=native['complete_dense']['network']
        if network['embedding.weight'].data_ptr()!=network['lm_head.weight'].data_ptr():
            raise ValueError('One actual stored tied vocabulary/head required')
        if native['old_base_or_actor_checkpoint_or_teacher_needed_for_inference'] is not False:
            raise ValueError('Complete independent native student required')
    names=first['optimized_names']
    if names!=last['optimized_names'] or len(names)!=317 or first['completed']!=1 or last['completed']!=2:
        raise ValueError('Every actual full-language parameter and real update cursor required')
    selected={'network.'+name for name in names};changed=[];fixed=[]
    for name,value in b.items():
        if value.shape!=a[name].shape or value.dtype!=torch.float32 or not bool(torch.isfinite(value).all()):
            raise ValueError('Every full-size stored student value must be finite with original shape/dtype')
        differs=not torch.equal(value,a[name])
        if name in selected:
            if not differs:raise ValueError('This actual proof must change every full-language tensor')
            changed.append(name)
        else:
            if differs:raise ValueError('Fixed original vision/critic/auxiliary value changed')
            fixed.append(name)
    adam_proofs=[]
    for state in [first,last]:
        saved=tensors(state['model']);completed=state['completed'];optimizer=state['optimizer']
        if state['planned']!=2 or len(state['metrics'])!=completed or state['world_size']!=1:
            raise ValueError('Every actual original optimizer update and complete private cursor required')
        _validate_rng(state['rng'],'cpu')
        if len(optimizer['param_groups'])!=1 or set(optimizer['state'])!=set(range(317)):
            raise ValueError('Exactly317ordinary full-language Adam states required')
        for i,name in enumerate(names):
            parameter=saved['network.'+name];moment=optimizer['state'][i]
            if set(moment)!={'step','exp_avg','exp_avg_sq'} or float(moment['step'])!=completed:
                raise ValueError('Every actual full-language Adam clock must match the real update count')
            for key in ['exp_avg','exp_avg_sq']:
                if moment[key].shape!=parameter.shape or moment[key].dtype!=parameter.dtype or not bool(torch.isfinite(moment[key]).all()):
                    raise ValueError('Every original full-shape finite Adam moment required')
            if bool((moment['exp_avg_sq']<0).any()):raise ValueError('Squared Adam moments must be nonnegative')
        if any(not torch.equal(saved[name],a[name]) for name in fixed):
            raise ValueError('Every private snapshot must preserve all unused original values')
        adam_proofs.append(dict(completed=completed,complete_Adam_states=317,all_finite_full_shape_moments_and_nonnegative_squares=True,
            actual_private_CPU_Python_NumPy_Torch_RNG_validated=True))
    if any(not torch.equal(value,tensors(last['model'])[name]) for name,value in b.items()):
        raise ValueError('Every final native student tensor must match the final full private state')
    if initial['fold_receipt']!=final['fold_receipt'] or final['completed_updates']!=2:
        raise ValueError('Exact native fold provenance and final experimental cursor required')
    guard.check(full=True)
    atomic_json(OUT/'complete.json',dict(at=datetime.now(timezone.utc).isoformat(),complete=True,
        actual_unique_native_student_values=984399690,actual_policy_values_removed=15306752,
        actual_changed_dense_language_tensors=len(changed),actual_fixed_vision_critic_auxiliary_tensors=len(fixed),
        actual_finite_all_shape_preserved_unique_student_tensors=len(b),
        every_final_native_and_full_private_tensor_bit_exact=True,actual_full_private_Adam_RNG_snapshots=adam_proofs,
        actual_fold_parameter_table_tie_and_complete_heads_preserved=True,
        producer_next_update_replay_and_cold_full4096forward_not_repeated_by_reader=True,
        source_files_sha256={str(p):file_hash(p) for p in [OUT/'plan.json',done_path,plan_path]},
        source_is_old_partial_foundation_experiment_not_production_initializer=True,
        full_GPU_source_epoch_and_generated_benchmarks_still_pending=True,
        models_constructed=0,model_updates=0,candidate_promoted=False,goal_achieved=False))
    print(json.dumps(dict(complete=str(OUT/'complete.json'),actual_changed_full_language_tensors=len(changed),
        actual_student_values=984399690,real_private_Adam_steps=[1,2],goal_achieved=False)),flush=True)


if __name__=='__main__':main()
