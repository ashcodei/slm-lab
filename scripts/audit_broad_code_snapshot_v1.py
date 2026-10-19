"""Read the actual interim broad-code weights/Adam without touching live workers."""
from datetime import datetime,timezone
import json
from pathlib import Path

import torch

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.recurrent_ablation import state_digest

ROOT=Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(2)
    path=ROOT/'checkpoints/dialogue-retention-v1/broad-code/step-00000250.pt'
    state_before=(path.stat().st_ino,path.stat().st_size,path.stat().st_mtime_ns)
    source=ROOT/'checkpoints/extended-curriculum-v1/joint/model.pt'
    plan_path=ROOT/'reports/dialogue-retention-v1/broad-code-plan.json';plan=json.loads(plan_path.read_text())
    bundle=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    base=torch.load(source,map_location='cpu',weights_only=True,mmap=True)
    if (bundle['steps']!=250 or bundle['planned_steps']!=936 or bundle['stage']!='pretrain'
            or bundle['world_size']!=4 or bundle['trainable']!='language' or bundle['truncated_sequences']!=0
            or bundle['config']!=base['config'] or bundle['tokenizer_sha256']!=base['tokenizer_sha256']
            or bundle['data_sha256']!=plan['data_sha256'] or file_hash(source)!=plan['base_checkpoint_sha256']):
        raise ValueError('Exact own base, complete original broad-code stage and interim250-update source identity required')
    packed=ROOT/'data/broad-code-continuation-v1/packed/corpus.json'
    if file_hash(packed)!=bundle['data_sha256']:raise ValueError('Actual immutable packed-source manifest differs')
    model=bundle['model'];original=base['model']
    if set(model)!=set(original) or not torch.equal(model['embedding.weight'],model['lm_head.weight']):
        raise ValueError('All original own tensors and tied token/output weights required')
    changed=[]
    for name,tensor in model.items():
        if tensor.shape!=original[name].shape or tensor.dtype!=original[name].dtype or not bool(torch.isfinite(tensor).all()):
            raise ValueError('Every actual own learned tensor must be finite and preserve layout')
        if not torch.equal(tensor,original[name]):changed.append(name)
    if not changed or any(n.startswith('vision.') or n=='value_head.weight' for n in changed):
        raise ValueError('Language must change while every original vision/critic tensor stays identical')
    unique=sum(t.numel() for name,t in model.items() if name!='lm_head.weight')
    if unique!=706012672:raise ValueError('Actual own706M unique parameter inventory required')
    parameters=[(n,t) for n,t in model.items() if not n.startswith('vision.') and n!='lm_head.weight']
    opt=bundle['optimizer'];group=opt['param_groups'][0]
    if len(opt['param_groups'])!=1 or group['params']!=list(range(len(parameters))):
        raise ValueError('Exact original language-only optimizer parameter order required')
    expected={i for i,(name,_) in enumerate(parameters) if name!='value_head.weight'}
    if set(opt['state'])!=expected or len(expected)!=218:raise ValueError('Every used language parameter needs its Adam moments; unused critic remains untouched')
    for i in expected:
        state=opt['state'][i];parameter=parameters[i][1]
        if set(state)!=set(['step','exp_avg','exp_avg_sq']) or float(state['step'])!=250:
            raise ValueError('Every actual used language Adam step must be250')
        for key in ['exp_avg','exp_avg_sq']:
            value=state[key]
            if value.shape!=parameter.shape or value.dtype!=parameter.dtype or not bool(torch.isfinite(value).all()):
                raise ValueError('Complete finite matching actual Adam moment arrays required')
        if bool(torch.any(state['exp_avg_sq']<0)):raise ValueError('Squared moment cannot be negative')
    torch.Generator().set_state(bundle['torch_rng_state'])
    if bundle['cuda_rng_state'].dtype!=torch.uint8 or bundle['cuda_rng_state'].ndim!=1:
        raise ValueError('Original saved CUDA RNG payload required')
    if (path.stat().st_ino,path.stat().st_size,path.stat().st_mtime_ns)!=state_before:
        raise ValueError('Interim snapshot rotated during readback; do not certify a mixed observation')
    result=dict(observed_at=datetime.now(timezone.utc).isoformat(),passed=True,goal_achieved=False,
        snapshot=str(path),snapshot_sha256=file_hash(path),snapshot_bytes=path.stat().st_size,
        snapshot_may_be_rotated_by_original_retention_policy=True,steps=250,planned_steps=936,world_size=4,
        unique_model_parameters=unique,all_model_tensors_finite=len(model),changed_language_tensor_names=changed,
        every_original_vision_and_critic_tensor_identical=True,tied_embedding_output_exact=True,
        all218_used_language_Adam_steps=250,all436_Adam_moments_finite=True,
        saved_CPU_RNG_valid_and_CUDA_payload_retained=True,complete_independent_four_rank_resume_certificate=False,
        source_base_sha256=plan['base_checkpoint_sha256'],data_sha256=bundle['data_sha256'],tokenizer_sha256=bundle['tokenizer_sha256'],
        nominal_target_exposures_at_snapshot=250*262080,
        source_files_sha256={str(Path(__file__).resolve()):file_hash(__file__),str(plan_path):file_hash(plan_path),str(packed):file_hash(packed)},
        new_execution_preference_feedback_used=False,capability_gain_claim=False,
        scope='Read-only actual interim broad/library-code state. Complete finite weights/moments and original-source identities verified, with frozen vision/critic. Original live workers untouched. This certifies an observed interim state, not full-stage completion, exact independent four-rank recovery or improved code/knowledge/image capability.')
    atomic_json(ROOT/'reports/broad-code-step250-state-readback-v1.json',result)
    print(json.dumps({k:result[k] for k in ['passed','steps','planned_steps','unique_model_parameters','all_model_tensors_finite','all218_used_language_Adam_steps','all436_Adam_moments_finite','goal_achieved']},indent=2))


if __name__=='__main__':main()
