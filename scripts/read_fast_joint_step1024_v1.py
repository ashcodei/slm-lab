"""Read actual durable 1024-update weights/Adam/RNG/source records on CPU.

This verifies saved state and every logged source batch, without replaying
1024 neural backwards or claiming an independent capability improvement.
"""
from collections import Counter
from datetime import datetime,timezone
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import random

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports/fast-joint-step1024-readback-v1'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU reader must hide all GPUs')
    if REPORTS.exists():raise ValueError('Preserve previous actual state readback')
    import numpy as np
    import torch
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    from slm_lab.fast_joint_data_v1 import FastJointSources
    from slm_lab.model import ModelConfig
    from slm_lab.cuda_rng_readback_v1 import review_cuda_rng_layout
    torch.set_num_threads(2);REPORTS.mkdir()
    live=ROOT/'checkpoints/fast-joint-v3/resume.pt';snapshot=REPORTS/'resume-step1024.pt'
    os.link(live,snapshot)  # Atomic future replacement keeps this inode intact.
    raw=torch.load(snapshot,map_location='cpu',weights_only=True,mmap=True)
    if raw['completed']!=1024:raise ValueError('Exact first durable 1024-update production state required')
    plan_path=ROOT/'reports/fast-joint-v3/plan.json';plan=json.loads(plan_path.read_text());digest=file_hash(plan_path)
    checkpoint=ROOT/'checkpoints/fast-joint-v3/step-1024/model.pt'
    initial=ROOT/'checkpoints/fast-joint-v3/initial/model.pt'
    native=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
    before=torch.load(initial,map_location='cpu',weights_only=True,mmap=True)
    if (raw['mode']!='train' or raw['world_size']!=4 or raw['run_plan_sha256']!=digest
            or native['run_plan_sha256']!=digest or before['run_plan_sha256']!=digest
            or native['arm']!='step-1024' or native['completed_updates']!=1024
            or before['completed_updates']!=0 or native['model']['actual_counts']!=plan['actual_counts']):
        raise ValueError('Actual source-bound counted initial/trained/resume identity required')
    now=native['model'];old=before['model'];saved=raw['model']['model']
    groups={name:[] for name in ['language','vision','word']};fixed=[]
    for namespace in ['network','heads']:
        if now[namespace].keys()!=old[namespace].keys() or now[namespace].keys()!=saved[namespace].keys():raise ValueError('Every actual model/head key required')
        for key,value in now[namespace].items():
            if not torch.equal(value,saved[namespace][key]) or not bool(torch.isfinite(value).all()):
                raise ValueError('Native policy and complete resumable weights must agree and be finite')
            if namespace=='network' and key=='lm_head.weight':continue
            qualified=namespace+'.'+key
            if qualified.startswith('network.vision.'):group='vision'
            elif qualified.startswith('network.') and not qualified.startswith('network.value_head.'):group='language'
            elif qualified.startswith('heads.word.'):group='word'
            else:group=None
            (groups[group] if group else fixed).append((qualified,value,old[namespace][key]))
    if not torch.equal(now['network']['embedding.weight'],now['network']['lm_head.weight']):raise ValueError('One exact tied table required')
    total=sum(t.numel() for rows in [*groups.values(),fixed] for _,t,_ in rows)
    if total!=plan['actual_counts']['total_unique'] or total>=10**9:raise ValueError('Every actual unique parameter must remain under1B')
    if any(not torch.equal(t,base) for _,t,base in fixed):raise ValueError('Every unused critic/auxiliary value must remain exact')
    records=raw['metrics'];journal=(ROOT/'reports/fast-joint-v3/training-metrics.jsonl').read_text().splitlines()
    if len(records)!=1024 or [json.loads(s) for s in journal[:1024]]!=records:raise ValueError('Durable state and original update journal differ')
    data=FastJointSources(ROOT,ModelConfig(**now['config']))
    if data.budget()!=plan['complete_source_budget']:raise ValueError('Exact complete original curriculum required')
    batches=[]
    for ordinal,name in enumerate(sorted(data.lengths)):
        rng=random.Random(plan['seed']+ordinal*1009);positions=list(range(len(data.lengths[name])));rng.shuffle(positions)
        width=plan['global_family_batch_widths'][name];parts=[]
        for start in range(0,len(positions),width*32):
            chunk=sorted(positions[start:start+width*32],key=lambda p:(data.lengths[name][p],p))
            parts.extend([chunk[i:i+width] for i in range(0,len(chunk),width)])
        rng.shuffle(parts);batches.extend((Fraction(2*i+1,2*len(parts)),ordinal,name,p) for i,p in enumerate(parts))
    batches.sort();counts=Counter();targets=Counter();contexts=Counter();updates=Counter();seen={k:set() for k in data.lengths}
    for i,r in enumerate(records):
        _,_,family,positions=batches[i]
        if r['step']!=i+1 or r['family']!=family or r['global_original_positions']!=positions:raise ValueError('Whole original source order differs')
        if seen[family].intersection(positions):raise ValueError('A source-family example was repeated')
        seen[family].update(positions)
        target=sum(data.targets[family][p] for p in positions);context=sum(data.lengths[family][p] for p in positions)
        if (r['response_EOS_targets']!=target or r['complete_context_tokens']!=context or r['global_examples']!=len(positions)
                or not math.isclose(r['CE_sum']/target,r['token_mean_CE'],rel_tol=1e-12)):
            raise ValueError('Every complete context, target and objective total must reconstruct')
        counts[family]+=len(positions);targets[family]+=target;contexts[family]+=context;updates[data.kinds[family]]+=1
    expected_steps=dict(language=1024,vision=updates['image']+updates['word'],word=updates['word'])
    changed={};optimized=0;all_ids=set()
    for group in raw['optimizer']['param_groups']:
        name=group['name'];leaves=groups[name]
        if len(group['params'])!=len(leaves):raise ValueError('Actual full-weight group leaf inventory differs')
        changed[name]=dict(actual_values=sum(t.numel() for _,t,_ in leaves),actual_leaves=len(leaves),
            changed_leaves=sum(not torch.equal(t,b) for _,t,b in leaves),actual_per_leaf_Adam_updates=expected_steps[name])
        if not changed[name]['changed_leaves']:raise ValueError('Every selected group must actually change')
        for key,(_,tensor,_) in zip(group['params'],leaves):
            state=raw['optimizer']['state'][key];all_ids.add(key);optimized+=1
            if (set(state)!={'step','exp_avg','exp_avg_sq'} or state['exp_avg'].shape!=tensor.shape
                    or state['exp_avg_sq'].shape!=tensor.shape or float(state['step'])!=expected_steps[name]
                    or any(not bool(torch.isfinite(s).all()) for s in state.values())
                    or bool((state['exp_avg_sq']<0).any())):raise ValueError('Actual finite complete Adam state and exact family-derived leaf clocks required')
    if all_ids!=set(raw['optimizer']['state']) or optimized!=445:raise ValueError('Every and only selected Adam state must be present')
    rngs=[]
    for rank,r in enumerate(raw['rngs']):
        random.Random().setstate(r['python']);n=r['numpy'];np.random.RandomState().set_state((n['kind'],np.array(n['keys'],dtype=np.uint32),n['position'],n['has_gauss'],n['cached_gaussian']))
        private=torch.Generator(device='cpu');private.set_state(r['torch']);draw=float(torch.rand((),generator=private))
        rngs.append(dict(rank=rank,all_CPU_streams_valid=True,next_private_CPU_draw=draw,CUDA_layout=review_cuda_rng_layout(r['cuda'])))
    files=[Path(__file__),plan_path,snapshot,checkpoint,initial]
    atomic_json(REPORTS/'complete.json',dict(observed_at=datetime.now(timezone.utc).isoformat(),actual_completed_updates=1024,
        actual_unique_parameters=total,actual_Adam_states=optimized,actual_group_changes=changed,
        actual_source_family_records=sum(counts.values()),actual_complete_context_tokens=sum(contexts.values()),actual_targets=sum(targets.values()),
        source_records_by_family=dict(counts),target_tokens_by_family=dict(targets),all1024original_batches_and_objective_arithmetic_reconstructed=True,
        actual_native_weights_equal_resumable_weights=True,all_unused_heads_and_critic_exact=True,
        all_four_CPU_RNG_states_valid_CUDA_layout_only=rngs,
        actual_private_CUDA_semantics_and_next_update_proved_in_prior_owned_GPU_capacity=True,
        this_reader_does_not_replay1024neural_backwards_or_establish_held_quality=True,
        source_files_sha256={str(p):file_hash(p) for p in files},candidate_promoted=False,goal_achieved=False))
    print(json.dumps(dict(updates=1024,total_parameters=total,actual_group_changes=changed,actual_targets=sum(targets.values()))),flush=True)


if __name__=='__main__':main()
