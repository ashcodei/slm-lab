"""Actual paired173-update actor/Adam/private-state integrity and divergence."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import random

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports/execution-final173-readback-v1'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU readback must hide all GPUs')
    if REPORTS.exists():raise ValueError('Preserve prior state readback')
    import numpy as np
    import torch
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    torch.set_num_threads(1);run=ROOT/'reports/execution-single-update-recovery-v3'
    receipt_path=run/'train-verified_solution-173-distributed-state.json';receipt=json.loads(receipt_path.read_text())
    if len(receipt['all_four_private_states'])!=4:raise ValueError('Every actual private replica state required')
    proofs=[];bindings={str(receipt_path):file_hash(receipt_path),str(Path(__file__)):file_hash(__file__),str(run/'plan.json'):file_hash(run/'plan.json')};parts_by_arm={};fixed=set()
    for item in sorted(receipt['all_four_private_states'],key=lambda x:x['rank']):
        path=Path(item['path']);rank=item['rank'];state=torch.load(path,map_location='cpu',weights_only=True,mmap=True);core=state['private_replica_state']
        arm='verified_solution' if rank<2 else 'execution_hybrid'
        if (file_hash(path)!=item['sha256'] or core['completed']!=173 or core['planned']!=173 or state['settings']['method']!=arm
                or state['settings']['replicas']!=2 or state['settings']['replica_rank']!=rank%2
                or core['model']['actual_counts']['actual_unique']!=999706442 or state['private_state_is_not_whole_distributed_checkpoint'] is not True):
            raise ValueError('Exact complete own actor/method/private replica173-state required')
        raw=(run/(arm+'-metrics.jsonl')).read_text();rows=list(map(json.loads,raw[:raw.rfind('\n')+1].splitlines()))[:173]
        if rows!=core['metrics']:raise ValueError('Every actual persisted trajectory/reward/objective record must match its original journal')
        native_path=ROOT/'checkpoints/execution-single-update-recovery-v3'/(arm+'-actor.pt')
        native=torch.load(native_path,map_location='cpu',weights_only=True,mmap=True)
        if (native['sources']!=core['model']['sources'] or native['actual_counts']!=core['model']['actual_counts']
                or native['parts'].keys()!=core['model']['parts'].keys()
                or any(not torch.equal(t,native['parts'][n]) for n,t in core['model']['parts'].items())):
            raise ValueError('Complete final native export and every private trained tensor must match')
        bindings[str(native_path)]=file_hash(native_path)
        fixed.add(core['model']['frozen_dense_sha256']);parts=core['model']['parts'];names=core['optimized_names'];adam=core['optimizer']
        if len(names)!=561 or len(adam['state'])!=561:raise ValueError('All561real actor Adam states required')
        for key,name in zip(adam['param_groups'][0]['params'],names):
            s=adam['state'][key]
            if (float(s['step'])!=173 or s['exp_avg'].shape!=parts[name].shape or s['exp_avg_sq'].shape!=parts[name].shape
                    or any(not bool(torch.isfinite(t).all()) for t in s.values()) or bool((s['exp_avg_sq']<0).any())):raise ValueError('Every actual finite selected Adam moment and173clock must match')
        digest=tensor_digest(parts.items());moments=tensor_digest((str(i)+'.'+k,s[k]) for i,s in adam['state'].items() for k in ['step','exp_avg','exp_avg_sq'])
        rng=core['rng'];random.Random().setstate(rng['python']);torch.Generator(device='cpu').set_state(rng['torch']);n=rng['numpy']
        np.random.RandomState().set_state((n['kind'],np.asarray(n['keys'],dtype=np.uint32),n['position'],n['has_gauss'],n['cached_gaussian']))
        proofs.append(dict(global_rank=rank,arm=arm,actual_Adam_states=561,all_selected_Adam_clocks=173,parts_sha256=digest,
            Adam_sha256=moments,private_Python_NumPy_Torch_states_valid=True,private_CUDA_payload_bytes=int(rng['cuda'].numel()),
            private_CUDA_generator_semantics_previously_proved_by_actual_GPU_capacity=True))
        parts_by_arm.setdefault(arm,parts);bindings[str(path)]=file_hash(path)
    if len(fixed)!=1:raise ValueError('Both actual arms must keep the same frozen own dense base')
    for arm in parts_by_arm:
        same=[p for p in proofs if p['arm']==arm]
        if len({p['parts_sha256'] for p in same})!=1 or len({p['Adam_sha256'] for p in same})!=1:
            raise ValueError('Every actual replica inside an arm must agree exactly')
    left,right=[parts_by_arm[a] for a in ['verified_solution','execution_hybrid']]
    different=sum(not torch.equal(t,right[name]) for name,t in left.items() if name!='critic.weight')
    if different==0 or not torch.equal(left['critic.weight'],right['critic.weight']):raise ValueError('Actual selected objectives must diverge while unused critic stays exact')
    REPORTS.mkdir();atomic_json(REPORTS/'complete.json',dict(at=datetime.now(timezone.utc).isoformat(),actual_completed_updates_per_arm=173,
        actual_unique_parameters=999706442,proofs=proofs,within_arm_weights_and_Adam_identical=True,
        differing_actor_tensors_between_actual_objectives=different,actual_own_frozen_dense_base_identical=True,
        all_original_trajectories_rewards_metrics_match_native_journals=True,source_files_sha256=bindings,
        all_final_native_export_and_private_actor_tensors_match=True,explicit_legacy_objective_updates=138,new_single_update_production_updates_per_arm=35,held_native_quality_gain_not_proven=True,not_a_173step_neural_backward_replay=True,candidate_promoted=False,goal_achieved=False))
    print(json.dumps(dict(actual_updates_per_arm=173,differing_actor_tensors=different,all_replica_states_reconciled=True)),flush=True)


if __name__=='__main__':main()
