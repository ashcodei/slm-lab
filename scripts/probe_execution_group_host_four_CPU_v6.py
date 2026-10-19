"""Four CPU/Gloo workers, real original code sources, paired objective arms.

Small random networks and two-action samples test distribution and recovery.
This never consumes a GPU or trains the full-size production student.
"""
from datetime import datetime,timezone,timedelta
import copy
import json
import os
from pathlib import Path
import random
import time

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports/execution-group-host-four-CPU-v6'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Every GPU must be hidden')
    import numpy as np
    import torch
    from torch import distributed as dist
    from slm_lab.codenet_checkpoint_v1 import atomic_checkpoint,capture_rng
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    from slm_lab.execution_group_host_reduce_v6 import HostReducedCausalExecutionGroup
    from slm_lab.model import ModelConfig
    from slm_lab.own_dense_actor_v1 import OwnDenseActor
    from slm_lab.own_dense_growth_v1 import OwnDenseAssistant,new_backbone
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);torch.set_num_threads(1)
    if world!=4:raise ValueError('Exactly four CPU workers required')
    torch.use_deterministic_algorithms(True);dist.init_process_group('gloo',timeout=timedelta(minutes=4));started=time.monotonic()
    try:
        if rank==0:
            if REPORTS.exists():raise ValueError('Preserve previous four-worker experiment')
            REPORTS.mkdir()
        dist.barrier()
        first=dist.new_group([0,1],backend='gloo');second=dist.new_group([2,3],backend='gloo')
        arm='verified_solution' if rank<2 else 'execution_hybrid';group=first if rank<2 else second;replica=rank%2
        config=ModelConfig(vocab_size=32768,hidden_size=32,intermediate_size=64,num_layers=2,num_heads=4,num_kv_heads=2,
            max_text_length=4096,vision_hidden_size=8,vision_intermediate_size=16,vision_layers=1,vision_heads=2,
            image_size=16,patch_size=4,pool_size=2)
        base=OwnDenseAssistant(new_backbone(config,957221),reward_hidden=8,word_temporal=4,pointing_rank=8,router_hidden=8,detector_hidden=4)
        model=OwnDenseActor(base,rank=2,seed=957221)
        params=[p for _,p in model.named_selected(('actor',))];optimizer=torch.optim.AdamW(params,lr=.001,foreach=False)
        stage=HostReducedCausalExecutionGroup(model,optimizer,ROOT,method=arm,seed=957221,group_size=2,max_new_tokens=2,
            world=2,rank=replica,gradient_group=group,control_group=group)
        files=[Path(__file__),ROOT/'src/slm_lab/execution_group_controlled_v2.py',ROOT/'src/slm_lab/execution_group_RL_v1.py',
            ROOT/'src/slm_lab/public_example_code_reward_v1.py',ROOT/'src/slm_lab/execution_group_causal_v3.py',
            ROOT/'src/slm_lab/causal_code_reward_catalog_v1.py',ROOT/'src/slm_lab/causal_code_reward_v2.py',
            ROOT/'reports/causal-code-rewards-v2/admission.json',ROOT/'data/documented-doctest-verified-v1/admission.json',
            ROOT/'reports/execution-group-controlled-v2-tests.log',
            ROOT/'src/slm_lab/execution_group_host_reduce_v6.py',ROOT/'reports/execution-group-causal-four-CPU-v1/complete.json']
        sources={str(p):file_hash(p) for p in files}
        random.seed(957222+replica);np.random.seed(957222+replica);torch.manual_seed(957222+replica)
        fixed=tensor_digest(base.named_parameters());initial=tensor_digest(model.parts.named_parameters())
        first_record=stage.update();snapshot=stage.state(sources)
        atomic_checkpoint(REPORTS/f'private-resume-{rank}.pt',snapshot,16*1024**2,26*1024**3)
        expected=stage.update();learned=tensor_digest(model.parts.named_parameters())
        def adam():return tensor_digest((str(i)+'.'+k,s[k]) for i,s in enumerate(optimizer.state.values()) for k in ('step','exp_avg','exp_avg_sq'))
        expected_adam=adam();RNG=capture_rng('cpu');bad=copy.deepcopy(snapshot)
        bad['private_replica_state']['rng']['torch']=torch.zeros(3,dtype=torch.uint8)
        try:stage.resume(bad,sources)
        except ValueError:pass
        else:raise ValueError('Invalid private CPU RNG must reject')
        if (learned!=tensor_digest(model.parts.named_parameters()) or adam()!=expected_adam
                or not torch.equal(RNG['torch'],torch.get_rng_state())):raise ValueError('Invalid restore mutated policy/Adam/CPU RNG')
        stage.resume(snapshot,sources);repeated=stage.update()
        if expected!=repeated or learned!=tensor_digest(model.parts.named_parameters()) or adam()!=expected_adam:
            raise ValueError('Actual distributed next samples/rewards/weights/Adam/private RNG must recover exactly')
        if (fixed!=tensor_digest(base.named_parameters()) or initial==learned
                or any(not r['all_global_execution_rewards_zero'] for r in (first_record,expected))):
            raise ValueError('Real complete target fallback must learn with unchanged base and honest zero rewards')
        result=dict(rank=rank,arm=arm,replica_rank=replica,actual_counts=model.counts(),actual_Adam_states=len(optimizer.state),
            learned_parts_sha256=learned,Adam_sha256=adam(),actual_logical_updates=2,physical_optimizer_calls=3,
            all_original_prompt_target_and_group_records=[first_record,expected],
            exact_next_distributed_samples_rewards_Adam_RNG_and_weights=True,invalid_private_RNG_rejected_before_mutation=True,
            all_frozen_base_vision_heads_and_critic_exact=True,small_random_network=True,production_updates=0,
            seconds=time.monotonic()-started,source_files_sha256=sources,goal_achieved=False)
        atomic_json(REPORTS/f'rank-{rank}.json',result)
        proofs=[None]*4;dist.all_gather_object(proofs,result)
        if rank==0:
            if len({p['learned_parts_sha256'] for p in proofs})!=1 or len({p['Adam_sha256'] for p in proofs})!=1:
                raise ValueError('Matched seeds/all-zero groups must yield identical CE-control and hybrid weights/Adam on every replica')
            baseline=json.loads((ROOT/'reports/execution-group-causal-four-CPU-v1/complete.json').read_text())
            for actual,prior in zip(sorted(proofs,key=lambda p:p['rank']),sorted(baseline['proofs'],key=lambda p:p['rank'])):
                if any(actual[k]!=prior[k] for k in ['learned_parts_sha256','Adam_sha256','actual_counts']):
                    raise ValueError('Original objective and host transport must yield exact baseline actor and Adam values')
                for a,b in zip(actual['all_original_prompt_target_and_group_records'],prior['all_original_prompt_target_and_group_records']):
                    if any(a[k]!=b[k] for k in ['source_id','global_rewards','global_advantages','complete_public_target_used','gradient_norm_before_clip']):
                        raise ValueError('Original source, reward, objective and gradient decisions must remain exact')
                    for x,y in zip(a['trajectories'],b['trajectories']):
                        if any(x[k]!=y[k] for k in ['response_ids','actual_cached_behavior_logps','own_response']):
                            raise ValueError('Actual own scalar trajectories/probabilities must match the original baseline')
            atomic_json(REPORTS/'complete.json',dict(observed_at=datetime.now(timezone.utc).isoformat(),proofs=proofs,
                all_four_CPU_workers_passed=True,all_zero_group_control_and_hybrid_weights_Adam_identical=True,
                actual_original_functions_used=2,actual_global_samples_per_arm_update=4,maximum_own_actions_per_sample=2,
                full_size_full_horizon_GPU_capacity_and_held_quality_not_tested=True,production_updates=0,
                exact_baseline_scalar_trajectories_weights_and_Adam=True,goal_achieved=False))
            print(json.dumps(dict(all_four_CPU_workers_passed=True,paired_controls_identical_when_rewards_zero=True,seconds=time.monotonic()-started)),flush=True)
        dist.barrier()
    finally:dist.destroy_process_group()


if __name__=='__main__':main()
