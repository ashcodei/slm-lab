"""Four actual Gloo workers: whole admitted programs, uneven tail and recovery.

This is a five-pair mechanics probe on a small own network. It does not claim
full-size GPU capacity, a full source epoch, or generated-code improvement.
"""
import copy
from datetime import datetime, timezone, timedelta
import json
import os
from pathlib import Path
import random
import time

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT/'reports/parallel-machine-preference-four-CPU-v2'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('This mechanics experiment must hide every GPU')
    import numpy as np
    import torch
    from torch import distributed as dist
    from probe_dense_human_preference_stage_full_CPU_v1 import equal
    from slm_lab.codenet_checkpoint_v1 import FileBindings, atomic_checkpoint, capture_rng
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    from slm_lab.execution_contrast_data_v1 import ExecutionContrastData
    from slm_lab.frozen_own_actor_reference_v1 import FrozenOwnActorReference
    from slm_lab.model import ModelConfig
    from slm_lab.own_dense_actor_v1 import OwnDenseActor
    from slm_lab.own_dense_growth_v1 import OwnDenseAssistant, new_backbone
    from slm_lab.parallel_machine_preference_v1 import ParallelMachinePreferenceStage
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    rank = int(os.environ['RANK']); world = int(os.environ['WORLD_SIZE'])
    if world != 4:
        raise ValueError('Four actual independent workers required')
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    dist.init_process_group('gloo', timeout=timedelta(minutes=5))
    started = time.monotonic()
    try:
        if rank == 0:
            if REPORTS.exists(): raise ValueError('Preserve every prior attempt')
            REPORTS.mkdir()
        dist.barrier()
        paths = [Path(__file__), ROOT/'src/slm_lab/parallel_machine_preference_v1.py',
            ROOT/'src/slm_lab/machine_code_preference_stage_v1.py',
            ROOT/'src/slm_lab/execution_contrast_data_v1.py',
            ROOT/'src/slm_lab/own_dense_actor_checkpoint_v1.py',
            ROOT/'src/slm_lab/frozen_own_actor_reference_v1.py',
            ROOT/'tests/test_parallel_machine_preference_v1.py',
            ROOT/'reports/parallel-machine-preference-v1-tests.log',
            ROOT/'data/CodeContests-execution-contrast-admitted-v1/admission.json',
            ROOT/'data/CodeContests-execution-contrast-admitted-v1/manifest.json',
            ROOT/'data/CodeContests-execution-contrast-admitted-v1/pairs.jsonl',
            ROOT/'data/CodeContests-execution-contrast-admitted-v1/tokenizer.json']
        bindings = {str(p):file_hash(p) for p in paths}; guard=FileBindings(bindings)
        data=ExecutionContrastData(ROOT,purpose='optimization')
        order=sorted(range(len(data)),key=lambda p:(max(len(data.sequence(p,f)[0]) for f in ['chosen','rejected']),p))[:5]
        if len(data)!=1620 or len(order)!=5: raise ValueError('Exact admitted population required')
        config=ModelConfig(vocab_size=32768,hidden_size=32,intermediate_size=64,
            num_layers=2,num_heads=4,num_kv_heads=2,max_text_length=4096,
            vision_hidden_size=8,vision_intermediate_size=16,vision_layers=1,
            vision_heads=2,image_size=16,patch_size=4,pool_size=2)
        base=OwnDenseAssistant(new_backbone(config,991421),reward_hidden=8,
            word_temporal=4,pointing_rank=8,router_hidden=8,detector_hidden=4)
        model=OwnDenseActor(base,rank=2,seed=991421)
        parameters=[p for _,p in model.named_selected(('actor',))]
        warm=torch.optim.AdamW(parameters,lr=.001,foreach=False)
        batch=data.batch([order[0]],'chosen','cpu')
        model(**batch)['loss'].backward(inputs=parameters);warm.step();warm.zero_grad(set_to_none=True)
        del warm,batch
        starting=copy.deepcopy(model.export(dict(own_probe='a'*64)))
        frozen=tensor_digest(base.named_parameters());critic=model.parts['critic'].weight.detach().clone();results=[]
        for method in ['chosen_CE','machine_DPO']:
            model.import_export(starting,expected_sources=dict(own_probe='a'*64))
            reference=FrozenOwnActorReference(model,sources=dict(own_probe='a'*64))
            if reference.binding()['frozen_actor_parts_sha256']==tensor_digest(OwnDenseActor(
                    OwnDenseAssistant(new_backbone(config,991421),reward_hidden=8,word_temporal=4,
                    pointing_rank=8,router_hidden=8,detector_hidden=4),rank=2,seed=991421).parts.named_parameters()):
                raise ValueError('The own reference must include an actually learned nonzero actor')
            options=dict(method=method,seed=991423,global_batch_size=4,learning_rate=.001,
                betas=[.9,.95],weight_decay=.01,gradient_clip=1.,beta=.1,chosen_CE_weight=.1)
            optimizer=torch.optim.AdamW(parameters,lr=.001,betas=(.9,.95),weight_decay=.01,foreach=False)
            sources=dict(machine_admission_sha256=file_hash(paths[-4]),kernel_sha256=file_hash(paths[1]))
            stage=ParallelMachinePreferenceStage(model,optimizer,data,reference,options,sources,
                world=4,rank=rank,gradient_group=dist.group.WORLD,control_group=dist.group.WORLD,CPU_order=order)
            random.seed(991427+rank);np.random.seed(991427+rank);torch.manual_seed(991427+rank)
            first=stage.update();path=REPORTS/f'{method}-private-rank-{rank}.pt'
            atomic_checkpoint(path,stage.state(),32*1024**2,26*1024**3)
            checkpoint_hash=file_hash(path)
            def draws():return [random.random(),float(np.random.random()),float(torch.rand(()))]
            expected_draws=draws();expected=stage.update()
            weights=tensor_digest(model.parts.named_parameters());moments=copy.deepcopy(optimizer.state_dict());rng=capture_rng('cpu')
            bad=copy.deepcopy(torch.load(path,weights_only=True,mmap=True))
            bad['core']['core']['rng']['torch']=torch.zeros(3,dtype=torch.uint8)
            try:stage.resume(bad)
            except ValueError:pass
            else:raise ValueError('Invalid private RNG must reject')
            if weights!=tensor_digest(model.parts.named_parameters()) or not equal(moments,optimizer.state_dict()) or not equal(rng,capture_rng('cpu')):
                raise ValueError('Rejected recovery changed actual values or private RNG')
            stage.resume(torch.load(path,weights_only=True,mmap=True))
            if (draws()!=expected_draws or stage.update()!=expected or weights!=tensor_digest(model.parts.named_parameters())
                    or not equal(moments,optimizer.state_dict()) or not equal(rng,capture_rng('cpu')) or file_hash(path)!=checkpoint_hash):
                raise ValueError('Actual next global update, Adam and private RNG must recover exactly')
            if (expected['per_replica_original_pairs']!=[1,0,0,0] or frozen!=tensor_digest(base.named_parameters())
                    or not torch.equal(critic,model.parts['critic'].weight)):
                raise ValueError('Real empty-tail workers and every original base value must remain correct')
            if sorted(first['original_source_positions']+expected['original_source_positions'])!=sorted(order):
                raise ValueError('All five declared complete original pairs must occur once without rounded repeats')
            results.append(dict(method=method,logical_updates=2,physical_optimizer_calls=3,
                actual_global_pair_records=[first,expected],learned_parts_sha256=weights,
                Adam_sha256=tensor_digest((str(i)+'.'+k,s[k]) for i,s in enumerate(optimizer.state.values()) for k in ['step','exp_avg','exp_avg_sq']),
                exact_next_global_update_weights_Adam_and_private_RNG=True,
                invalid_private_RNG_rejected_before_mutation=True,
                final_replica_pair_counts=[1,0,0,0],checkpoint_bytes_unchanged=True))
            del stage,optimizer,reference
        guard.check(full=True)
        result=dict(rank=rank,actual_counts=model.counts(),source_files_sha256=bindings,arms=results,
            all_frozen_base_vision_word_and_unused_critic_exact=True,own_reference_actually_nonzero=True,
            actual_admitted_source_population=len(data),CPU_mechanics_order=order,
            small_random_network=True,full_source_epoch=False,full_size_GPU_capacity_pending=True,
            generated_code_or_broad_quality_not_measured=True,production_updates=0,
            seconds=time.monotonic()-started,goal_achieved=False)
        atomic_json(REPORTS/f'rank-{rank}.json',result)
        proofs=[None]*4;dist.all_gather_object(proofs,result)
        if rank==0:
            for index in range(2):
                if (len({p['arms'][index]['learned_parts_sha256'] for p in proofs})!=1
                        or len({p['arms'][index]['Adam_sha256'] for p in proofs})!=1):
                    raise ValueError('Every actual worker must end with identical actor and Adam values')
            atomic_json(REPORTS/'complete.json',dict(at=datetime.now(timezone.utc).isoformat(),proofs=proofs,
                all_four_CPU_workers_passed=True,both_matched_arms=True,actual_unique_original_pairs_per_arm=5,
                actual_admitted_population=1620,full_source_epoch=False,production_updates=0,goal_achieved=False))
            print(json.dumps(dict(all_four_CPU_workers_passed=True,unique_original_pairs_per_arm=5,
                                  seconds=time.monotonic()-started)),flush=True)
        dist.barrier()
    finally:dist.destroy_process_group()


if __name__=='__main__':main()
