"""Four real CPU replicas: admitted complete code, empty tails and recovery.

This uses a small randomly initialized own network and five previously admitted
complete public code examples. It is a distributed mechanics check, not new
billion-corpus admission, a full-size capacity test or a coding-quality score.
"""
import copy
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import random
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/public-pretraining-four-CPU-v1'
FAMILY = 'retained:original_complete_code'


class ProbeSchedule:
    def __init__(self, order):
        self.entries = [(FAMILY,list(order[:4])), (FAMILY,list(order[4:]))]
        self.identity_sha256 = hashlib.sha256(json.dumps(self.entries).encode()).hexdigest()
    def __len__(self):return len(self.entries)
    def at(self,i):return self.entries[i][0],list(self.entries[i][1])
    def selected_budget(self,data):
        from slm_lab.public_pretraining_data_v1 import TokenBudgetSchedule
        return TokenBudgetSchedule.selected_budget(self,data)


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or int(os.environ['WORLD_SIZE']) != 4:
        raise ValueError('Exactly four real GPU-hidden CPU workers required')
    import numpy as np
    import torch
    from torch import distributed as dist
    from types import SimpleNamespace
    from probe_dense_human_preference_stage_full_CPU_v1 import equal
    from slm_lab.codenet_checkpoint_v1 import FileBindings, atomic_checkpoint, capture_rng, save_collective
    from slm_lab.code_contests_CE_view_v1 import CodeContestsCEView
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    from slm_lab.fast_joint_stage_v1 import make_optimizer, parameter_groups
    from slm_lab.focused_code_continuation_data_v1 import collate_complete
    from slm_lab.local_source_closure_v2 import python_sources
    from slm_lab.model import ModelConfig
    from slm_lab.own_dense_growth_v1 import OwnDenseAssistant, new_backbone
    from slm_lab.public_pretraining_checkpoint_v1 import make_state, restore_state
    from slm_lab.public_pretraining_update_v1 import update
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    rank = int(os.environ['RANK']);started = time.monotonic()
    dist.init_process_group('gloo',timeout=timedelta(minutes=10))
    try:
        if rank == 0:
            if OUT.exists():raise ValueError('Every prior scientific attempt must remain preserved')
            OUT.mkdir()
        dist.barrier()
        tests = ROOT/'reports/public-pretraining-data-update-checkpoint-v1-tests.log'
        if '7 passed' not in tests.read_text():raise ValueError('Actual source/rebatching/modality/recovery controls must pass first')
        view = CodeContestsCEView(ROOT)
        order = [];problems = set()
        for p in sorted(range(len(view)),key=lambda p:(int(view.pack.index[view.positions[p]]['length']),p)):
            problem = view.row(p)['source_problem_id']
            if problem in problems:continue
            order.append(p);problems.add(problem)
            if len(order) == 5:break
        if len(order) != 5:raise ValueError('Five distinct already-admitted complete public training problems required')
        lengths = [int(view.pack.index[p]['length']) for p in view.positions]
        targets = [int(view.pack.index[p]['length'])-int(view.pack.index[p]['target_start']) for p in view.positions]
        data = SimpleNamespace(role='train',purpose='optimization',kinds={FAMILY:'text'},lengths={FAMILY:lengths},targets={FAMILY:targets})
        data.batch = lambda name,p,device:collate_complete([view.sequence(i) for i in p],device)
        schedule = ProbeSchedule(order)
        config = ModelConfig(vocab_size=32768,hidden_size=32,intermediate_size=64,num_layers=2,num_heads=4,
            num_kv_heads=2,max_text_length=4096,vision_hidden_size=8,vision_intermediate_size=16,
            vision_layers=1,vision_heads=2,image_size=16,patch_size=4,pool_size=2)
        def fresh():
            return OwnDenseAssistant(new_backbone(config,991539),reward_hidden=8,word_temporal=4,
                pointing_rank=8,router_hidden=8,detector_hidden=4)
        model = fresh()
        initial_digest = tensor_digest(model.named_parameters())
        initializer = OUT/'own-small-scratch-initializer.pt'
        if rank == 0:
            atomic_checkpoint(initializer,model.export(dict(admitted_source_view_sha256=file_hash(view.folder/'manifest.json'))),16*1024**2,40*1024**3)
        dist.barrier()
        settings = dict(learning_rates=dict(language=.001,vision=.001,word=.001),betas=[.9,.95],weight_decay=.01,
            gradient_clip=1.,warmup_fraction=.02,cosine_floor=.1,maximum_microbatch=1,maximum_microbatch_tokens=4096,
            ctc_weight=1.,word_native_CE_weight=1.,gradient_bucket_values=65536,image_size=16)
        files = {str(p):file_hash(p) for p in [tests,initializer,view.folder/'manifest.json',view.folder/'admission.json',
            *python_sources(ROOT,[Path(__file__)])]}
        files.update({str(p):h for p,h in view.guard.files.items()})
        guard = FileBindings(files)
        plan = dict(initial_dense_settings=dict(initializer_sha256=file_hash(initializer)),actual_counts=model.counts(),
            planned_steps=len(schedule),training_options=settings,source_files_sha256=files,
            original_admitted_code_positions=order,small_scratch_CPU_control=True,production_updates=0,goal_achieved=False)
        digest = hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()
        if rank == 0:atomic_json(OUT/'plan.json',dict(plan,owner=dict(pid=os.getpid(),ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19])))
        optimizer = make_optimizer(model,settings)
        groups = parameter_groups(model)
        inactive_ids = {id(p) for name,rows in groups.items() if name != 'language' for _,p in rows}
        inactive_ids.update(id(p) for p in model.parameters() if not p.requires_grad)
        fixed = tensor_digest((n,p) for n,p in model.named_parameters() if id(p) in inactive_ids)
        random.seed(991541+rank);np.random.seed(991541+rank);torch.manual_seed(991541+rank)
        first = update(model,optimizer,data,schedule,0,settings,rank=rank,world=4)
        state_path = OUT/'after1-all-workers.pt'
        save_collective(state_path,lambda rngs:make_state(model,optimizer,plan,digest,1,[first],rngs,'capacity',
            schedule=schedule,data=data),'cpu',4,rank,32*1024**2,40*1024**3)
        dist.barrier()
        checkpoint_sha = file_hash(state_path)
        def draws():return [random.random(),float(np.random.random()),float(torch.rand(()))]
        expected_draws = draws()
        expected = update(model,optimizer,data,schedule,1,settings,rank=rank,world=4)
        expected_weights = tensor_digest(model.named_parameters());moments = copy.deepcopy(optimizer.state_dict());rng = capture_rng('cpu')
        for corruption in ['source','clock','other_worker_RNG']:
            bad = copy.deepcopy(torch.load(state_path,map_location='cpu',weights_only=True,mmap=True))
            if corruption == 'source':bad['core']['metrics'][0]['global_original_positions'] = [-1]
            elif corruption == 'clock':next(iter(bad['core']['optimizer']['state'].values()))['step'] += 1
            else:bad['core']['rngs'][(rank+1)%4]['torch'] = torch.zeros(3,dtype=torch.uint8)
            try:restore_state(model,optimizer,bad,plan,digest,'cpu',rank,4,'capacity',schedule=schedule,data=data)
            except (ValueError,RuntimeError):pass
            else:raise ValueError('Corrupt original source, leaf clock or another worker RNG must reject')
            if (expected_weights != tensor_digest(model.named_parameters()) or not equal(moments,optimizer.state_dict())
                    or not equal(rng,capture_rng('cpu'))):raise ValueError('Rejected restore mutated the actual model, optimizer or private RNG')
        count,metrics = restore_state(model,optimizer,torch.load(state_path,map_location='cpu',weights_only=True,mmap=True),
            plan,digest,'cpu',rank,4,'capacity',schedule=schedule,data=data)
        if count != 1 or metrics != [first] or draws() != expected_draws:
            raise ValueError('Every private worker RNG and original source cursor must recover exactly')
        repeated = update(model,optimizer,data,schedule,1,settings,rank=rank,world=4)
        if (repeated != expected or expected_weights != tensor_digest(model.named_parameters())
                or not equal(moments,optimizer.state_dict()) or not equal(rng,capture_rng('cpu'))
                or file_hash(state_path) != checkpoint_sha):
            raise ValueError('The actual next distributed update, full weights, Adam and every private RNG must be exact')
        if (sorted(first['global_original_positions']+expected['global_original_positions']) != sorted(order)
                or [len(schedule.at(1)[1][r::4]) for r in range(4)] != [1,0,0,0]
                or fixed != tensor_digest((n,p) for n,p in model.named_parameters() if id(p) in inactive_ids)):
            raise ValueError('No rounded tail repeat or inactive vision/word/head update is allowed')
        proof = dict(rank=rank,initial_parameters_sha256=initial_digest,actual_counts=model.counts(),
            actual_original_admitted_code_positions=order,actual_original_records=[first,expected],
            actual_private_next_update_weights_Adam_and_RNG_bit_exact=True,
            invalid_source_clock_and_other_worker_RNG_rejected_before_mutation=True,
            final_replica_examples=[1,0,0,0],final_model_sha256=expected_weights,
            final_Adam_sha256=tensor_digest((str(i)+'.'+k,s[k]) for i,s in enumerate(optimizer.state.values()) for k in ['step','exp_avg','exp_avg_sq']),
            all_inactive_vision_word_and_auxiliary_values_exact=True,logical_updates=2,physical_optimizer_calls=3,
            small_random_network=True,full_size_four_L40_capacity_not_established=True,
            billion_source_not_optimized=True,generated_code_quality_not_measured=True,production_updates=0,goal_achieved=False)
        atomic_json(OUT/f'rank-{rank}.json',proof)
        proofs = [None]*4;dist.all_gather_object(proofs,proof)
        if len({p['final_model_sha256'] for p in proofs}) != 1 or len({p['final_Adam_sha256'] for p in proofs}) != 1:
            raise ValueError('All four replicas must end with identical full model and Adam values')
        if rank == 0:
            oracle = fresh();oracle_optimizer = make_optimizer(oracle,settings)
            for i in range(len(schedule)):update(oracle,oracle_optimizer,data,schedule,i,settings)
            maximum = max(float((p-q).abs().max()) for p,q in zip(model.parameters(),oracle.parameters()))
            if maximum > 2e-6:raise ValueError('Distributed globally normalized updates differ from actual serial complete-source updates')
            guard.check(full=True)
            atomic_json(OUT/'complete.json',dict(at=datetime.now(timezone.utc).isoformat(),complete=True,
                all_four_actual_CPU_workers_passed=True,actual_original_complete_admitted_code_examples=5,
                final_replica_examples=[1,0,0,0],serial_full_update_maximum_parameter_difference=maximum,
                proofs=proofs,source_files_sha256=files,seconds=time.monotonic()-started,
                production_updates=0,full_source_epoch=False,full_size_four_L40_capacity_pending=True,
                new_billion_source_admission_or_optimization=False,model_quality_gain_measured=False,goal_achieved=False))
            print(json.dumps(dict(complete=str(OUT/'complete.json'),four_actual_CPU_workers_passed=True,
                serial_parameter_maximum_difference=maximum,seconds=time.monotonic()-started)),flush=True)
        dist.barrier()
    finally:dist.destroy_process_group()


if __name__ == '__main__':main()
