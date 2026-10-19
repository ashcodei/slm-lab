"""Two complete original joint source steps and exact durable next replay.

Own-small CPU diagnostic only: real whole code/human/word/photo batches,
all5362declared source entries, pre-Adam journals, sparse Adam, each private
RNG and native cold export. This does not train the full-size final model.
"""
from datetime import datetime,timedelta,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import traceback

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/complete-joint-indexed-training-CPU-v1'
RESERVE=40*1024**3


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path,value):
    raw=(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
    if len(raw)>32*1024**2 or shutil.disk_usage(ROOT).free-len(raw)<RESERVE:raise ValueError('The complete CPU source/controller diagnostic must retain40GiBshared free space')
    pending=Path(path).with_name(Path(path).name+'.pending')
    with pending.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    pending.replace(path)


def worker(rank,files):
    import random
    import numpy as np
    import torch
    from torch import distributed as dist
    from slm_lab.codenet_checkpoint_v1 import capture_rng,FileBindings
    from slm_lab.complete_joint_training_data_v1 import CompleteJointTrainingDataV1
    from slm_lab.complete_joint_objective_bridge_v1 import complete_entries
    from slm_lab.complete_joint_indexed_training_v1 import CompleteJointIndexedTrainingV1
    from slm_lab.deterministic_positions_v1 import install
    from slm_lab.frozen_own_text_reference_v1 import FrozenOwnTextReferenceV1
    from slm_lab.joint_objective_step_v1 import JointReplicaGuardV1
    from slm_lab.model import ModelConfig
    from slm_lab.own_dense_actor_v1 import OwnDenseActor
    from slm_lab.own_dense_finalizer_v1 import finalize
    from slm_lab.own_dense_growth_v1 import OwnDenseAssistant,new_backbone
    from slm_lab.parallel_function_repair_stage_v1 import value_digest
    from slm_lab.shared_joint_actor_vision_v1 import SharedJointActorVisionV1
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and os.nice(0)>=19 and os.sched_getscheduler(0)==os.SCHED_IDLE
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    dist.init_process_group('gloo',init_method='file://'+str(OUT/'rendezvous'),rank=rank,world_size=4,timeout=timedelta(minutes=30))
    try:
        config=ModelConfig(vocab_size=32768,hidden_size=32,intermediate_size=64,num_layers=2,num_heads=4,num_kv_heads=2,max_text_length=4096,
            vision_hidden_size=16,vision_intermediate_size=32,vision_layers=1,vision_heads=4,image_size=224,patch_size=16,pool_size=2)
        source=CompleteJointTrainingDataV1(ROOT,config,word_expansion=5);entries=complete_entries(source)
        own=install(OwnDenseAssistant(new_backbone(config,995001),reward_hidden=8,word_temporal=4,pointing_rank=8,
            router_hidden=8,detector_hidden=4,word_expansion=5)).eval()
        view=SharedJointActorVisionV1(OwnDenseActor(own,rank=2,seed=995002)).eval()
        reference=FrozenOwnTextReferenceV1(own,sources={'complete_admitted_data_sha256':source.identity_sha256},tokenizer_sha256=source.tokenizer_sha256)
        optimizer=torch.optim.AdamW([p for _,p in view.named_selected()],lr=1e-4,eps=1e-5,weight_decay=.01,foreach=False)
        guard=JointReplicaGuardV1(view,optimizer)
        options=dict(maximum_examples=1,maximum_padded_tokens=16384,gradient_bucket_values=4096,gradient_clip=1.,human_weight=.2,visual_weight=.3,
            CTC_weight=1.,native_word_CE_weight=1.,matching_weight=.1,temperature=.07,correction_weight=.1,DPO_beta=.1,chosen_CE_weight=.1,project_conflicts=True,precision='bf16')
        sources=dict(text_initializer_sha256=tensor_digest(own.language_named_parameters()),
            visual_initializer_sha256=tensor_digest((n,p) for n,p in own.named_parameters() if n.startswith(('network.vision.','heads.pointing.','heads.word.'))),
            tokenizer_sha256=source.tokenizer_sha256,run_plan_sha256=value_digest(dict(script_sources=files,options=options,source_identity=source.identity_sha256,
                complete_schedule_sha256=value_digest(entries),explicit_own_small_scratch_integration_control=True)),
            source_admission_sha256=source.identity_sha256,complete_schedule_sha256=value_digest(entries),fixed_values_sha256=guard.fixed.sha256)
        initializer_guard=FileBindings(files)
        history_budgets=dict(maximum_records=5362,maximum_compact_record_bytes=1024**2,maximum_expanded_record_bytes=1024**2,
            maximum_logical_bytes=16*1024**3,maximum_physical_bytes=16*1024**3,minimum_free_bytes=RESERVE)
        stage=CompleteJointIndexedTrainingV1(source,view,reference,optimizer,sources,options,history_root=OUT/'history',history_budgets=history_budgets,
            initializer_guard=initializer_guard.check,maximum_attempts=3)
        write(OUT/('rank-'+str(rank)+'-ready.json'),dict(rank=rank,actual_own_small_counts=view.actor.counts(),full_original_joint_steps=len(stage.entries)))
        if rank==0:write(OUT/'original-source-bindings.json',source.files)
        random.seed(995010+rank);np.random.seed(995010+rank);torch.manual_seed(995010+rank)
        stage.open_history();first=stage.train_next();stage.commit(OUT/'state',maximum_bytes=8*1024**2,minimum_free_bytes=RESERVE)
        second=stage.train_next()
        expected=(tensor_digest(view.named_parameters()),value_digest(optimizer.state_dict()),value_digest(capture_rng('cpu')),value_digest(second))
        stage.restore(OUT/'state');stage.open_history();replayed=stage.train_next()
        actual=(tensor_digest(view.named_parameters()),value_digest(optimizer.state_dict()),value_digest(capture_rng('cpu')),value_digest(replayed))
        if actual!=expected:raise ValueError('The complete actual next code/human/photo update,all values,Adam,private RNG and raw metric must replay exactly')
        stage.commit(OUT/'state',maximum_bytes=8*1024**2,minimum_free_bytes=RESERVE)
        reference.assert_bound(full=True);stage.guard.fixed.check(view,optimizer,full=True)
        clocks={n:int(optimizer.state[p]['step']) for n,p in view.named_selected() if p in optimizer.state}
        if any(v!=(2 if n.startswith(('actor.','vision.')) else 1) for n,v in clocks.items()):raise ValueError('Every actual native/CTC/matching/actor clock must match the full original source prefix')
        cold=None
        if rank==0:
            word=source.prepare_visual(source.metadata(0)['components']['visual'],rank=0,world=4,device='cpu')
            photo=source.prepare_visual(source.metadata(1)['components']['visual'],rank=0,world=4,device='cpu')
            def one(batch):
                width=int(batch['attention_mask'][0].sum())
                return {k:v[:1] if k=='pixel_values' else v[:1,:width] for k,v in batch.items()}
            p=min(source.metadata(1)['components']['code']['entry']['original_positions'],key=lambda p:source.code.information(p)['complete_context_tokens'])
            cold=finalize(view.actor,OUT/'standalone',source.code.weak.folder/'tokenizer.json',sources=sources,completed_updates=2,
                probes=[('actual_complete_code',source.code.batch([p])),('actual_original_photo',one(photo['native'])),('actual_original_word',one(word['native']))],
                word_pixels=word['word']['word_pixels'],maximum_logit_error=1e-5,minimum_free_bytes=RESERVE,maximum_checkpoint_bytes=8*1024**2)
        write(OUT/('rank-'+str(rank)+'-complete.json'),dict(rank=rank,actual_own_small_counts=view.actor.counts(),actual_full_original_declared_joint_steps=5362,
            actual_logical_original_prefix_updates=2,actual_physical_AdamW_calls=3,actual_first_metric=first,actual_second_metric=second,
            exact_next_complete_source_values_Adam_private_RNG_and_metric_replay=True,pre_Adam_full_records_retained_in_indexed_files=True,
            final_all_parameters_sha256=actual[0],final_Adam_sha256=actual[1],final_private_RNG_sha256=actual[2],actual_sparse_clocks=clocks,
            actual_native_cold_export=cold,full_size_GPUcapacity_training_or_model_quality_gain=False,goal_achieved=False))
        source.guard.check(full=True);initializer_guard.check(full=True)
        if torch.cuda.is_initialized():raise ValueError('The CPU source diagnostic must not acquire a GPU context')
        dist.barrier()
    finally:dist.destroy_process_group()


def main():
    if (OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.nice(0)<19
            or any(os.environ.get(k)!='1' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'])
            or shutil.disk_usage(ROOT).free<RESERVE+64*1024**2):raise ValueError('A fresh funded idle CUDA-hidden four-replica CPU diagnostic is required')
    os.sched_setscheduler(0,os.SCHED_IDLE,os.sched_param(0))
    from slm_lab.local_source_closure_v2 import python_sources
    files={str(p):sha(p) for p in python_sources(ROOT,[Path(__file__)])}
    owner=dict(pid=os.getpid(),ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19])
    OUT.mkdir();write(OUT/'plan.json',dict(owner=owner,at=datetime.now(timezone.utc).isoformat(),source_files_sha256=files,
        actual_original_schedule_preserved=5362,planned_own_small_logical_updates=2,planned_actual_AdamW_calls_including_replay=12,
        budgets_are_per_record_not_full16GiBphysical_GPU_funding=True,production_joint_training_admitted=False,goal_achieved=False))
    try:
        from torch.multiprocessing import spawn
        spawn(worker,args=(files,),nprocs=4,join=True)
        rows=[json.loads((OUT/('rank-'+str(rank)+'-complete.json')).read_text()) for rank in range(4)]
        if len({r['final_all_parameters_sha256'] for r in rows})!=1 or len({r['final_Adam_sha256'] for r in rows})!=1 or len({r['final_private_RNG_sha256'] for r in rows})!=4:
            raise ValueError('All actual peers must agree on learned values/Adam while preserving independent private RNG')
        write(OUT/'complete.json',dict(complete=True,owner=owner,at=datetime.now(timezone.utc).isoformat(),source_files_sha256=files,
            actual_original_source_bindings_sha256=sha(OUT/'original-source-bindings.json'),actual_own_small_counts=rows[0]['actual_own_small_counts'],
            actual_full_declared_original_joint_steps=5362,actual_logical_original_prefix_updates=2,actual_physical_AdamW_calls_including_replay=12,
            logical_actual_sources=dict(code=32,human_pairs=32,whole_words=64,photos=64),physical_actual_sources_including_replay=dict(code=48,human_pairs=48,whole_words=64,photos=128),
            logical_original_context_tokens=33270,logical_original_response_EOS_targets=12316,whole_CTC_UTF8_bytes=264,
            physical_context_tokens_including_replay=51309,physical_response_EOS_targets_including_replay=20462,
            exact_actual_next_source_values_Adam_all_private_RNG_full_metric_replay=True,actual_native_export=rows[0]['actual_native_cold_export'],
            actual_rank_reports_sha256={str(rank):sha(OUT/('rank-'+str(rank)+'-complete.json')) for rank in range(4)},
            full_size_GPU_capacity_training_or_model_quality_gain=False,goal_achieved=False))
    except BaseException as exc:
        write(OUT/'failure.json',dict(owner=owner,error=repr(exc),traceback=traceback.format_exc(),automatic_retry=False,goal_achieved=False));raise


if __name__=='__main__':main()
