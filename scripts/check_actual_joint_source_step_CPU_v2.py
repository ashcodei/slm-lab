"""One full real public-source joint batch on four small own CPU replicas.

Exercise the actual3630th scheduled batch, not its earlier untrained prefix:
all16code sources,16human pairs and the original whole-word singleton with
three truly empty visual ranks. This is a disposable integration diagnostic,
not full-size capacity, curriculum training, alignment or a quality result.
"""
from datetime import datetime,timedelta,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import traceback

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/actual-joint-source-step-CPU-v2'
RESERVE=40*1024**3


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path,value):
    raw=(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
    if len(raw)>32*1024**2 or shutil.disk_usage(ROOT).free-len(raw)<RESERVE:raise ValueError('The entire actual source diagnostic must retain the40GiBshared reserve')
    pending=Path(path).with_name(Path(path).name+'.pending')
    with pending.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    pending.replace(path)


def worker(rank):
    import numpy as np
    import random
    import torch
    from torch import distributed as dist
    from slm_lab.codenet_checkpoint_v1 import capture_rng
    from slm_lab.complete_joint_objective_bridge_v1 import train_step
    from slm_lab.complete_joint_training_data_v1 import CompleteJointTrainingDataV1
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
    dist.init_process_group('gloo',init_method='file://'+str(OUT/'rendezvous'),rank=rank,world_size=4,timeout=timedelta(minutes=20))
    try:
        config=ModelConfig(vocab_size=32768,hidden_size=32,intermediate_size=64,num_layers=2,num_heads=4,num_kv_heads=2,
            max_text_length=4096,vision_hidden_size=16,vision_intermediate_size=32,vision_layers=1,vision_heads=4,
            image_size=224,patch_size=16,pool_size=2)
        source=CompleteJointTrainingDataV1(ROOT,config,word_expansion=5)
        own=install(OwnDenseAssistant(new_backbone(config,993301),reward_hidden=8,word_temporal=4,pointing_rank=8,
            router_hidden=8,detector_hidden=4,word_expansion=5)).eval()
        view=SharedJointActorVisionV1(OwnDenseActor(own,rank=2,seed=993302)).eval()
        reference=FrozenOwnTextReferenceV1(own,sources={'complete_admitted_data_sha256':source.identity_sha256},tokenizer_sha256=source.tokenizer_sha256)
        optimizer=torch.optim.AdamW([p for _,p in view.named_selected()],lr=1e-4,eps=1e-5,weight_decay=.01,foreach=False)
        guard=JointReplicaGuardV1(view,optimizer)
        metadata=source.metadata(3629);parts=metadata['components']
        if (set(parts)!={'code','human','visual'} or parts['visual']['entry']['family']!='words'
                or parts['visual']['entry']['rank_source_counts']!=[1,0,0,0]
                or [len(parts[n]['entry']['original_positions']) for n in ['code','human','visual']]!=[16,16,1]):
            raise ValueError('The exact complete actual singleton-tail joint source batch must remain')
        write(OUT/('rank-'+str(rank)+'-ready.json'),dict(rank=rank,source_identity_sha256=source.identity_sha256,actual_small_counts=view.actor.counts()))
        if rank==0:write(OUT/'original-source-bindings.json',source.files)
        options=dict(maximum_examples=1,maximum_padded_tokens=16384,gradient_bucket_values=4096,gradient_clip=1.,human_weight=.2,visual_weight=.3,
            CTC_weight=1.,native_word_CE_weight=1.,matching_weight=.1,temperature=.07,correction_weight=.1,
            DPO_beta=.1,chosen_CE_weight=.1,project_conflicts=True,precision='bf16')
        random.seed(993310+rank);np.random.seed(993310+rank);torch.manual_seed(993310+rank)
        before=tensor_digest(view.named_parameters());initial_RNG=value_digest(capture_rng('cpu'))
        entry,metric=train_step(source,3629,view,reference,optimizer,guard,options=options)
        after=tensor_digest(view.named_parameters());guard.fixed.check(view,optimizer,full=True);reference.assert_bound(full=True)
        if before==after or value_digest(capture_rng('cpu'))!=initial_RNG or torch.cuda.is_initialized():raise ValueError('One deterministic actual update must change selected weights only and use no GPU')
        clocks={n:int(optimizer.state[p]['step']) if p in optimizer.state else 0 for n,p in view.named_selected()}
        if any(v!=(0 if n.startswith('pointing.') else 1) for n,v in clocks.items()):raise ValueError('Every truly active native/CTC/actor leaf must get exactly one Adam step')
        exported=None
        if rank==0:
            packet=source.prepare_visual(parts['visual'],rank=0,world=4,device='cpu')
            code_position=min(parts['code']['entry']['original_positions'],key=lambda p:source.code.information(p)['complete_context_tokens'])
            exported=finalize(view.actor,OUT/'standalone',source.code.weak.folder/'tokenizer.json',
                sources={'complete_admitted_data_sha256':source.identity_sha256,'actual_scheduled_entry_sha256':value_digest(entry)},completed_updates=1,
                probes=[('actual_complete_code',source.code.batch([code_position])),('actual_complete_native_word',packet['native'])],
                word_pixels=packet['word']['word_pixels'],maximum_logit_error=1e-5,minimum_free_bytes=RESERVE,maximum_checkpoint_bytes=8*1024**2)
        write(OUT/('rank-'+str(rank)+'-complete.json'),dict(rank=rank,precision='CPU_bf16_forward_FP32_master_parameters',
            actual_small_counts=view.actor.counts(),actual_full_scheduled_batch_number=3630,actual_joint_logical_updates=1,
            actual_joint_physical_AdamW_calls=1,preceding3629steps_not_trained_in_this_diagnostic=True,
            all_actual_code_sources=16,all_actual_public_human_pairs=16,all_actual_whole_words=1,
            original_complete_context_tokens=8319+5120+32,original_response_EOS_targets=5037+1708+17,
            complete_whole_CTC_UTF8_bytes=16,CTC_frames=140,actual_global_entry=entry,actual_complete_global_metric=metric,
            final_all_parameters_sha256=after,final_Adam_sha256=value_digest(optimizer.state_dict()),private_RNG_sha256=initial_RNG,
            all_fixed_values_reference_and_private_RNG_unchanged=True,actual_standalone_export=exported,
            full_size_GPUcapacity_training_or_model_quality_proved=False,goal_achieved=False))
        source.guard.check(full=True);dist.barrier()
    finally:dist.destroy_process_group()


def main():
    if (OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.nice(0)<19
            or any(os.environ.get(k)!='1' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'])
            or shutil.disk_usage(ROOT).free<RESERVE+64*1024**2):raise ValueError('A fresh funded idle CUDA-hidden actual-source CPU diagnostic is required')
    os.sched_setscheduler(0,os.SCHED_IDLE,os.sched_param(0))
    from slm_lab.local_source_closure_v2 import python_sources
    paths=python_sources(ROOT,[Path(__file__)])
    files={str(p):sha(p) for p in paths}
    owner=dict(pid=os.getpid(),ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19])
    OUT.mkdir();write(OUT/'plan.json',dict(owner=owner,at=datetime.now(timezone.utc).isoformat(),source_files_sha256=files,
        planned_small_model_logical_updates=1,planned_physical_AdamW_calls=4,production_joint_training_admitted=False,goal_achieved=False))
    try:
        from torch.multiprocessing import spawn
        spawn(worker,args=(),nprocs=4,join=True)
        rows=[json.loads((OUT/('rank-'+str(rank)+'-complete.json')).read_text()) for rank in range(4)]
        if (len({r['final_all_parameters_sha256'] for r in rows})!=1 or len({r['final_Adam_sha256'] for r in rows})!=1
                or len({r['private_RNG_sha256'] for r in rows})!=4):raise ValueError('Every actual replica must agree on learned weights/Adam and retain its own generator')
        write(OUT/'complete.json',dict(complete=True,owner=owner,at=datetime.now(timezone.utc).isoformat(),source_files_sha256=files,
            original_source_bindings_sha256=sha(OUT/'original-source-bindings.json'),actual_small_model_counts=rows[0]['actual_small_counts'],
            actual_full_source_batch_number=3630,actual_logical_updates=1,actual_physical_AdamW_calls=4,
            actual_rank_report_sha256={str(rank):sha(OUT/('rank-'+str(rank)+'-complete.json')) for rank in range(4)},
            actual_code_sources=16,actual_human_pairs=16,actual_whole_word_sources=1,actual_empty_visual_ranks=3,
            all_complete_context_tokens=13471,all_response_EOS_targets=6762,all_whole_CTC_bytes=16,CTC_frames=140,
            actual_complete_public_source_gradients_and_one_joint_Adam_boundary=True,standalone_cold_export=rows[0]['actual_standalone_export'],
            preceding3629steps_not_trained=True,full_size_GPUcapacity_training_or_quality_gain=False,goal_achieved=False))
    except BaseException as exc:
        write(OUT/'failure.json',dict(owner=owner,error=repr(exc),traceback=traceback.format_exc(),automatic_retry=False,goal_achieved=False));raise


if __name__=='__main__':main()
