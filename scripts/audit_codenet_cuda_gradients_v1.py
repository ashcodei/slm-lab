"""Four-GPU capped gradient-only audit alongside the unchanged live curriculum."""
from datetime import datetime, timezone, timedelta
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer
import torch
from torch import distributed as dist

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.codenet_replay_v1 import INDEX_DTYPE, family
from slm_lab.codenet_runner_v1 import FixedReductionModel, VerifiedReplayEncoder, average_gradients
from slm_lab.codenet_stage_v1 import Objective, freeze_for_code_stage, objective_gradients
from slm_lab.codenet_token_pack_v1 import CompleteCodePack, packed_encoder
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.recurrent_ablation import state_digest
from slm_lab.training import load_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def main():
    rank,world,local=(int(os.environ.get(k,v)) for k,v in [('RANK','0'),('WORLD_SIZE','1'),('LOCAL_RANK','0')])
    if world != 4 or torch.cuda.device_count() != 4:
        raise ValueError('Exactly four explicitly launched CUDA workers required')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device=torch.device('cuda',local)
    torch.cuda.set_device(device)
    if not torch.cuda.is_bf16_supported():
        raise ValueError('Actual BF16-capable CUDA device required')
    free,total=torch.cuda.mem_get_info(device)
    if free < 22*1024**3:
        raise ValueError('Keep at least22GiB initially free on every shared GPU')
    torch.cuda.set_per_process_memory_fraction(.40,device)
    torch.cuda.reset_peak_memory_stats(device)
    folder=ROOT/'reports/codenet-cuda-gradients-v1'
    if rank==0:
        if folder.exists():raise ValueError('Preserve earlier CUDA audit evidence')
        folder.mkdir()
    dist.init_process_group('nccl',timeout=timedelta(seconds=180))
    try:
        dist.barrier()
        def status(message,**fields):
            atomic_json(folder/f'rank-{rank}-status.json',{'status':message,
                'updated_at':datetime.now(timezone.utc).isoformat(),'rank':rank,'device':str(device),
                'optimizer_exists':False,'optimizer_step_taken':False,'model_saved':False,
                'dataset_admitted_to_training':False,'new_model_training_started':False,'goal_achieved':False,**fields})
        status('verifying candidate train-only inputs and own preserved source model')
        checkpoint=ROOT/'checkpoints/extended-curriculum-v1/language-pretraining-interrupted-1/step-00012000.pt'
        cpu_path=ROOT/'reports/codenet-full-model-gradients-v1/results.json'
        cpu=json.loads(cpu_path.read_text())
        old_plan=json.loads((cpu_path.parent/'plan.json').read_text())
        checkpoint_digest=old_plan['source_files_sha256'][str(checkpoint)]
        if rank==0 and file_hash(checkpoint)!=checkpoint_digest:
            raise ValueError('Protected own checkpoint differs from earlier verified source')
        dist.barrier()
        pack=CompleteCodePack(ROOT/'data/codenet-durable-pack-v1','train',purpose='audit')
        replay_folder=ROOT/'data/codenet-replay-v1'
        manifest_path=replay_folder/'manifest.json'
        manifest=json.loads(manifest_path.read_text())
        audit_path=ROOT/'reports/codenet-replay-v1-independent-audit/results.json'
        audit=json.loads(audit_path.read_text())
        if (file_hash(manifest_path)!=audit['replay_manifest_sha256']
            or audit['complete_fit_rows_verified']!=manifest['complete_fit_rows']
            or audit['source_rows_dropped_or_duplicated'] or audit['encoder_truncated_sequences']):
            raise ValueError('Completed independent replay verification required')
        index=np.load(replay_folder/'index.npy',mmap_mode='r',allow_pickle=False)
        if index.dtype!=INDEX_DTYPE:raise ValueError('Exact replay index required')
        source=Path(manifest['original_training_source'])
        paths=[Path(__file__),ROOT/'src/slm_lab/codenet_runner_v1.py',ROOT/'src/slm_lab/codenet_checkpoint_v1.py',
            ROOT/'src/slm_lab/codenet_replay_v1.py',ROOT/'src/slm_lab/codenet_stage_v1.py',
            ROOT/'src/slm_lab/codenet_replay_schedule_v1.py',ROOT/'src/slm_lab/codenet_token_pack_v1.py',
            ROOT/'src/slm_lab/training.py',ROOT/'src/slm_lab/model.py',ROOT/'src/slm_lab/recurrent_ablation.py',
            manifest_path,audit_path,cpu_path,source,pack.folder/'manifest.json',pack.folder/'tokenizer.json',
            replay_folder/'index.npy']
        bound={str(p):file_hash(p) for p in paths}
        if bound[str(source)]!=manifest['original_training_source_sha256']:
            raise ValueError('Original frozen replay training file differs')
        replay=[]
        with source.open('rb') as stream:
            for number,name in enumerate(manifest['family_names']):
                item=index[np.flatnonzero(index['family']==number)[0]]
                stream.seek(int(item['offset']));raw=stream.read(int(item['bytes']))
                if hashlib.sha256(raw).digest()!=item['row_sha256'].tobytes():
                    raise ValueError('Actual selected original replay bytes differ')
                row=json.loads(raw)
                if family(row)!=name:raise ValueError('Actual replay family/role differs')
                row.update(_replay_input_sha256=item['input_sha256'].tobytes().hex(),
                    _replay_context_tokens=int(item['context_tokens']),_replay_target_tokens=int(item['target_tokens']))
                if row.get('image'):
                    image=Path(row['image']);image=image if image.is_absolute() else source.parent/image
                    bound[str(image)]=file_hash(image)
                replay.append((name,row))
        guard=FileBindings(bound)
        if rank==0:
            atomic_json(folder/'plan.json',{'recorded_at':datetime.now(timezone.utc).isoformat(),
                'source_files_sha256':{**bound,str(checkpoint):checkpoint_digest},
                'own_parameter_count':cpu['own_parameter_count'],'checkpoint':str(checkpoint),
                'memory_fraction_per_audit_process':.40,'minimum_initial_free_bytes':22*1024**3,
                'dummy_memory_reservation':'Two FP32 tensors per trainable parameter, matching Adam moment storage; no optimizer is constructed.',
                'world_size':4,'precision':'BF16 CUDA autocast, FP32 parameters/gradients',
                'code_records':cpu['complete_code_records'],'replay_families':[name for name,row in replay],
                'objective':'Complete code mean plus0.5/11 for each original replay-family token mean;5 replay microbatches per family.',
                'shared_with_live_training':True,'standalone_training_throughput_not_measured':True,
                'optimizer_exists':False,'optimizer_step_taken':False,'model_saved':False,
                'dataset_admitted_to_training':False,'new_model_training_started':False,'goal_achieved':False})
        model,metadata=load_checkpoint(checkpoint,'cpu',training_state=False)
        if metadata['tokenizer_sha256']!=file_hash(pack.folder/'tokenizer.json'):
            raise ValueError('Exact own tokenizer required')
        before=state_digest(model.state_dict())
        if before!=cpu['before_model_state_sha256']:
            raise ValueError('Actual own preserved model state differs')
        trainable,frozen=freeze_for_code_stage(model)
        model.gradient_checkpointing=True
        model.train();model=model.to(device)
        # Reserve realistic moment-state bytes without constructing/updating an optimizer.
        dummy_moments=[torch.zeros_like(p) for p in trainable for _ in range(2)]
        dummy_bytes=sum(t.numel()*t.element_size() for t in dummy_moments)
        wrapped=FixedReductionModel(model)
        tokenizer=Tokenizer.from_file(str(pack.folder/'tokenizer.json'))
        code_encoder=packed_encoder(pack,tokenizer,model.config)
        replay_encoder=VerifiedReplayEncoder(tokenizer,model.config,source.parent,image_size=448)
        long,short=[row['local_pack_index'] for row in cpu['complete_code_records']]
        layout=[(long,long),(short,short),(long,short),(short,long)][rank]
        code_groups=[[{'packed_code_index':position}] for position in layout]
        objectives=[Objective('complete_code',code_encoder,code_groups,'response',1.)]
        objectives.extend(Objective(name,replay_encoder,[[row] for _ in range(5)],'response',.5/len(replay))
                          for name,row in replay)
        def amp():return torch.autocast(device_type='cuda',dtype=torch.bfloat16)
        guard.check()
        status('computing full own-model capped CUDA gradients with dummy moment-state memory',dummy_moment_bytes=dummy_bytes)
        torch.cuda.synchronize(device)
        started=torch.cuda.Event(enable_timing=True);finished=torch.cuda.Event(enable_timing=True)
        started.record()
        result=objective_gradients(wrapped,objectives,device,amp,world=world)
        average_gradients(wrapped,device,world)
        finished.record();torch.cuda.synchronize(device)
        if any(p.grad is not None for p in frozen) or code_encoder.truncated or replay_encoder.truncated:
            raise ValueError('Frozen gradients or source clipping in actual CUDA path')
        nonzero,norm=0,0.
        for p in trainable:
            if p.grad is None or not bool(torch.isfinite(p.grad).all()):
                raise ValueError('Every language tensor needs finite CUDA gradients')
            magnitude=float(torch.linalg.vector_norm(p.grad))
            nonzero+=magnitude>0;norm=math.hypot(norm,magnitude)
        if nonzero!=len(trainable) or not math.isfinite(norm):
            raise ValueError('Finite nonzero gradients required for every language tensor')
        peak=torch.cuda.max_memory_allocated(device)
        reserved=torch.cuda.max_memory_reserved(device)
        after=state_digest(model.state_dict())
        if after!=before:raise ValueError('Gradient-only CUDA audit changed own model weights')
        guard.check(full=True)
        record={'completed_at':datetime.now(timezone.utc).isoformat(),'rank':rank,'device':torch.cuda.get_device_name(device),
            'initial_free_bytes':free,'memory_fraction_cap':.40,'dummy_FP32_moment_bytes':dummy_bytes,
            'peak_audit_allocated_bytes':peak,'peak_audit_reserved_bytes':reserved,
            'gradient_and_reduction_milliseconds_shared_GPU':started.elapsed_time(finished),
            'finite_nonzero_language_gradient_tensors':nonzero,'frozen_gradient_tensors':0,
            'unclipped_global_gradient_l2':norm,'before_model_state_sha256':before,'after_model_state_sha256':after,
            'actual_objectives':result,'own_parameter_count':sum(p.numel() for p in model.parameters()),
            'optimizer_exists':False,'optimizer_step_taken':False,'model_saved':False,
            'shared_with_live_training':True,'actual_training_throughput_or_checkpoint_verified':False,
            'dataset_admitted_to_training':False,'new_model_training_started':False,'goal_achieved':False}
        atomic_json(folder/f'rank-{rank}-results.json',record)
        status('full own-model CUDA gradient audit finished; model weights unchanged',peak_audit_allocated_bytes=peak)
        records=[None]*world;dist.all_gather_object(records,record)
        if rank==0:
            atomic_json(folder/'results.json',{'completed_at':datetime.now(timezone.utc).isoformat(),
                'plan_sha256':file_hash(folder/'plan.json'),'workers':records,'world_size':world,
                'every_worker_full_model_state_unchanged':all(r['before_model_state_sha256']==r['after_model_state_sha256'] for r in records),
                'every_worker_all218_language_gradient_tensors_finite_and_nonzero':all(r['finite_nonzero_language_gradient_tensors']==218 for r in records),
                'actual_FP32_dummy_moment_bytes_per_worker':dummy_bytes,'all11_replay_fronts_with_actual448px_images_exercised':True,
                'optimizer_exists':False,'optimizer_step_taken':False,'model_saved':False,
                'dataset_admitted_to_training':False,'new_model_training_started':False,'goal_achieved':False,
                'scope':'Full own706M BF16/CUDA/NCCL gradient plumbing, complete targets, actual image rehearsal and capped moment-memory capacity alongside the live job. No Adam step/checkpoint, full-schedule training-throughput, resume, capability gain or source admission established.'})
        del dummy_moments
    except Exception as error:
        if folder.exists():
            atomic_json(folder/f'rank-{rank}-error.json',{'error':str(error),'rank':rank,
                'optimizer_step_taken':False,'new_model_training_started':False,'goal_achieved':False})
        raise
    finally:
        dist.destroy_process_group()


if __name__=='__main__':
    main()
