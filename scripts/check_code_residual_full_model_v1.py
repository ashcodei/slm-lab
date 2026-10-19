"""Two bounded Adam updates on the actual own model; prove frozen-base isolation."""
from dataclasses import asdict
from datetime import datetime,timezone
import gc
import gzip
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from tokenizers import Tokenizer
import torch

from prepare_codenet_native_contract_pack_v2 import occupied_writer_reservation
from slm_lab.code_residual_branch_v1 import RoutedCodeResidual
from slm_lab.codenet_checkpoint_v1 import FileBindings,atomic_checkpoint
from slm_lab.codenet_replay_v1 import FullReplayEncoder
from slm_lab.codenet_token_pack_v1 import CompleteCodePack,packed_encoder
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.model import ModelConfig
from slm_lab.recurrent_ablation import state_digest
from slm_lab.training import load_checkpoint,TrainingOutputs,supervised_gradients

ROOT=Path(__file__).resolve().parents[1]


def main():
    output=ROOT/'checkpoints/code-residual-readiness-v1';reports=ROOT/'reports/code-residual-readiness-v1'
    if output.exists() or reports.exists():raise ValueError('Preserve every earlier readiness/update artifact')
    occupied=occupied_writer_reservation();reserve=26*1024**3-occupied['credited_bytes']+256*1024**2
    if shutil.disk_usage(ROOT).free<reserve+1024**3+64*1024**2:
        raise ValueError('Whole adapter artifact/reports above remaining original writer reserve required')
    device=torch.device('cuda:0');torch.cuda.set_device(device)
    if not torch.cuda.is_bf16_supported() or torch.cuda.mem_get_info(device)[0]<12*1024**3:
        raise ValueError('At least12GiB initially free on the one shared BF16 GPU required')
    torch.cuda.set_per_process_memory_fraction(.20,device);torch.set_num_threads(1)
    torch.manual_seed(220419);torch.cuda.manual_seed_all(220419)
    source=ROOT/'checkpoints/extended-curriculum-v1/language-pretraining/model.pt'
    expected_source='4c05d99b0c3fc775bf89e70c6cbac9c63017eb36f67e17c738b5be268fb191ca'
    if file_hash(source)!=expected_source:raise ValueError('Exact own completed-language source policy required')
    pack=CompleteCodePack(ROOT/'data/codenet-durable-pack-v2','train',purpose='training')
    selected=[];groups=set();languages=set()
    with gzip.open(pack.folder/'original-rows.jsonl.gz','rt') as stream:
        for line in stream:
            row=json.loads(line);position=row['token_pack_index'];item=pack.index[position]
            if (row['language'] not in {'C++','Python'} or row['language'] in languages or row['group_id'] in groups
                    or int(item['role'])!=0 or not 512<=int(item['length'])<=1024
                    or int(item['length'])-int(item['target_start'])<64):continue
            local=int(np.searchsorted(pack.positions,position))
            ids,labels=pack.sequence(local)
            if (row['split']!='train' or row['target_tokens']!=sum(v!=-100 for v in labels)
                    or hashlib.sha256(np.asarray(ids,dtype='<u2').tobytes()).hexdigest()!=row['complete_input_ids_sha256']):
                raise ValueError('Exact complete admitted original training programme required')
            selected.append(dict(local_position=local,global_position=position,id=row['id'],group_id=row['group_id'],
                language=row['language'],context_tokens=len(ids),target_tokens=row['target_tokens'],
                full_input_ids_sha256=row['complete_input_ids_sha256']))
            groups.add(row['group_id']);languages.add(row['language'])
            if len(selected)==2:break
    if languages!={'C++','Python'}:raise ValueError('Two distinct complete original training problem/language controls required')
    joint=ROOT/'data/extended-curriculum-v1/joint.train.jsonl';token_path=pack.folder/'tokenizer.json'
    token=Tokenizer.from_file(str(token_path));config=ModelConfig.load(ROOT/'configs/model.json')
    encoder=FullReplayEncoder(token,config,joint.parent,448);controls=[];kinds=set()
    for line in joint.open():
        row=json.loads(line);kind='image' if row.get('image') else 'text'
        if kind in kinds or kind=='image' and row.get('task')!='caption':continue
        try:ids,_=encoder.sequence(row,'response')
        except ValueError:continue
        if len(ids)>512:continue
        controls.append((kind,row));kinds.add(kind)
        if len(kinds)==2:break
    if kinds!={'text','image'}:raise ValueError('Original text and actual448px photo invariance controls required')
    files=[Path(__file__),source,token_path,joint,ROOT/'configs/model.json',
        ROOT/'src/slm_lab/code_residual_branch_v1.py',ROOT/'src/slm_lab/model.py',ROOT/'src/slm_lab/training.py',
        ROOT/'src/slm_lab/codenet_checkpoint_v1.py',ROOT/'src/slm_lab/codenet_token_pack_v1.py',ROOT/'src/slm_lab/codenet_replay_v1.py',
        ROOT/'src/slm_lab/recurrent_ablation.py',ROOT/'scripts/prepare_codenet_native_contract_pack_v2.py',pack.folder/'manifest.json',pack.folder/'admission.json']
    bound={str(p):file_hash(p) for p in files}
    bound.update({str(pack.folder/name):h for name,h in pack.manifest['files_sha256'].items()})
    for kind,row in controls:
        if row.get('image'):
            image=Path(row['image']);image=(image if image.is_absolute() else joint.parent/image).resolve()
            bound[str(image)]=file_hash(image)
            if bound[str(image)]!=row['stored_image_sha256']:raise ValueError('Original photo control bytes changed')
    guard=FileBindings(bound);output.mkdir();reports.mkdir()
    def status(message,**fields):
        atomic_json(reports/'status.json',dict(status=message,updated_at=datetime.now(timezone.utc).isoformat(),
            candidate_promoted=False,goal_achieved=False,**fields))
    status('loading actual own frozen backbone and code residuals')
    base,metadata=load_checkpoint(source,'cpu',training_state=False)
    if (metadata.get('stage')!='pretrain' or metadata.get('steps')!=17746 or metadata.get('planned_steps')!=17746
            or metadata.get('world_size')!=4 or metadata['tokenizer_sha256']!=file_hash(token_path)):
        raise ValueError('Exact own completed language metadata/tokenizer required')
    base_before=state_digest(base.state_dict());model=RoutedCodeResidual(base,width=2048).to(device)
    model.code.gradient_checkpointing=True;model.eval();model.assert_partition()
    counts=model.parameter_counts()
    if counts!=dict(total_unique=857007616,frozen_base=706012672,trainable_code=150994944):
        raise ValueError('Actual unique source/residual parameter budget differs')
    code_encoder=packed_encoder(pack,token,config)
    batches=[encoder.batch([row],'response',device) for kind,row in controls]
    def amp():return torch.autocast('cuda',dtype=torch.bfloat16)
    def logits(batch,route):
        with torch.no_grad(),amp():
            return model(batch['input_ids'],pixel_values=batch['pixel_values'],attention_mask=batch['attention_mask'],route=route)['logits'].cpu()
    general_before=[logits(batch,'general') for batch in batches]
    initial=[]
    for batch,original in zip(batches,general_before):
        actual=logits(batch,'code');delta=(actual.float()-original.float()).abs().max().item()
        initial.append(dict(maximum_abs_difference=delta,all_argmax_equal=bool(torch.equal(actual.argmax(-1),original.argmax(-1)))))
        torch.testing.assert_close(actual,original,rtol=0,atol=0)
    control_ids=torch.tensor([pack.sequence(selected[0]['local_position'])[0][:24]],device=device)
    with torch.no_grad():
        general_cache=model.base.prefill(control_ids[:,:16])['cache'];stale_code_cache=model.code.prefill(control_ids[:,:16])['cache']
    atomic_json(reports/'plan.json',dict(recorded_at=datetime.now(timezone.utc).isoformat(),source_files_sha256=bound,
        own_base_checkpoint=str(source),own_base_checkpoint_sha256=expected_source,model_counts=counts,
        residual_width=2048,selected_complete_original_training_rows=selected,optimizer_steps=2,
        original_training_control_ids=[dict(kind=kind,id=row['id'],group_id=row.get('group_id')) for kind,row in controls],
        general_and_code_routes_explicit=True,automatic_routing_quality_unverified=True,
        cuda_device='cuda:0',maximum_process_memory_fraction=.20,maximum_artifact_bytes=1024**3,
        occupied_verified_writer=occupied,remaining_writer_reserve_bytes=reserve,
        no_new_external_model_weights=True,benchmark_or_capability_gain_claimed=False,goal_achieved=False))
    # Two actual target-token-normalized Adam updates; no general-route weights enter the optimizer.
    trainable=model.adapter_parameters();optimizer=torch.optim.AdamW(trainable,lr=3e-5,weight_decay=.01,foreach=False)
    wrapped=TrainingOutputs(model.code,'sft');records=[]
    groups_of_rows=[[{'packed_code_index':s['local_position']}] for s in selected]
    torch.cuda.reset_peak_memory_stats(device)
    for step in range(2):
        guard.check();model.train();optimizer.zero_grad(set_to_none=True)
        loss,targets=supervised_gradients(wrapped,code_encoder,groups_of_rows,'response',device,amp,world=1)
        if code_encoder.truncated:raise ValueError('No original programme clipping allowed')
        if any(p.grad is not None for p in model.base.parameters()):raise ValueError('Frozen own general backbone received gradients')
        if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in trainable):raise ValueError('Finite residual gradients required')
        nonzero_down=sum(bool(r.down.weight.grad.abs().sum()>0) for r in model.residuals)
        nonzero_up=sum(bool(r.up.weight.grad.abs().sum()>0) for r in model.residuals)
        if nonzero_up!=len(model.residuals) or (step>0 and nonzero_down!=len(model.residuals)):
            raise ValueError('Every residual projection must receive nonzero gradients after zero initialization opens')
        norm=torch.nn.utils.clip_grad_norm_(trainable,1.,error_if_nonfinite=True);optimizer.step();model.assert_partition()
        records.append(dict(step=step+1,token_mean_nll=float(loss),actual_complete_response_targets=targets,
            nonzero_down_gradient_layers=nonzero_down,nonzero_up_gradient_layers=nonzero_up,
            gradient_norm_before_clipping=float(norm)))
        status('actual own-model residual Adam update completed',completed_steps=step+1,latest=records[-1]);print('actual_residual_adam_update',step+1,records[-1],flush=True)
    model.eval();optimizer.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
    after=state_digest(model.base.state_dict())
    if after!=base_before:raise ValueError('Any original general/vision/critic tensor changed')
    preserved=[]
    for batch,original in zip(batches,general_before):
        actual=logits(batch,'general');preserved.append(bool(torch.equal(actual,original)))
        torch.testing.assert_close(actual,original,rtol=0,atol=0)
    stale_rejected=False
    try:model.code.decode(control_ids[:,16:17],stale_code_cache)
    except ValueError as error:
        stale_rejected='parameters changed' in str(error)
    if not stale_rejected:raise ValueError('Optimizer updates must invalidate code KV cache')
    with torch.no_grad():
        general_next=model.base.decode(control_ids[:,16:17],general_cache)['logits']
        general_full=model.base(control_ids[:,:17])['logits'][:,-1:]
        code_state=model.code.prefill(control_ids[:,:16]);code_next=model.code.decode(control_ids[:,16:17],code_state['cache'])['logits']
        code_full=model.code(control_ids[:,:17])['logits'][:,-1:]
    torch.testing.assert_close(general_next,general_full,rtol=1e-4,atol=1e-4)
    torch.testing.assert_close(code_next,code_full,rtol=1e-4,atol=1e-4)
    changed=not torch.equal(logits(batches[0],'code'),general_before[0])
    if not changed:raise ValueError('Actual residual updates must change code-route outputs')
    state=model.adapter_state();path=output/'adapter.pt'
    atomic_checkpoint(path,dict(format='own-code-residual-readiness-v1',base_checkpoint_sha256=expected_source,
        residual_width=2048,base_config=asdict(config),residuals=state,completed_steps=2,
        trainable='code_residuals_only',model_counts=counts,source_manifest_sha256=file_hash(pack.folder/'manifest.json'),
        role='train',optimizer_state_serialized=False,candidate_promoted=False,goal_achieved=False),1024**3,reserve)
    restored=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    if any(not torch.equal(restored['residuals'][k],v) for k,v in state.items()):raise ValueError('Every updated adapter tensor must survive serialization')
    guard.check(full=True)
    result=dict(completed_at=datetime.now(timezone.utc).isoformat(),source_files_sha256=bound,model_counts=counts,
        actual_own_model_GPU_Adam_steps=2,selected_complete_original_training_rows=selected,updates=records,
        original_backbone_state_sha256_before=base_before,original_backbone_state_sha256_after=after,
        all_original706M_general_vision_critic_tensors_unchanged=True,initial_code_general_BF16_controls=initial,
        sampled_general_text_and448px_image_logits_bitwise_unchanged=all(preserved),code_outputs_changed=True,
        code_KV_invalidated_after_updates=True,general_KV_valid_after_code_updates=True,
        trained_code_cache_maximum_FP32_abs_difference=float((code_next-code_full).abs().max()),
        peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(device),peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved(device),
        adapter_checkpoint=str(path),adapter_checkpoint_sha256=file_hash(path),adapter_checkpoint_bytes=path.stat().st_size,
        all_updated_adapter_tensors_roundtripped=True,full_training_optimizer_restart_unverified=True,
        encoder_truncated_sequences=code_encoder.truncated,shared_with_original_four_GPU_run=True,
        actual_model_functional_or_general_knowledge_gain_assessed=False,candidate_promoted=False,goal_achieved=False,
        scope='Actual857M own-backbone/residual graph, two bounded train-only complete-programme Adam updates and exact original-base state/output isolation. Two programme controls do not prove coding skill, all-round retention under automatic routing, sustained training, final-base compatibility or frontier capability.')
    atomic_json(reports/'results.json',result);status('actual own-model code-residual update/isolation readiness complete; capability unverified',completed_steps=2)
    print('actual_857M_code_residual_readiness_complete',counts,flush=True)


if __name__=='__main__':main()
