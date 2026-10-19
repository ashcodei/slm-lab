"""Real Adam updates on maximum admitted whole-context/response training rows."""
from collections import Counter
from datetime import datetime,timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
from tokenizers import Tokenizer
import torch

from prepare_codenet_native_contract_pack_v2 import occupied_writer_reservation
from slm_lab.code_residual_branch_v1 import RoutedCodeResidual
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.codenet_token_pack_v1 import CompleteCodePack,packed_encoder
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.recurrent_ablation import state_digest
from slm_lab.training import load_checkpoint,TrainingOutputs,supervised_gradients


ROOT=Path(__file__).resolve().parents[1]


def main():
    reports=ROOT/'reports/code-residual-max-context-v1'
    if reports.exists():raise ValueError('Preserve earlier maximum-context audit evidence')
    occupied=occupied_writer_reservation();reserve=26*1024**3-occupied['credited_bytes']+256*1024**2
    if shutil.disk_usage(ROOT).free<reserve+64*1024**2:raise ValueError('Original writer reserve/report budget required')
    device=torch.device('cuda:0');torch.cuda.set_device(device)
    if not torch.cuda.is_bf16_supported() or torch.cuda.mem_get_info(device)[0]<12*1024**3:
        raise ValueError('Initially12GiB free on shared BF16 GPU required')
    torch.cuda.set_per_process_memory_fraction(.20,device);torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True);torch.manual_seed(77719);torch.cuda.manual_seed_all(77720)
    source=ROOT/'checkpoints/extended-curriculum-v1/language-pretraining/model.pt'
    expected_source='4c05d99b0c3fc775bf89e70c6cbac9c63017eb36f67e17c738b5be268fb191ca'
    if file_hash(source)!=expected_source:raise ValueError('Exact own completed-language source required')
    pack=CompleteCodePack(ROOT/'data/codenet-durable-pack-v2','train',purpose='training')
    lengths=pack.index['length'][pack.positions].astype(np.int64)
    targets=lengths-pack.index['target_start'][pack.positions].astype(np.int64)
    choices={}
    for reason,local in [('maximum_complete_context',int(lengths.argmax())),('maximum_complete_response',int(targets.argmax()))]:
        global_position=int(pack.positions[local]);choices.setdefault(global_position,dict(local_position=local,reasons=[]))['reasons'].append(reason)
    selected=[];languages=Counter()
    with gzip.open(pack.folder/'original-rows.jsonl.gz','rt') as stream:
        for position,line in enumerate(stream):
            row=json.loads(line)
            if row['split']=='train':languages[row['language']]+=1
            if position not in choices:continue
            choice=choices[position];ids,labels=pack.sequence(choice['local_position'])
            digest=hashlib.sha256(np.asarray(ids,dtype='<u2').tobytes()).hexdigest()
            if (row['split']!='train' or row['token_pack_index']!=position or digest!=row['complete_input_ids_sha256']
                    or sum(v!=-100 for v in labels)!=row['target_tokens']):
                raise ValueError('Whole original admitted maximum-size train programme required')
            selected.append(dict(**choice,global_position=position,id=row['id'],group_id=row['group_id'],
                language=row['language'],context_tokens=len(ids),target_tokens=row['target_tokens'],full_input_ids_sha256=digest))
    if len(selected)!=len(choices) or sum(languages.values())!=len(pack):raise ValueError('Entire admitted population and maxima required')
    files=[Path(__file__),source,pack.folder/'manifest.json',pack.folder/'admission.json',
           ROOT/'src/slm_lab/code_residual_branch_v1.py',ROOT/'src/slm_lab/model.py',ROOT/'src/slm_lab/training.py',
           ROOT/'src/slm_lab/codenet_checkpoint_v1.py',ROOT/'src/slm_lab/codenet_token_pack_v1.py',
           ROOT/'src/slm_lab/recurrent_ablation.py',ROOT/'scripts/prepare_codenet_native_contract_pack_v2.py']
    bound={str(p):file_hash(p) for p in files}
    bound.update({str(pack.folder/name):h for name,h in pack.manifest['files_sha256'].items()})
    guard=FileBindings(bound);reports.mkdir()
    def status(message,**fields):
        atomic_json(reports/'status.json',dict(status=message,updated_at=datetime.now(timezone.utc).isoformat(),
                    candidate_promoted=False,goal_achieved=False,**fields))
    atomic_json(reports/'plan.json',dict(recorded_at=datetime.now(timezone.utc).isoformat(),source_files_sha256=bound,
        selected_complete_original_training_rows=selected,admitted_training_rows=len(pack),admitted_language_counts=dict(languages),
        whole_population_max_context_tokens=int(lengths.max()),whole_population_max_response_tokens=int(targets.max()),
        complete_context_quantiles=np.quantile(lengths,[0,.5,.9,.99,1]).tolist(),optimizer_updates=2,
        optimizer=dict(name='AdamW',learning_rate=3e-5,weight_decay=.01,foreach=False),batch_size=1,
        accumulation=len(selected),gradient_checkpointing=True,model_total_unique_parameters=857007616,
        maximum_GPU_process_memory_fraction=.20,remaining_original_writer_reserve_bytes=reserve,
        no_external_model_weights=True,model_saved=False,candidate_promoted=False,goal_achieved=False))
    status('loading exact own model for maximum admitted source contexts')
    base,metadata=load_checkpoint(source,'cpu',training_state=False);before=state_digest(base.state_dict())
    token=Tokenizer.from_file(str(pack.folder/'tokenizer.json'))
    if metadata['tokenizer_sha256']!=file_hash(pack.folder/'tokenizer.json'):raise ValueError('Exact source tokenizer required')
    model=RoutedCodeResidual(base,width=2048).to(device);model.code.gradient_checkpointing=True
    encoder=packed_encoder(pack,token,model.config);wrapped=TrainingOutputs(model.code,'sft')
    parameters=model.adapter_parameters();optimizer=torch.optim.AdamW(parameters,lr=3e-5,weight_decay=.01,foreach=False)
    def amp():return torch.autocast('cuda',dtype=torch.bfloat16)
    groups=[[{'packed_code_index':s['local_position']}] for s in selected];records=[]
    torch.cuda.reset_peak_memory_stats(device)
    for step in range(2):
        guard.check();model.train();optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize();started=time.perf_counter()
        loss,count=supervised_gradients(wrapped,encoder,groups,'response',device,amp)
        if count!=sum(s['target_tokens'] for s in selected) or encoder.truncated:
            raise ValueError('Complete maximum-context/response targets without clipping required')
        if any(p.grad is not None for p in model.base.parameters()):raise ValueError('Original backbone received gradients')
        if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in parameters):raise ValueError('Finite residual gradients required')
        down=sum(bool(r.down.weight.grad.abs().sum()>0) for r in model.residuals)
        up=sum(bool(r.up.weight.grad.abs().sum()>0) for r in model.residuals)
        if up!=24 or step>0 and down!=24:raise ValueError('Every residual projection must receive nonzero gradients by step2')
        norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True);optimizer.step();torch.cuda.synchronize()
        row=dict(step=step+1,token_mean_nll=float(loss),response_targets=count,shared_GPU_elapsed_seconds=time.perf_counter()-started,
                 nonzero_up_layers=up,nonzero_down_layers=down,gradient_norm_before_clipping=float(norm))
        records.append(row);print('maximum_complete_context_adam_update',row,flush=True)
        status('maximum admitted whole-context actual Adam update complete',completed_steps=step+1,latest=row)
    if any(not bool(torch.isfinite(v).all()) for state in optimizer.state.values() for v in state.values()):
        raise ValueError('All actual Adam moments/steps must remain finite')
    after=state_digest(model.base.state_dict())
    if after!=before:raise ValueError('Original general/vision/critic backbone changed')
    guard.check(full=True)
    result=dict(completed_at=datetime.now(timezone.utc).isoformat(),source_files_sha256=bound,
        model_counts=model.parameter_counts(),admitted_training_rows=len(pack),admitted_language_counts=dict(languages),
        whole_population_max_context_tokens=int(lengths.max()),whole_population_max_response_tokens=int(targets.max()),
        selected_complete_original_training_rows=selected,actual_full_model_Adam_updates=2,updates=records,
        all_original_frozen_base_tensors_unchanged=True,original_base_state_sha256=before,
        all_48_residual_gradient_and_Adam_states_finite=True,all_residual_projection_gradients_nonzero_by_second_update=True,
        actual_Adam_moment_tensors=sum(len(s) for s in optimizer.state.values()),encoder_truncated_sequences=encoder.truncated,
        peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(device),peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved(device),
        shared_GPU=True,batch_size=1,accumulation=len(selected),model_saved=False,
        sustained_whole_corpus_training_or_code_quality_assessed=False,candidate_promoted=False,goal_achieved=False,
        scope='Actual own857M batch1 BF16/FP32-Adam updates at the exact whole admitted source context and response maxima, without clipping. Shared-device sampled capacity only; no sustained throughput, multi-example padded batches, retention/routing or functional capability claim.')
    atomic_json(reports/'results.json',result);status('actual857M maximum admitted whole-context and response Adam capacity passed',completed_steps=2)
    print('actual_857M_maximum_context_Adam_capacity_passed',result['peak_GPU_allocated_bytes'],flush=True)


if __name__=='__main__':main()
