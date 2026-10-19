"""Whole640-source teacher prediction decomposition; never a functional score."""
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import shutil
import string

import numpy as np
from tokenizers import Tokenizer
import torch
from torch.nn import functional as F

from slm_lab.code_repetition_inference_v1 import load_objective_policy
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.codenet_token_pack_v1 import CompleteCodePack
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.recurrent_ablation import state_digest

ROOT=Path(__file__).resolve().parents[1]
CATEGORIES=['true_EOS','whitespace_only_piece','ASCII_punctuation_only_piece','digit_only_piece','other_text_piece']
DTYPE=np.dtype([('gold','<u2'),('prediction','<u2'),('NLL','<f4'),('category','u1')])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--profile',choices=['CE_control','CE_plus_repetition_UL'],required=True)
    args=parser.parse_args();source=ROOT/'reports/code-repetition-objective-pilot-v1'
    reports=ROOT/'reports/code-repetition-objective-token-diagnostics-v1'/args.profile
    if reports.exists():raise ValueError('Preserve previous token diagnostic')
    if shutil.disk_usage(ROOT).free<16*1024**3+32*1024**2:raise ValueError('Bound16MiB reports above16GiB reserve required')
    device=torch.device('cuda:2');torch.cuda.set_device(device);torch.set_num_threads(1)
    if not torch.cuda.is_bf16_supported() or torch.cuda.mem_get_info(device)[0]<12*1024**3:raise ValueError('Initially12GiB free on BF16 diagnostic GPU required')
    torch.cuda.set_per_process_memory_fraction(.20,device)
    result_path=source/args.profile/'results.json';trained=json.loads(result_path.read_text())
    plan_path=source/'plan.json';plan=json.loads(plan_path.read_text());checkpoint=Path(trained['checkpoint'])
    base=Path(plan['original_base_checkpoint']);token_path=checkpoint.parent/'tokenizer.json'
    selected_path=ROOT/'reports/code-residual-lr-pilot-v1/selected-original-rows.json'
    selection=json.loads(selected_path.read_text());rows=selection['train']+selection['validation']
    packs={role:CompleteCodePack(ROOT/'data/codenet-durable-pack-v2',role,purpose='audit') for role in ['train','validation']}
    generation_path=source/args.profile/'generations.json'
    raw={g['id']:g for g in json.loads(generation_path.read_text())['records']}
    files=[Path(__file__),result_path,plan_path,checkpoint,base,token_path,selected_path,generation_path,
        ROOT/'src/slm_lab/code_repetition_inference_v1.py',ROOT/'src/slm_lab/model.py',
        ROOT/'src/slm_lab/training.py',packs['train'].folder/'manifest.json']
    bound={str(p):file_hash(p) for p in files}
    bound.update({str(packs['train'].folder/n):h for n,h in packs['train'].manifest['files_sha256'].items()})
    assert bound[str(checkpoint)]==trained['checkpoint_sha256']
    guard=FileBindings(bound);reports.mkdir(parents=True)
    atomic_json(reports/'plan.json',dict(created_at=datetime.now(timezone.utc).isoformat(),source_files_sha256=bound,
        profile=trained['profile'],complete_original_train_sources=512,complete_original_validation_sources=128,
        categories=CATEGORIES,category_policy='Literal decoded single BPE pieces only. Punctuation/operators and other text are not semantic correctness classes.',
        model_updates=0,primary_benchmark_modified=False,maximum_report_bytes=16*1024**2,goal_achieved=False))
    model,metadata=load_objective_policy(checkpoint,base,token_path,
        expected_checkpoint_sha256=trained['checkpoint_sha256'],expected_base_sha256=plan['original_base_checkpoint_sha256'],
        expected_plan_sha256=file_hash(plan_path),expected_starting_policy_sha256=plan['starting_policy_sha256'],
        expected_selection_sha256=plan['original_selection_sha256'],expected_profile=trained['profile'],device=str(device))
    before=state_digest(model.state_dict());tokenizer=Tokenizer.from_file(str(token_path))
    category=np.empty(32768,dtype=np.uint8)
    for i in range(32768):
        piece=tokenizer.decode([i],skip_special_tokens=False);stripped=piece.strip()
        category[i]=(0 if i==2 else 1 if piece.isspace() else 2 if stripped and all(c in string.punctuation and c!='_' for c in stripped)
            else 3 if stripped.isdigit() else 4)
    arrays=[];records=[];offset=0;summary=defaultdict(lambda:dict(tokens=0,NLL_sum=0.,top1_matches=0))
    with torch.no_grad():
        for number,row in enumerate(rows,1):
            guard.check();ids,labels=packs[row['role']].sequence(row['local_position'])
            start=next(i for i,y in enumerate(labels) if y!=-100)
            full=torch.tensor([ids],device=device);gold=torch.tensor(ids[start:],device=device)
            with torch.autocast('cuda',dtype=torch.bfloat16):out=model(full,labels=torch.tensor([labels],device=device))
            logits=out['logits'][0,start-1:-1].float()
            losses=F.cross_entropy(logits,gold,reduction='none');predictions=logits.argmax(-1)
            assert len(gold)==row['target_tokens'] and abs(float(losses.mean())-float(out['loss']))<1e-5
            data=np.empty(len(gold),dtype=DTYPE);data['gold']=gold.cpu().numpy();data['prediction']=predictions.cpu().numpy()
            data['NLL']=losses.cpu().numpy();data['category']=category[data['gold']]
            first={}
            for name,pos in [('first_response',0),('true_EOS',len(gold)-1)]:
                values=logits[pos];target=int(gold[pos]);first[name]=dict(gold=target,prediction=int(predictions[pos]),
                    NLL=float(losses[pos]),gold_strict_greater_rank=int((values>values[target]).sum())+1,
                    maximum_ties=int((values==values.max()).sum()))
            grouped={}
            for i,name in enumerate(CATEGORIES):
                selected=data[data['category']==i]
                grouped[name]=dict(tokens=len(selected),NLL_sum=float(selected['NLL'].astype(np.float64).sum()),
                    top1_matches=int((selected['gold']==selected['prediction']).sum()))
                totals=summary[(row['role'],name)]
                for key,value in grouped[name].items():totals[key]+=value
            record=dict(id=row['id'],role=row['role'],language=row['language'],offset=offset,
                target_tokens=len(gold),categories=grouped,boundaries=first)
            if row['id'] in raw:
                g=raw[row['id']];prefix=g['matching_reference_prefix_tokens']
                record['sealed_free_generation']=dict(generated_tokens=len(g['generated_token_ids']),
                    terminal_EOS=g['terminal_EOS'],exact_reference=g['exact_complete_reference_token_match'],
                    matching_prefix=prefix,
                    first_divergence_teacher_gold_NLL=float(losses[prefix]) if prefix<len(gold) else None,
                    first_divergence_teacher_argmax=int(predictions[prefix]) if prefix<len(gold) else None,
                    scope='Teacher full-forward BF16 predictions can differ from cached greedy tokens near numeric ties; this is not a decoder-bug test.')
            arrays.append(data);records.append(record);offset+=len(data)
            del out,logits,full,gold,losses,predictions
            if number%32==0:
                atomic_json(reports/'status.json',dict(status='decomposing all complete source teacher targets',completed=number,planned=640,
                    updated_at=datetime.now(timezone.utc).isoformat(),goal_achieved=False))
    guard.check(full=True);after=state_digest(model.state_dict());assert before==after
    flat=np.concatenate(arrays);np.savez_compressed(reports/'per-target-predictions.npz',targets=flat)
    atomic_json(reports/'records.json',dict(records=records))
    totals={role:{name:dict(values,mean_NLL=values['NLL_sum']/values['tokens'] if values['tokens'] else None,
        top1_accuracy=values['top1_matches']/values['tokens'] if values['tokens'] else None)
        for (r,name),values in summary.items() if r==role} for role in ['train','validation']}
    for role in ['train','validation']:
        data=[v for (r,k),v in summary.items() if r==role]
        mean=sum(v['NLL_sum'] for v in data)/sum(v['tokens'] for v in data)
        assert abs(mean-trained[role]['token_mean_NLL'])<1e-5
    assert sum(p.stat().st_size for p in reports.iterdir() if p.is_file())<=16*1024**2
    atomic_json(reports/'results.json',dict(completed_at=datetime.now(timezone.utc).isoformat(),profile=trained['profile'],
        records=640,complete_target_tokens=len(flat),literal_piece_categories=totals,
        records_sha256=file_hash(reports/'records.json'),targets_sha256=file_hash(reports/'per-target-predictions.npz'),
        policy_state_sha256_before=before,policy_state_sha256_after=after,model_updates=0,
        scope='Teacher probability/top1 breakdown on all original512 train and128 validation source programs, including true EOS. Literal BPE piece categories are not algorithm/semantic correctness; source precision and earlier-exposure qualifications remain.',
        primary_benchmark_modified=False,candidate_promoted=False,goal_achieved=False))
    print(json.dumps(dict(complete_source_prediction_diagnostic=True,profile=trained['profile'],targets=len(flat),report=str(reports))))


if __name__=='__main__':main()
