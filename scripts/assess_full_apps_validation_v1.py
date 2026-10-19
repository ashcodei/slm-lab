"""Read-only full original validation loss in an owned idle assessment window.

Three shards cover every internal-validation program. Never changes training
inputs, chooses a stopping step, or treats likelihood as functional correctness.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import torch

import train_apps_full_language_v1 as owner
from slm_lab.apps_human_solution_view_v1 import AppsHumanSolutions
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.training import load_checkpoint


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--phase',choices=['baseline','candidate'],required=True)
    parser.add_argument('--shard',type=int,choices=[0,1,2],required=True);args=parser.parse_args()
    physical=args.shard+1
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(physical):raise ValueError('Exact idle GPU assignment required')
    plan=owner.register(args.phase+'-full-APPS-validation-shard-'+str(args.shard))
    checkpoint=owner.BASE if args.phase=='baseline' else owner.OUTPUT/'model.pt'
    folder=owner.REPORTS/'APPS-validation';folder.mkdir(exist_ok=True)
    target=folder/(args.phase+'-'+str(args.shard)+'.json')
    if target.exists():raise ValueError('Preserve earlier full validation shard')
    def checked():
        owner.ownership(plan)
        if json.loads((owner.REPORTS/'status.json').read_text()).get('phase')!=args.phase:
            raise ValueError('Owned idle validation window ended; stop without overlapping the next phase')
    checked()
    for i in [1,2,3]:
        if not (owner.REPORTS/args.phase/('suite-'+str(i)+'-complete.json')).exists():
            raise ValueError('All assigned original assessment workers must have ended first')
    source=AppsHumanSolutions(owner.SOURCE,role='validation',purpose='audit')
    checked();torch.set_num_threads(1);device=torch.device('cuda:0');torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.4,device)
    model,metadata=load_checkpoint(checkpoint,device,training_state=False);model.eval()
    if metadata['tokenizer_sha256']!=source.manifest['tokenizer_sha256']:raise ValueError('Exact own original tokenizer required')
    checkpoint_sha=file_hash(checkpoint);positions=list(range(args.shard,len(source),3))
    records=[];total_targets=total_correct=0;total_nll=0.
    with torch.no_grad():
        for position in positions:
            checked();original=source.example(position)
            if original['role']!='validation':raise ValueError('Only explicit held source rows in read-only validation')
            ids,labels=original['positive']
            if len(ids)>model.config.max_text_length:raise ValueError('Complete original held example required; no clipping')
            batch=dict(input_ids=torch.tensor([ids],device=device),labels=torch.tensor([labels],device=device))
            with torch.autocast('cuda',dtype=torch.bfloat16):out=model(**batch)
            mask=batch['labels'][:,1:]!=-100;n=int(mask.sum());loss=float(out['loss'])
            correct=int(((out['logits'][:,:-1].argmax(-1)==batch['labels'][:,1:])&mask).sum())
            records.append(dict(position=position,id=original['id'],context_tokens=len(ids),response_targets=n,mean_NLL=loss,correct_teacher_forced_targets=correct))
            total_targets+=n;total_correct+=correct;total_nll+=n*loss
            if len(records)%256==0:print('whole_original_validation_programs_scored',len(records),'/',len(positions),flush=True)
    checked();source.guard.check(full=True)
    if file_hash(checkpoint)!=checkpoint_sha:raise ValueError('Frozen assessed checkpoint changed')
    atomic_json(target,dict(completed_at=datetime.now(timezone.utc).isoformat(),phase=args.phase,shard=args.shard,shards=3,
        checkpoint_sha256=checkpoint_sha,source_admission_sha256=file_hash(owner.SOURCE/'admission.json'),
        tokenizer_sha256=metadata['tokenizer_sha256'],programs=len(records),response_targets=total_targets,
        target_NLL_sum=total_nll,token_mean_NLL=total_nll/total_targets,correct_teacher_forced_targets=total_correct,
        teacher_forced_token_accuracy=total_correct/total_targets,records=records,
        model_updates=0,held_gold_used_for_gradients_or_stopping=False,functional_correctness_measured=False,
        public_internal_validation_not_certified_final_benchmark=True,goal_achieved=False))
    print('complete_validation_shard',args.phase,args.shard,len(records),flush=True)


if __name__=='__main__':main()
