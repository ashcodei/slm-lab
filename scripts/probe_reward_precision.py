"""Matched reward-head precision probe using identical BF16 backbone activations."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from tokenizers import Tokenizer

from slm_lab.corpus import file_hash
from slm_lab.data import read_jsonl
from slm_lab.reward_gate import gate
from slm_lab.training import Encoder, last_scores, load_checkpoint


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True);parser.add_argument('--data',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--device',default='cuda:3')
    args=parser.parse_args();torch.set_num_threads(1)
    if args.device.startswith('cuda'):torch.cuda.set_device(torch.device(args.device))
    model,meta=load_checkpoint(args.checkpoint,args.device,training_state=False);model.eval()
    tokenizer_path=Path(args.checkpoint).parent/'tokenizer.json'
    if file_hash(tokenizer_path)!=meta['tokenizer_sha256']:raise ValueError('Tokenizer mismatch')
    tokenizer=Tokenizer.from_file(str(tokenizer_path))
    encoder=Encoder(tokenizer,model.config,Path(args.data).parent)
    rows=read_jsonl(args.data)
    if not rows or any(row.get('feedback_source')!='human' for row in rows):raise ValueError('Genuine human preferences required')
    records=[]
    amp=lambda:torch.autocast('cuda',dtype=torch.bfloat16) if args.device.startswith('cuda') else nullcontext()
    for i,row in enumerate(rows):
        scores={}
        for field in ['chosen','rejected']:
            batch=encoder.batch([row],field,args.device)
            inputs={key:value for key,value in batch.items() if key!='labels'}
            with amp():
                out=model(**inputs,return_hidden=True,return_logits=False)
                legacy_values=F.linear(out['hidden'],model.value_head.weight).squeeze(-1)
            scores[field+'_bf16']=float(last_scores({'values':legacy_values},batch))
            with torch.autocast(device_type=out['hidden'].device.type,enabled=False):
                values=F.linear(out['hidden'].float(),model.value_head.weight.float()).squeeze(-1)
            scores[field+'_fp32_head']=float(last_scores({'values':values},batch))
        records.append({'id':row.get('id',str(i)),**scores})
        if (i+1)%100==0:print('reward precision pairs',i+1,flush=True)
    reports={}
    for precision in ['bf16','fp32_head']:
        correct=sum(row['chosen_'+precision]>row['rejected_'+precision] for row in records)
        ties=sum(row['chosen_'+precision]==row['rejected_'+precision] for row in records)
        reports[precision]={'task':'preference','examples':len(rows),'available_examples':len(rows),
                            'correct':correct,'accuracy':correct/len(rows),'ties':ties,
                            'checkpoint_sha256':file_hash(args.checkpoint),'data_sha256':file_hash(args.data),
                            'feedback_sources':['human'],'truncated_sequences':encoder.truncated}
        reports[precision]['gate']=gate(reports[precision])
    changed=sum((row['chosen_bf16']>row['rejected_bf16'])!=(row['chosen_fp32_head']>row['rejected_fp32_head']) for row in records)
    result={'method':'Same BF16 hidden activations; original autocast scalar projection versus explicit FP32 scalar projection.',
            'reports':reports,'changed_rankings':changed,'records':records,
            'limitations':['A diagnostic numeric comparison, not fresh human evaluation of own-policy responses.',
                           'Passing an offline ranking gate does not prove on-policy reward reliability.']}
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:value for key,value in result.items() if key!='records'},indent=2))


if __name__=='__main__':main()
