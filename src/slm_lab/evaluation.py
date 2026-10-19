"""Held-out pipeline metrics; these do not establish broad model capability."""
import argparse
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import re
import time
import torch
from tokenizers import Tokenizer
from .data import read_jsonl
from .corpus import PackedCorpus,file_hash
from .training import Encoder,load_checkpoint,forward,last_scores,response_logps
from .evaluation_provenance import capture,seal


@torch.no_grad()
def evaluate(checkpoint,data,task,output,device='cpu',threads=4,max_examples=128,reference_checkpoint=None,image_size=None):
    provenance=capture()
    digest_path=Path(data)/'corpus.json' if Path(data).is_dir() else Path(data)
    initial_data_digest=file_hash(digest_path)
    torch.set_num_threads(threads)
    model,meta=load_checkpoint(checkpoint,device,training_state=False);model.eval()
    tokenizer_path=Path(checkpoint).parent/'tokenizer.json'
    token_hash=hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
    if meta['tokenizer_sha256']!=token_hash:
        raise ValueError('Saved tokenizer checksum differs from checkpoint.')
    if image_size is None:image_size=meta.get('training_options',{}).get('image_size',model.config.image_size)
    encoder=Encoder(Tokenizer.from_file(str(tokenizer_path)),model.config,Path(data).parent,image_size)
    reference=None
    if task=='policy_preference':
        if not reference_checkpoint:raise ValueError('Policy preference evaluation needs the fixed training reference.')
        reference,reference_meta=load_checkpoint(reference_checkpoint,device,training_state=False);reference.eval()
        if reference.config!=model.config or reference_meta['tokenizer_sha256']!=token_hash:raise ValueError('Policy/reference architecture or tokenizer mismatch.')
    if max_examples<1:raise ValueError('Evaluation sample limit must be positive.')
    data_path=Path(data)
    if data_path.is_dir():
        if task!='loss':raise ValueError('Packed corpus evaluation supports token loss only.')
        rows=PackedCorpus(data_path,token_hash,model.config.max_text_length,split='valid')
        data_hash=file_hash(data_path/'corpus.json')
    else:
        rows=read_jsonl(data);data_hash=file_hash(data_path)
    count=min(max_examples,len(rows))
    indices=[i*len(rows)//count for i in range(count)]
    use_bf16=torch.device(device).type=='cuda' and torch.cuda.is_bf16_supported()
    def amp():return torch.autocast(device_type='cuda',dtype=torch.bfloat16) if use_bf16 else nullcontext()
    loss_sum=0;token_count=0;correct=0;examples=[];group_losses={};ties=0;margins=[];started=time.monotonic()
    for index in indices:
        row=rows[index]
        if task=='loss':
            batch=encoder.batch([row],None if 'text' in row or 'tokens' in row else 'response',device)
            with amp():out=model(**batch)
            n=int((batch['labels'][:,1:]!=-100).sum())
            loss_sum+=float(out['loss'])*n;token_count+=n
            for field in ['domain','language_extension','language']:
                if row.get(field):
                    key=field+':'+str(row[field]);bucket=group_losses.setdefault(key,{'loss_sum':0.,'target_tokens':0,'examples':0})
                    bucket['loss_sum']+=float(out['loss'])*n;bucket['target_tokens']+=n;bucket['examples']+=1
        elif task=='policy_preference':
            good,bad=encoder.batch([row],'chosen',device),encoder.batch([row],'rejected',device)
            with amp():
                c,r=forward(model,good),forward(model,bad);rc,rr=forward(reference,good),forward(reference,bad)
            margin=(response_logps(c['logits'],good['input_ids'],good['labels'])-
                    response_logps(r['logits'],bad['input_ids'],bad['labels'])-
                    response_logps(rc['logits'],good['input_ids'],good['labels'])+
                    response_logps(rr['logits'],bad['input_ids'],bad['labels']))
            value=float(margin);margins.append(value);ties+=int(value==0);correct+=int(value>0)
        elif task=='preference':
            good,bad=encoder.batch([row],'chosen',device),encoder.batch([row],'rejected',device)
            with amp():c,r=last_scores(forward(model,good,return_logits=False),good),last_scores(forward(model,bad,return_logits=False),bad)
            correct+=int(c.item()>r.item())
            margins.append(float(c-r));ties+=int(c.item()==r.item())
        else:
            # Fixed completion budget; no answer is included in the model input.
            body=encoder.tokens(encoder.prompt(row))[-(model.config.max_text_length-33):]
            ids=torch.tensor([[encoder.bos]+body],device=device)
            pixels=encoder.pixels(row)
            with amp():generated=model.generate(ids,None if pixels is None else pixels[None].to(device),
                max_new_tokens=32,eos_token_id=encoder.eos)
            answer=encoder.tokenizer.decode(generated[0,ids.shape[1]:].tolist())
            target=re.search(r'\b([A-D])\b',row['response'])
            prediction=re.search(r'\b([A-D])\b',answer)
            matched=bool(target and prediction and target[1]==prediction[1])
            correct+=int(matched)
            examples.append({'id':row['id'],'prediction':answer,'target':row['response'],
                             'correct':matched})
    report={'task':task,'examples':count,'available_examples':len(rows),'sample_indices':indices,'device':str(device),'bf16':use_bf16,'checkpoint_sha256':file_hash(checkpoint),
        'data_sha256':data_hash,
        'image_size':image_size,
        'feedback_sources':sorted({rows[index].get('feedback_source') for index in indices if rows[index].get('feedback_source')})
            if task in {'preference','policy_preference'} else meta.get('feedback_sources',[]),
        'checkpoint_feedback_sources':meta.get('feedback_sources',[]),
        'elapsed_seconds':time.monotonic()-started,'truncated_sequences':encoder.truncated,
        'scope':'Held-out local sample only; no frontier-comparison claim.'}
    if task=='loss':
        report.update({'target_tokens':token_count,'nll':loss_sum/token_count,
                       'perplexity':math.exp(min(700,loss_sum/token_count)),
                       'by_group':{k:{'nll':v['loss_sum']/v['target_tokens'],
                           'target_tokens':v['target_tokens'],'examples':v['examples']} for k,v in group_losses.items()}})
    else:
        report.update({'correct':correct,'accuracy':correct/count})
    if task=='policy_preference':
        report.update({'reference_checkpoint_sha256':file_hash(reference_checkpoint),'ties':ties,'mean_reference_adjusted_margin':sum(margins)/len(margins),
            'metric':'Human-chosen response has greater policy/reference log-probability improvement than rejected response; not a live human policy comparison.'})
    if task=='preference':report.update({'ties':ties,'mean_reward_margin':sum(margins)/len(margins),'critic_projection_precision':'fp32'})
    if examples:report['predictions']=examples
    seal(report,provenance,data=digest_path,digest=initial_data_digest,device=device,tokenizer_sha256=token_hash)
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    Path(output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='predictions'},indent=2))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--data',required=True)
    p.add_argument('--task',choices=['loss','preference','policy_preference','choice'],default='loss')
    p.add_argument('--output',required=True);p.add_argument('--device',default='cpu')
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--reference-checkpoint')
    p.add_argument('--image-size',type=int)
    p.add_argument('--max-examples',type=int,default=128)
    a=p.parse_args()
    evaluate(a.checkpoint,a.data,a.task,a.output,a.device,a.threads,a.max_examples,a.reference_checkpoint,a.image_size)


if __name__=='__main__':main()
