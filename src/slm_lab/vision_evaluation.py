"""Held-out visual grounding: image/caption retrieval and image interventions."""
import argparse
from collections import defaultdict
from contextlib import nullcontext
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from tokenizers import Tokenizer
from .corpus import file_hash
from .data import read_jsonl,stable_id
from .training import Encoder,load_checkpoint


def retrieval(images,texts,caption_ids):
    scores=F.normalize(images.float(),dim=-1)@F.normalize(texts.float(),dim=-1).T
    positive=caption_ids[:,None]==caption_ids[None,:]
    image_correct=positive.gather(1,scores.argmax(1)[:,None]).squeeze(1)
    text_correct=positive.T.gather(1,scores.argmax(0)[:,None]).squeeze(1)
    return {'image_to_text_accuracy':float(image_correct.float().mean()),
            'text_to_image_accuracy':float(text_correct.float().mean()),
            'chance_accuracy':float(positive.float().mean()),'images':len(images)}


@torch.no_grad()
def evaluate(checkpoint,data,output,device='cuda',max_examples=128,image_size=None):
    from .evaluation_provenance import capture,seal,pixels_sha256
    provenance=capture();data_digest=file_hash(data)
    torch.set_num_threads(2);model,meta=load_checkpoint(checkpoint,device,training_state=False);model.eval()
    tokenizer_path=Path(checkpoint).parent/'tokenizer.json';tok=Tokenizer.from_file(str(tokenizer_path))
    if file_hash(tokenizer_path)!=meta['tokenizer_sha256']:raise ValueError('Tokenizer mismatch')
    rows=read_jsonl(data)
    if not rows or any(not row.get('image') for row in rows):raise ValueError('Visual evaluation needs image-bearing held-out rows')
    if image_size is None:image_size=meta.get('training_options',{}).get('image_size',model.config.image_size)
    encoder=Encoder(tok,model.config,Path(data).parent,image_size);count=min(max_examples,len(rows))
    selected=[rows[i*len(rows)//count] for i in range(count)]
    group_ids=[row['group_id'] for row in selected]
    if len(set(group_ids))!=len(group_ids):raise ValueError('Evaluation must use unique image groups')
    caption_ids=torch.tensor([int(stable_id(row['response'])[:15],16) for row in selected],device=device)
    amp=lambda:torch.autocast('cuda',dtype=torch.bfloat16) if device.startswith('cuda') else nullcontext()
    image_features=[];text_features=[];records=[];totals=defaultdict(float)
    pixels=[encoder.pixels(row) for row in selected]
    for i,row in enumerate(selected):
        # Choose a wrong image with a different ground-truth caption whenever possible.
        alternatives=[j for j in range(count) if caption_ids[j]!=caption_ids[i]]
        if not alternatives:raise ValueError('Grounding ablation needs at least two distinct captions')
        wrong=alternatives[i%len(alternatives)]
        batch=encoder.batch([row],'response',device);targets=int((batch['labels'][:,1:]!=-100).sum())
        with amp():
            proper=float(model(**batch)['loss'])
            blank=float(model(**{**batch,'pixel_values':torch.zeros_like(batch['pixel_values'])})['loss'])
            mismatched=float(model(**{**batch,'pixel_values':pixels[wrong][None].to(device)})['loss'])
            image_features.append(model.vision(batch['pixel_values']).mean(1))
            caption_batch=encoder.batch([dict(row,text=row['response'],image=None)],None,device)
            out=model(input_ids=caption_batch['input_ids'],attention_mask=caption_batch['attention_mask'],
                      return_hidden=True,return_logits=False)
            last=int(caption_batch['attention_mask'].sum())-1;text_features.append(out['hidden'][:,last])
        for key,value in [('proper',proper),('blank',blank),('mismatched',mismatched)]:totals[key]+=targets*value
        totals['targets']+=targets
        records.append({'id':row['id'],'domain':row.get('domain'),'target_tokens':targets,'proper_nll':proper,
                        'blank_nll':blank,'mismatched_nll':mismatched,'wrong_image_id':selected[wrong]['id'],
                        'input_pixels_sha256':pixels_sha256(pixels[i]),'wrong_pixels_sha256':pixels_sha256(pixels[wrong])})
    features=torch.cat(image_features);texts=torch.cat(text_features)
    retrieval_report=retrieval(features,texts,caption_ids)
    by_domain={}
    for domain in sorted({r['domain'] for r in records}):
        items=[r for r in records if r['domain']==domain];n=sum(r['target_tokens'] for r in items)
        by_domain[domain]={'images':len(items),'proper_nll':sum(r['proper_nll']*r['target_tokens'] for r in items)/n,
            'blank_nll_gain':sum((r['blank_nll']-r['proper_nll'])*r['target_tokens'] for r in items)/n,
            'mismatch_nll_gain':sum((r['mismatched_nll']-r['proper_nll'])*r['target_tokens'] for r in items)/n}
    report={'checkpoint_sha256':file_hash(checkpoint),'data_sha256':file_hash(data),'images':count,
        'image_size':image_size,'visual_tokens':model.config.image_token_count(image_size),
        'caption_retrieval':retrieval_report,'target_tokens':int(totals['targets']),
        'proper_nll':totals['proper']/totals['targets'],'blank_nll_gain':(totals['blank']-totals['proper'])/totals['targets'],
        'mismatch_nll_gain':(totals['mismatched']-totals['proper'])/totals['targets'],
        'by_domain':by_domain,'samples':records,'truncated_sequences':encoder.truncated,
        'limitations':['Paired held-out examples only; broad visual reasoning/OCR/VQA still needs independent tests.',
            'Retrieval uses the trained vision projection and our own frozen text features, not an external CLIP model.',
            'A positive intervention gain indicates image dependence, not that every caption fact is visually grounded.']}
    seal(report,provenance,data=data,digest=data_digest,device=device,tokenizer_sha256=meta['tokenizer_sha256'])
    Path(output).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='samples'},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True)
    p.add_argument('--data',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda')
    p.add_argument('--max-examples',type=int,default=128);p.add_argument('--image-size',type=int);a=p.parse_args()
    evaluate(a.checkpoint,a.data,a.output,a.device,a.max_examples,a.image_size)

if __name__=='__main__':main()
