"""Held-out image reading: exact transcription, character/word error, and image ablation."""
import argparse
from collections import defaultdict
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import unicodedata
import torch
from tokenizers import Tokenizer
from .corpus import file_hash
from .data import read_jsonl
from .training import Encoder,load_checkpoint

TASKS={'word_ocr','region_ocr','document_ocr','document_region_ocr'}


def normalized(text):return ' '.join(unicodedata.normalize('NFKC',text).split())


def distance(first,second):
    previous=list(range(len(second)+1))
    for i,a in enumerate(first,1):
        current=[i]
        for j,b in enumerate(second,1):
            current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+int(a!=b)))
        previous=current
    return previous[-1]


def transcription_scores(expected,generated):
    expected=normalized(expected);generated=normalized(generated)
    if not expected:raise ValueError('OCR reference is empty')
    return {'exact_match':expected==generated,'case_insensitive_exact_match':expected.casefold()==generated.casefold(),
            'character_edits':distance(expected,generated),'reference_characters':len(expected),
            'word_edits':distance(expected.split(),generated.split()),'reference_words':len(expected.split())}


def select(rows,count):
    groups=defaultdict(list)
    for row in rows:
        if row.get('task') in TASKS:groups[row['task']].append(row)
    chosen=[]
    for task,records in sorted(groups.items()):
        records.sort(key=lambda r:hashlib.sha256(r['id'].encode()).digest())
        seen=set();unique=[]
        for row in records:
            if row['group_id'] in seen:continue
            unique.append(row);seen.add(row['group_id'])
        chosen.extend(unique[:max(1,count//len(groups))])
    return chosen[:count]


def summarize(rows):
    chars=sum(r['reference_characters'] for r in rows);words=sum(r['reference_words'] for r in rows)
    targets=sum(r['target_tokens'] for r in rows)
    return {'examples':len(rows),'original_images':len({r['group_id'] for r in rows}),
            'exact_match_accuracy':sum(r['exact_match'] for r in rows)/len(rows),
            'case_insensitive_exact_match_accuracy':sum(r['case_insensitive_exact_match'] for r in rows)/len(rows),
            'character_error_rate':sum(r['character_edits'] for r in rows)/chars,
            'word_error_rate':sum(r['word_edits'] for r in rows)/words,
            'blank_image_nll_gain':sum((r['blank_nll']-r['proper_nll'])*r['target_tokens'] for r in rows)/targets}


@torch.no_grad()
def evaluate(checkpoint,data,output,device='cuda',max_examples=256,max_new_tokens=128,image_size=None):
    from .evaluation_provenance import capture,seal,pixels_sha256
    provenance=capture();data_digest=file_hash(data)
    torch.set_num_threads(2);model,meta=load_checkpoint(checkpoint,device,training_state=False);model.eval()
    tokenizer_path=Path(checkpoint).parent/'tokenizer.json'
    if file_hash(tokenizer_path)!=meta['tokenizer_sha256']:raise ValueError('Tokenizer mismatch')
    if image_size is None:image_size=meta.get('training_options',{}).get('image_size',model.config.image_size)
    tokenizer=Tokenizer.from_file(str(tokenizer_path));encoder=Encoder(tokenizer,model.config,Path(data).parent,image_size)
    rows=select(read_jsonl(data),max_examples)
    if not rows:raise ValueError('No OCR evaluation examples found')
    amp=lambda:torch.autocast('cuda',dtype=torch.bfloat16) if device.startswith('cuda') else nullcontext()
    results=[]
    for row in rows:
        body=encoder.tokens(encoder.prompt(row))
        if len(body)+max_new_tokens+1>model.config.max_text_length:raise ValueError('OCR prompt exceeds context budget')
        ids=torch.tensor([[encoder.bos]+body],device=device);pixels=encoder.pixels(row)[None].to(device)
        with amp():
            generated=model.generate(ids,pixels,max_new_tokens=max_new_tokens,eos_token_id=encoder.eos)
            batch=encoder.batch([row],'response',device)
            proper=float(model(**batch)['loss'])
            blank=float(model(**{**batch,'pixel_values':torch.zeros_like(pixels)})['loss'])
        response=tokenizer.decode(generated[0,ids.shape[1]:].tolist())
        result={**transcription_scores(row['response'],response),'id':row['id'],'group_id':row['group_id'],
                'task':row['task'],'expected':row['response'],'generated':response,'input_pixels_sha256':pixels_sha256(pixels),
                'proper_nll':proper,'blank_nll':blank,'target_tokens':int((batch['labels'][:,1:]!=-100).sum())}
        results.append(result)
        if len(results)%32==0:print('OCR evaluated',len(results),flush=True)
    report={'checkpoint_sha256':file_hash(checkpoint),'data_sha256':file_hash(data),
            'max_new_tokens':max_new_tokens,
            'image_size':image_size,'visual_tokens':model.config.image_token_count(image_size),
            'overall':summarize(results),'by_task':{task:summarize([r for r in results if r['task']==task]) for task in sorted({r['task'] for r in results})},
            'protocol':'Greedy cached generation; fixed 128-token default; NFKC and collapsed whitespace, case-sensitive primary exact match.',
            'results':results,'truncated_sequences':encoder.truncated,
            'limitations':['Bounded held-out public-source examples; not a standard OCR leaderboard or independent arbitrary-document test.',
                           'Crops and whole-page/scene reading are reported separately; dense small text can exceed the recorded input resolution.',
                           'Error rates can exceed one when extra text is hallucinated; they are not clipped.',
                           'Document text-layer labels can contain extraction errors; annotation quality is not exhaustively audited.']}
    seal(report,provenance,data=data,digest=data_digest,device=device,tokenizer_sha256=meta['tokenizer_sha256'])
    Path(output).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='results'},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True);p.add_argument('--data',required=True)
    p.add_argument('--output',required=True);p.add_argument('--device',default='cuda');p.add_argument('--max-examples',type=int,default=256)
    p.add_argument('--max-new-tokens',type=int,default=128);p.add_argument('--image-size',type=int);a=p.parse_args()
    if min(a.max_examples,a.max_new_tokens)<1:p.error('Evaluation limits must be positive')
    evaluate(a.checkpoint,a.data,a.output,a.device,a.max_examples,a.max_new_tokens,a.image_size)

if __name__=='__main__':main()
