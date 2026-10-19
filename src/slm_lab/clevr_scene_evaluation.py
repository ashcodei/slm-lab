"""Free scene-description generation scored against original object attributes/counts."""
import argparse
from collections import Counter
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import re

import torch
from tokenizers import Tokenizer
from .clevr_scene_supervision import ATTRIBUTES, PROMPT
from .corpus import file_hash
from .diverse_images import atomic_json,jsonl
from .evaluation_provenance import capture,seal,pixels_sha256
from .ocr_evaluation import normalized
from .training import Encoder,load_checkpoint

ROOT=Path(__file__).resolve().parents[2]
OBJECT=re.compile(r'([0-9]+)\s*\.\s*(small|large)\s+(gray|red|blue|green|brown|purple|cyan|yellow)\s+(rubber|metal)\s+(cube|cylinder|sphere)\s*\.',re.I)
COUNTS=re.compile(r'shape counts:\s*cubes\s+([0-9]+),\s*cylinders\s+([0-9]+),\s*spheres\s+([0-9]+)\s*\.',re.I)


def parse(value):
    lines=[line.strip() for line in value.strip().splitlines() if line.strip()]
    if len(lines)<5 or lines[0].casefold()!='objects from left to right:':return None
    objects=[]
    for index,line in enumerate(lines[1:-1],1):
        match=OBJECT.fullmatch(line)
        if not match or int(match.group(1))!=index:return None
        objects.append(tuple(part.casefold() for part in match.groups()[1:]))
    counts=COUNTS.fullmatch(lines[-1])
    if not counts or not 3<=len(objects)<=10:return None
    values=tuple(int(value) for value in counts.groups())
    if any(value>10 for value in values):return None
    return {'objects':objects,'counts':values}


def score(prediction,target):
    expected=parse(target)
    if expected is None:raise ValueError('Original dense target is not a complete valid scene description')
    actual=parse(prediction)
    if actual is None:return {'valid_format':False,'full_scene_exact':False,'object_count_exact':False,'shape_counts_exact':False,
                             'ordered_attributes_accuracy':0.,'object_attribute_multiset_f1':0.,'declared_counts_consistent':False}
    first=actual['objects'];second=expected['objects'];matching=sum((Counter(first)&Counter(second)).values())
    attributes=sum(a==b for left,right in zip(first,second) for a,b in zip(left,right))
    shapes=Counter(obj[3] for obj in first);derived=tuple(shapes[name] for name in ['cube','cylinder','sphere'])
    return {'valid_format':True,'full_scene_exact':actual==expected,'object_count_exact':len(first)==len(second),
            'shape_counts_exact':actual['counts']==expected['counts'],'ordered_attributes_accuracy':attributes/(4*max(len(first),len(second))),
            'object_attribute_multiset_f1':2*matching/(len(first)+len(second)),'declared_counts_consistent':actual['counts']==derived}


def summarize(rows):
    result={'examples':len(rows)}
    for image in ['proper','blank','wrong']:
        for metric in ['valid_format','full_scene_exact','object_count_exact','shape_counts_exact','ordered_attributes_accuracy',
                       'object_attribute_multiset_f1','declared_counts_consistent']:
            result[image+'_'+metric]=sum(row[image+'_scores'][metric] for row in rows)/len(rows)
    total=sum(row['target_tokens'] for row in rows)
    result['proper_nll']=sum(row['proper_nll']*row['target_tokens'] for row in rows)/total
    for name in ['blank','wrong']:
        result[name+'_nll_gain']=sum((row[name+'_nll']-row['proper_nll'])*row['target_tokens'] for row in rows)/total
    return result


@torch.no_grad()
def evaluate(checkpoint,output,data=None,device='cuda:2',max_examples=12,max_new_tokens=256):
    data=ROOT/'data/clevr-dense-scenes-v1/valid.jsonl' if data is None else Path(data)
    files=[ROOT/name for name in capture()['files_sha256']]+[Path(__file__),ROOT/'src/slm_lab/clevr_scene_supervision.py']
    provenance=capture(files);digest=file_hash(data);manifest_path=data.parent/'manifest.json';manifest_sha=file_hash(manifest_path)
    manifest=json.loads(manifest_path.read_text())
    if data.name!='valid.jsonl' or digest!=manifest['files_sha256'][data.name]:raise ValueError('Use original finalized dense held-out scenes')
    rows=sorted(jsonl(data),key=lambda row:hashlib.sha256(row['id'].encode()).digest())[:max_examples]
    if len(rows)<2 or len({row['group_id'] for row in rows})!=len(rows):raise ValueError('Need distinct original held-out scene images')
    if any(row['prompt']!=PROMPT or parse(row['response']) is None for row in rows):raise ValueError('Original scene prompt/target changed')
    torch.set_num_threads(1);model,metadata=load_checkpoint(checkpoint,device,training_state=False);model.eval()
    tokenizer=Path(checkpoint).parent/'tokenizer.json'
    if file_hash(tokenizer)!=manifest['tokenizer_sha256'] or metadata['tokenizer_sha256']!=manifest['tokenizer_sha256']:
        raise ValueError('Dense scene tokenizer differs from our model')
    encoder=Encoder(Tokenizer.from_file(str(tokenizer)),model.config,data.parent,448)
    amp=lambda:torch.autocast('cuda',dtype=torch.bfloat16) if device.startswith('cuda') else nullcontext()
    results=[];prompt_hashes=set()
    for index,row in enumerate(rows):
        other=rows[(index+1)%len(rows)]
        if row['response']==other['response']:raise ValueError('Wrong-image control must have another scene target')
        for sample in [row,other]:
            if file_hash(Path(sample['image']))!=sample['stored_image_sha256']:raise ValueError('Original scene image changed')
        prompt=encoder.prompt(row);tokens=[encoder.bos]+encoder.tokens(prompt)
        if len(tokens)+max_new_tokens>model.config.max_text_length:raise ValueError('Complete scene prompt/decoding budget exceeds context')
        ids=torch.tensor([tokens],device=device);proper=encoder.pixels(row)[None].to(device);wrong=encoder.pixels(other)[None].to(device)
        batch=encoder.batch([row],'response',device);prompt_digest=hashlib.sha256(prompt.encode()).hexdigest();prompt_hashes.add(prompt_digest)
        record={'id':row['id'],'group_id':row['group_id'],'expected':row['response'],'prompt_sha256':prompt_digest,
                'input_pixels_sha256':pixels_sha256(proper),'wrong_pixels_sha256':pixels_sha256(wrong),'wrong_image_id':other['id'],
                'target_tokens':int((batch['labels'][:,1:]!=-100).sum())}
        for name,pixels in [('proper',proper),('blank',torch.zeros_like(proper)),('wrong',wrong)]:
            with amp():
                generated=model.generate(ids,pixels,max_new_tokens=max_new_tokens,eos_token_id=encoder.eos)
                loss=float(model(**{**batch,'pixel_values':pixels})['loss'])
            raw=encoder.tokenizer.decode(generated[0,len(tokens):].tolist());record[name+'_generated']=raw
            record[name+'_scores']=score(raw,row['response']);record[name+'_nll']=loss
        results.append(record);print('Dense scene generation evaluated',len(results),'/',len(rows),flush=True)
    if len(prompt_hashes)!=1 or len({row['blank_generated'] for row in results})!=1:
        raise ValueError('Identical blank inputs/tasks must produce identical greedy descriptions')
    by_id={row['id']:row for row in results}
    if any(row['wrong_generated']!=by_id[row['wrong_image_id']]['proper_generated'] for row in results):
        raise ValueError('Counterpart image/control generation differs under the identical task')
    if encoder.truncated:raise ValueError('Original dense labels truncated')
    report={'checkpoint_sha256':file_hash(checkpoint),'data_sha256':digest,'manifest_sha256':manifest_sha,'image_size':448,
            'max_examples':max_examples,'max_new_tokens':max_new_tokens,'overall':summarize(results),'results':results,'goal_achieved':False,
            'protocol':'Hash-selected unique original held-out images; identical prompt;448px,greedy256-token actual/blank/rotated counterpart controls; full free descriptions plus exact target likelihood.',
            'limitations':['Twelve reused rendered development scenes; not all512 scenes, real images or independent final broad performance.',
                           'Strict declared description grammar; invalid formatting receives zero content scores.',
                           'Attribute multiset/ordered partial scores alone do not establish exact object detection or compositional reasoning.']}
    seal(report,provenance,files=files,data=data,digest=digest,device=device,tokenizer_sha256=metadata['tokenizer_sha256'])
    if file_hash(manifest_path)!=manifest_sha:raise ValueError('Dense scene manifest changed during assessment')
    atomic_json(Path(output),report);return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--checkpoint',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cuda:2');args=parser.parse_args();evaluate(args.checkpoint,args.output,device=args.device)


if __name__=='__main__':main()
