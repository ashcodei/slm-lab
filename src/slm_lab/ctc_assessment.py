"""Matched greedy/prefix CTC diagnostics on existing own-vision checkpoints."""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path

import torch

from .character_ctc import CharacterSequenceModel,from_base,batch_pixels,collapse
from .corpus import file_hash
from .ctc_decoding import prefix_search
from .diverse_images import atomic_json,jsonl
from .model import ModelConfig,VisionEncoder
from .ocr_evaluation import transcription_scores
from .visual_character_probe import state_digest


def select(rows,maximum):
    selected=[]
    for family in sorted({row['ctc_family'] for row in rows}):
        used=set()
        for row in sorted([row for row in rows if row['ctc_family']==family],key=lambda row:hashlib.sha256(row['id'].encode()).digest()):
            if row['group_id'] in used:continue
            selected.append(row);used.add(row['group_id'])
            if len(used)>=maximum:break
    return selected


@torch.no_grad()
def assess(model,rows,output,device,size,identity,beam_width=32,token_prune=16):
    model=model.to(device).eval().requires_grad_(False);before=state_digest(model)
    amp=lambda:torch.autocast('cuda',dtype=torch.bfloat16) if device.startswith('cuda') else nullcontext()
    results=[]
    for index,row in enumerate(rows):
        other=next(candidate for candidate in rows[index+1:]+rows[:index] if candidate['ctc_family']==row['ctc_family'] and candidate['group_id']!=row['group_id'] and candidate['response']!=row['response'])
        images=batch_pixels([row,other],size,device)
        with amp():logits=model(torch.cat([images[:1],torch.zeros_like(images[:1]),images[1:]],0))
        probabilities=logits.float().log_softmax(-1).cpu()
        greedy=[collapse(path) for path in logits.argmax(-1).tolist()]
        # Search consumes logits and fixed budgets only. Reference words are
        # accessed later for scoring, never as a lexicon or decoding constraint.
        searched=[prefix_search(probability,beam_width,token_prune) for probability in probabilities]
        result={'id':row['id'],'group_id':row['group_id'],'family':row['ctc_family'],'expected':row['response'],
                'greedy':greedy[0],'prefix':searched[0],'blank_prefix':searched[1],'wrong_prefix':searched[2],
                'greedy_scores':transcription_scores(row['response'],greedy[0]),
                'prefix_scores':transcription_scores(row['response'],searched[0])}
        results.append(result)
        if len(results)%16==0:print('assessed CTC decoding cases',identity,len(results),flush=True)
    def summarize(items):
        report={'examples':len(items)}
        for method in ['greedy','prefix']:
            values=[row[method+'_scores'] for row in items]
            report[method+'_exact_accuracy']=sum(value['exact_match'] for value in values)/len(values)
            report[method+'_character_error_rate']=sum(value['character_edits'] for value in values)/sum(value['reference_characters'] for value in values)
        report['prefix_blank_original_target_match']=sum(row['blank_prefix']==row['expected'] for row in items)/len(items)
        report['prefix_wrong_original_target_match']=sum(row['wrong_prefix']==row['expected'] for row in items)/len(items)
        return report
    if state_digest(model)!=before:raise ValueError('Frozen component weights changed during decoding assessment')
    report={'identity':identity,'component_state_sha256':before,'frames':model.frames,'image_size':size,
            'beam_width':beam_width,'token_prune':token_prune,'overall':summarize(results),
            'by_family':{family:summarize([row for row in results if row['family']==family]) for family in sorted({row['family'] for row in results})},
            'results':results,'goal_achieved':False,
            'protocol':'Identical unique-parent image samples; greedy path collapse and bounded prefix probability search; no lexicon, target access during search, language decoder or text repair.',
            'limitations':['A decoding diagnostic on reused development samples, not a new independent benchmark.','Any CTC gains remain separate from native language-model transcription and general image reasoning.']}
    atomic_json(output,report);del model
    if device.startswith('cuda'):torch.cuda.empty_cache()
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--data',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cuda:2');parser.add_argument('--maximum-per-family',type=int,default=64)
    args=parser.parse_args();torch.set_num_threads(1)
    if args.maximum_per_family<2:parser.error('Need multiple independent images per family')
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    data=Path(args.data);manifest=json.loads((data/'manifest.json').read_text())
    if file_hash(data/'valid.jsonl')!=manifest['files_sha256']['valid.jsonl']:raise ValueError('Reading validation changed')
    rows=select(list(jsonl(data/'valid.jsonl')),args.maximum_per_family);size=manifest['configuration']['image_size']
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=True,mmap=True)
    plan=state['plan'];source=plan['base_checkpoint']
    if file_hash(source)!=plan['base_checkpoint_sha256']:raise ValueError('Original own-model source changed')
    base,_=from_base(source,plan['frames'])
    base_report=assess(base,rows,output/'baseline.json',args.device,size,'base');del base
    vision=VisionEncoder(ModelConfig(**plan['config']))
    candidate=CharacterSequenceModel(vision,state['model']['prototypes'],plan['frames'],plan['temperature'])
    candidate.load_state_dict(state['model']);del state
    candidate_report=assess(candidate,rows,output/'candidate.json',args.device,size,'trained');del candidate
    atomic_json(output/'summary.json',{'checkpoint_sha256':file_hash(args.checkpoint),'validation_sha256':file_hash(data/'valid.jsonl'),
                'case_ids':[row['id'] for row in rows],'baseline':base_report['overall'],'candidate':candidate_report['overall'],
                'goal_achieved':False,'scope':'Matched bounded-prefix CTC diagnostic only; production native word reading remains unverified.'})


if __name__=='__main__':main()
