"""A supported subset of sanitized MBPP, with explicit exclusions and provenance."""
import argparse
import ast
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from .code_quality import execute, instruction_code
from .corpus import file_hash
from .diverse_images import atomic_json, jsonl
from .training import load_checkpoint
from .verified_code import assertion_cases, canonical_code
from .evaluation_provenance import capture, seal


def prepare(cache,training,output):
    cache,training,output=map(Path,[cache,training,output]);output.mkdir(parents=True,exist_ok=True)
    meta=json.loads((cache/'sources.json').read_text())
    record=next(item for item in meta['files'] if item['path']=='sanitized-mbpp.json')
    if file_hash(cache/record['path'])!=record['sha256']:raise ValueError('Pinned benchmark source changed')
    train_meta=json.loads((training/'manifest.json').read_text())
    if file_hash(training/'train.jsonl')!=train_meta['train_sha256']:raise ValueError('Verified training view changed')
    train_hashes={row['original_program_sha256'] for row in jsonl(training/'train.jsonl')}
    original=[row for row in json.loads((cache/'sanitized-mbpp.json').read_text()) if 11<=row['task_id']<=510]
    rows=[];excluded=[]
    for row in original:
        try:
            name,cases=assertion_cases(row['test_list'])
            code='\n'.join(row.get('test_imports',[]))+'\n'+row['code']
            digest=hashlib.sha256(canonical_code(code).encode()).hexdigest()
            plain_digest=hashlib.sha256(canonical_code(row['code']).encode()).hexdigest()
            if digest in train_hashes or plain_digest in train_hashes:raise ValueError('Exact normalized reference occurs in the verified training view')
            if len(cases)<2:raise ValueError('Fewer than two supported literal assertions')
            result=execute({'language':'Python','code':code,'function':name,'cases':cases,'stress':[]})
            if not result.get('functional_passed'):
                excluded.append({'task_id':row['task_id'],'reason':'Reference unsupported or did not pass original assertions','worker':result});continue
            function=next(node for node in ast.parse(code).body if isinstance(node,ast.FunctionDef) and node.name==name)
            rows.append({'id':'sanitized-mbpp:'+str(row['task_id']),'task_id':row['task_id'],
                         'function':name,'signature':'def '+name+'('+ast.unparse(function.args)+'):',
                         'specification':row['prompt'],'cases':cases,'reference_sha256':digest,
                         'source_url':record['url'],'license':'CC-BY-4.0'})
        except (ValueError,SyntaxError,TypeError,StopIteration) as error:
            excluded.append({'task_id':row['task_id'],'reason':str(error)})
    if not rows:raise ValueError('No supported held-out problems retained')
    (output/'questions.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=True)+'\n' for row in rows))
    (output/'excluded.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=True)+'\n' for row in excluded))
    manifest={'source':record,'revision':meta['revision'],'official_sanitized_test_problems':len(original),
              'supported_problems':len(rows),'excluded_problems':len(excluded),'test_task_id_range':[11,510],
              'verified_training_manifest_sha256':file_hash(training/'manifest.json'),
              'questions_sha256':file_hash(output/'questions.jsonl'),
              'limitations':['Restricted Python runtime subset selected by reference eligibility, not model performance.',
                             'This is not the full standardized MBPP benchmark or its original few-shot protocol.',
                             'Earlier pretraining corpora are not fully audited for benchmark contamination.',
                             'Only published correctness cases here; separate local checks assess efficiency.']}
    atomic_json(output/'manifest.json',manifest);return manifest


@torch.no_grad()
def evaluate(checkpoint,data,output,device='cuda',max_new_tokens=512):
    provenance=capture();data_digest=file_hash(data)
    torch.set_num_threads(1);model,meta=load_checkpoint(checkpoint,device,training_state=False);model.eval()
    tokenizer_path=Path(checkpoint).parent/'tokenizer.json'
    if file_hash(tokenizer_path)!=meta['tokenizer_sha256']:raise ValueError('Tokenizer mismatch')
    tokenizer=Tokenizer.from_file(str(tokenizer_path));rows=list(jsonl(data))
    manifest=json.loads((Path(data).parent/'manifest.json').read_text())
    if file_hash(data)!=manifest['questions_sha256']:raise ValueError('Prepared benchmark questions changed')
    if not rows:raise ValueError('No supported benchmark questions')
    amp=lambda:torch.autocast('cuda',dtype=torch.bfloat16) if device.startswith('cuda') else nullcontext()
    results=[]
    for row in rows:
        prompt=f"User: Write a Python function with signature {row['signature']} {row['specification']}\nUse built-in Python types; only public math imports are available. Return only complete code.\nAssistant:"
        ids=torch.tensor([[tokenizer.token_to_id('<bos>')]+tokenizer.encode(prompt).ids],device=device)
        if ids.shape[1]+max_new_tokens>model.config.max_text_length:raise ValueError('Benchmark prompt exceeds context budget')
        with amp():tokens=model.generate(ids,max_new_tokens=max_new_tokens,eos_token_id=tokenizer.token_to_id('<eos>'))
        raw=tokenizer.decode(tokens[0,ids.shape[1]:].tolist());code=instruction_code(raw)
        result=execute({'language':'Python','code':code,'function':row['function'],'cases':row['cases'],'stress':[]})
        results.append({'id':row['id'],'task_id':row['task_id'],'raw_response':raw,'code':code,'execution':result})
        if len(results)%25==0:print('public programming problems evaluated',len(results),flush=True)
    report={'checkpoint_sha256':file_hash(checkpoint),'data_sha256':file_hash(data),
            'official_sanitized_test_problems':manifest['official_sanitized_test_problems'],
            'supported_problems':len(rows),'excluded_problems':manifest['excluded_problems'],
            'syntax_rate_on_supported_subset':sum(row['execution'].get('syntax_valid',False) for row in results)/len(rows),
            'pass_at_1_on_supported_subset':sum(row['execution'].get('functional_passed',False) for row in results)/len(rows),
            'protocol':'Zero-shot native instruction, fixed greedy generation, constrained Python execution; only specification/signature shown to model.',
            'generation_budget':max_new_tokens,'results':results,'limitations':manifest['limitations']}
    seal(report,provenance,data=data,digest=data_digest,device=device,tokenizer_sha256=meta['tokenizer_sha256'])
    Path(output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:value for key,value in report.items() if key!='results'},indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--source-cache');parser.add_argument('--verified-training')
    parser.add_argument('--checkpoint');parser.add_argument('--data');parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cuda');parser.add_argument('--max-new-tokens',type=int,default=512)
    args=parser.parse_args()
    if args.prepare:
        if not args.source_cache or not args.verified_training:parser.error('Preparation needs source cache and frozen verified training view')
        print(json.dumps(prepare(args.source_cache,args.verified_training,args.output),indent=2))
    else:
        if not args.checkpoint or not args.data:parser.error('Evaluation needs checkpoint and prepared questions')
        evaluate(args.checkpoint,args.data,args.output,args.device,args.max_new_tokens)


if __name__=='__main__':main()
