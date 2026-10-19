"""All-row native math generation, sharded without changing per-question rules.

CPU and CUDA precision are distinct protocols. Each shard uses the original
prompts, greedy cached generation and exact rational final-answer parser.
"""
from contextlib import nullcontext
from datetime import datetime,timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from .corpus import file_hash
from .diverse_images import atomic_json,jsonl
from .evaluation_provenance import capture,seal
from .knowledge_evaluation import interval
from .math_corpus import final_answer,prompt
from .training import load_checkpoint


def load_source(data,expected_count=1319):
    data=Path(data);manifest_path=data.parent/'manifest.json'
    manifest=json.loads(manifest_path.read_text())
    if (data.name!='test.jsonl' or file_hash(data)!=manifest['files_sha256']['test.jsonl']
            or manifest['official_test_examples']!=expected_count):
        raise ValueError('Exact complete original native math test source required')
    rows=list(jsonl(data))
    if len(rows)!=expected_count or len({r['id'] for r in rows})!=expected_count:
        raise ValueError('Every distinct original test question required')
    rows.sort(key=lambda r:hashlib.sha256(r['question'].encode()).digest())
    return rows,manifest


@torch.no_grad()
def evaluate_shard(checkpoint,data,output,*,shard,shards,device='cuda:0',max_new_tokens=512,
        expected_count=1319,checked=lambda:None):
    if (any(type(v) is not int for v in [shard,shards,max_new_tokens]) or not 1<=shards<=16
            or not 0<=shard<shards or max_new_tokens<1):
        raise ValueError('Explicit valid full-source shard and generation budget required')
    output=Path(output);journal=output.with_suffix('.jsonl')
    if output.exists() or journal.exists():raise ValueError('Preserve earlier math generations; no implicit restart')
    rows,manifest=load_source(data,expected_count)
    provenance=capture();own_sha=file_hash(__file__);data_sha=file_hash(data);checkpoint_sha=file_hash(checkpoint)
    checked();torch.set_num_threads(1)
    model,meta=load_checkpoint(checkpoint,device,training_state=False);model.eval()
    token_path=Path(checkpoint).parent/'tokenizer.json'
    if file_hash(token_path)!=meta['tokenizer_sha256']:raise ValueError('Exact own native math tokenizer required')
    tokenizer=Tokenizer.from_file(str(token_path))
    amp=lambda:torch.autocast('cuda',dtype=torch.bfloat16) if str(device).startswith('cuda') else nullcontext()
    records=[]
    with journal.open('x') as stream:
        for position in range(shard,len(rows),shards):
            checked();row=rows[position]
            text='User: '+prompt(row['question'])+'\nAssistant:'
            prefix=[tokenizer.token_to_id('<bos>')]+tokenizer.encode(text).ids
            if len(prefix)+max_new_tokens>model.config.max_text_length:
                raise ValueError('Complete native math prompt plus fixed generation must fit; no clipping')
            ids=torch.tensor([prefix],device=device)
            with amp():tokens=model.generate(ids,max_new_tokens=max_new_tokens,
                eos_token_id=tokenizer.token_to_id('<eos>'),use_cache=True)
            body=tokens[0,len(prefix):].tolist();raw=tokenizer.decode(body);answer=final_answer(raw)
            gold=Fraction(row['answer_numerator'],row['answer_denominator'])
            result=dict(position=position,id=row['id'],raw_response=raw,generation_token_ids=body,
                answer_numerator=answer.numerator if answer is not None else None,
                answer_denominator=answer.denominator if answer is not None else None,
                expected_numerator=gold.numerator,expected_denominator=gold.denominator,
                correct=answer==gold,format_compliant=answer is not None,context_tokens=len(prefix),
                prefix_input_ids_sha256=hashlib.sha256(json.dumps(prefix,separators=(',',':')).encode()).hexdigest())
            records.append(result);stream.write(json.dumps(result,separators=(',',':'))+'\n');stream.flush()
            if len(records)%10==0:print('native_math_shard',shard,'evaluated',len(records),'/',len(rows[shard::shards]),flush=True)
    checked()
    if file_hash(checkpoint)!=checkpoint_sha or file_hash(__file__)!=own_sha:
        raise ValueError('Assessed fixed own checkpoint or shard implementation changed')
    report=dict(completed_at=datetime.now(timezone.utc).isoformat(),checkpoint_sha256=checkpoint_sha,
        data_sha256=data_sha,manifest_sha256=file_hash(Path(data).parent/'manifest.json'),
        official_test_examples=len(rows),evaluated_examples=len(records),shard=shard,shards=shards,
        generation_budget=max_new_tokens,decoding='greedy cached, temperature0',
        native_prompt_and_numeric_scoring_unchanged=True,shard_implementation_sha256=own_sha,
        journal_sha256=file_hash(journal),correct=sum(r['correct'] for r in records),
        format_compliant=sum(r['format_compliant'] for r in records),model_updates=0,goal_achieved=False)
    seal(report,provenance,data=data,digest=data_sha,device=str(device),tokenizer_sha256=meta['tokenizer_sha256'])
    atomic_json(output,report)
    return report


def reconcile(paths,data,tokenizer_path,output,*,expected_count=1319):
    output=Path(output)
    if output.exists():raise ValueError('Preserve previous full math reconciliation')
    rows,manifest=load_source(data,expected_count)
    reports=[json.loads(Path(p).read_text()) for p in paths]
    if (not reports or sorted(r['shard'] for r in reports)!=list(range(len(reports)))
            or any(r['shards']!=len(reports) for r in reports)):
        raise ValueError('Every original complete math shard exactly once required')
    for key in ['checkpoint_sha256','data_sha256','manifest_sha256','official_test_examples','generation_budget',
            'decoding','shard_implementation_sha256','inference_device_type','inference_precision',
            'tokenizer_sha256','evaluation_provenance']:
        if any(r[key]!=reports[0][key] for r in reports):raise ValueError('All math shard protocols/policy/source bindings must match: '+key)
    if (reports[0]['data_sha256']!=file_hash(data) or reports[0]['manifest_sha256']!=file_hash(Path(data).parent/'manifest.json')
            or reports[0]['tokenizer_sha256']!=file_hash(tokenizer_path)):
        raise ValueError('Actual unchanged complete source and own tokenizer required')
    tokenizer=Tokenizer.from_file(str(tokenizer_path));records=[]
    for path,certificate in zip(paths,reports):
        journal=Path(path).with_suffix('.jsonl')
        if file_hash(journal)!=certificate['journal_sha256']:raise ValueError('Actual full generation journal changed')
        batch=list(map(json.loads,journal.read_text().splitlines()))
        if ([r['position'] for r in batch]!=list(range(certificate['shard'],len(rows),len(reports)))
                or len(batch)!=certificate['evaluated_examples']):raise ValueError('Every assigned original question position exactly once required')
        if sum(r['correct'] for r in batch)!=certificate['correct']:raise ValueError('Actual shard result count differs')
        records.extend(batch)
    records.sort(key=lambda r:r['position'])
    if [r['position'] for r in records]!=list(range(len(rows))):raise ValueError('Full original math coverage required')
    for actual,row in zip(records,rows):
        prefix=[tokenizer.token_to_id('<bos>')]+tokenizer.encode('User: '+prompt(row['question'])+'\nAssistant:').ids
        gold=Fraction(row['answer_numerator'],row['answer_denominator']);answer=final_answer(actual['raw_response'])
        expected=dict(id=row['id'],expected_numerator=gold.numerator,expected_denominator=gold.denominator,
            correct=answer==gold,format_compliant=answer is not None,context_tokens=len(prefix),
            answer_numerator=answer.numerator if answer is not None else None,
            answer_denominator=answer.denominator if answer is not None else None,
            prefix_input_ids_sha256=hashlib.sha256(json.dumps(prefix,separators=(',',':')).encode()).hexdigest())
        if any(actual[k]!=v for k,v in expected.items()):raise ValueError('Every native original prompt/ID/gold/answer/score must reconstruct')
        if (not 1<=len(actual['generation_token_ids'])<=reports[0]['generation_budget']
                or tokenizer.decode(actual['generation_token_ids'])!=actual['raw_response']):
            raise ValueError('Actual complete generated tokens and raw answer text must match')
    correct=sum(r['correct'] for r in records);total=len(records)
    result=dict(completed_at=datetime.now(timezone.utc).isoformat(),checkpoint_sha256=reports[0]['checkpoint_sha256'],
        data_sha256=file_hash(data),manifest_sha256=file_hash(Path(data).parent/'manifest.json'),
        official_test_examples=total,evaluated_examples=total,full_official_test=True,correct=correct,
        numeric_exact_accuracy=correct/total,accuracy_95ci=interval(correct,total),
        format_compliance=sum(r['format_compliant'] for r in records)/total,generation_budget=reports[0]['generation_budget'],
        protocol='Original native zero-shot User/Assistant, greedy cached generation, no tools/calculator/repair; final complete #### numeric line, exact rational comparison. Full stable question-hash order, sharded across devices.',
        inference_device_type=reports[0]['inference_device_type'],inference_precision=reports[0]['inference_precision'],
        tokenizer_sha256=reports[0]['tokenizer_sha256'],evaluation_provenance=reports[0]['evaluation_provenance'],
        all_original_IDs_prompts_generated_tokens_numeric_answers_and_scores_reconciled=True,
        source_or_response_clipping=False,model_updates=0,goal_achieved=False,
        shard_reports_sha256={str(p):file_hash(p) for p in paths},results=records,
        limitations=manifest['limitations']+['Numeric answer agreement does not prove generated reasoning is valid.',
            'CPU/FP32 and CUDA/BF16 runs are distinct precision protocols; compare like with like.'])
    atomic_json(output,result)
    return result
