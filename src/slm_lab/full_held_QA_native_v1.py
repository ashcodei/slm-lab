"""Complete original QA populations with explicit source-overlap strata.

There is no training API or model loader. A production owner supplies its exact
own native model. A completed six-scope question audit and the full native input
preflight gate construction of the real16961-case population.
"""
from collections import defaultdict
from contextlib import nullcontext
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from .corpus import file_hash
from .diverse_images import atomic_json
from .knowledge_chat_diagnostic_v1 import question_prompt
from .knowledge_evaluation import letter
from .math_corpus import final_answer,prompt as math_prompt
from .nq_open_evaluation import correct as nq_correct

FORMAT='own-full-original-MMLU-GSM-NQ-native-evaluation-v1'
BUDGETS={'MMLU-Pro':32,'GSM8K':512,'NQ-open':32}
POPULATIONS={'MMLU-Pro':12032,'GSM8K':1319,'NQ-open':3610}


def native_prompt(case):
    benchmark=case['benchmark'];row=case['original']
    if case.get('split')!='held' or case.get('optimization_forbidden') is not True:
        raise ValueError('An explicitly held original question forbidden to optimization is required')
    if benchmark=='MMLU-Pro':return question_prompt(row,'chat')
    if benchmark=='GSM8K':return 'User: '+math_prompt(row['question'])+'\nAssistant:'
    if benchmark=='NQ-open':
        original=row['source_original_question']['question']
        if row['prompt']!=original+'\nAnswer briefly.':raise ValueError('Whole unchanged original NQ question required')
        return 'User: '+row['prompt']+'\nAssistant:'
    raise ValueError('Explicit supported original benchmark required')


class FullHeldQA:
    purpose='evaluation_only'
    optimization_forbidden=True

    def __init__(self,root):
        root=Path(root).resolve();self.root=root
        folder=root/'reports/full-held-question-audit-v2'
        complete_path=folder/'complete.json'
        if not complete_path.is_file():raise ValueError('Complete six-scope original question audit required before new held neural generation')
        complete=json.loads(complete_path.read_text());plan_path=folder/'plan.json';plan=json.loads(plan_path.read_text())
        results_path=folder/'question-results.jsonl'
        scopes={'current_native','future_human','future_code','wikipedia-expanded-v1','code-search-expanded-v1','broad-mixture-v1'}
        if (complete.get('all_six_declared_scopes_completed') is not True or set(complete['scope_counts'])!=scopes
                or file_hash(plan_path)!=complete['plan_sha256'] or file_hash(results_path)!=complete['question_results_sha256']):
            raise ValueError('Every exact declared input pool and sealed question verdict required')
        outcomes={r['audit_id']:r for r in map(json.loads,results_path.open())}
        if len(outcomes)!=17561:raise ValueError('All original question exposure identities required')
        preflight_path=root/'reports/full-held-native-context-preflight-v1/complete.json';preflight=json.loads(preflight_path.read_text())
        if not preflight['all_16961_original_held_QA_inputs_fit_complete_unchanged_budgets'] or preflight['own_native_context_positions']!=4096:
            raise ValueError('Every complete original QA input and full generation budget must fit')
        paths={'MMLU-Pro':root/'data/full-held-question-pool-v2/mmlu-pro.original-test.jsonl',
            'GSM8K':root/'data/math-reasoning-v2/test.jsonl','NQ-open':root/'data/nq-human-questions-v1/valid.jsonl'}
        token_path=root/'data/tokenizer-corpus/tokenizer.json';self.tokenizer_path=token_path;self.tokenizer_sha256=file_hash(token_path)
        self.tokenizer=Tokenizer.from_file(str(token_path))
        if plan['source_files_sha256'][str(token_path)]!=self.tokenizer_sha256:
            raise ValueError('Exact same own tokenizer as the source audit required')
        input_path=root/'reports/full-held-native-context-preflight-v1/original-native-input-bindings.jsonl'
        if file_hash(input_path)!=preflight['source_files_sha256'][str(input_path)]:raise ValueError('Actual preflight native input evidence changed')
        self.preflight={(r['benchmark'],r['id']):r for r in map(json.loads,input_path.open())}
        self.cases=[]
        for benchmark,path in paths.items():
            digest=file_hash(path)
            if digest!=plan['source_files_sha256'][str(path)] or digest!=preflight['source_files_sha256'][str(path)]:
                raise ValueError('Unchanged complete original question population required')
            rows=[json.loads(s) for s in path.open()]
            if len(rows)!=POPULATIONS[benchmark] or len({str(r['id']) for r in rows})!=len(rows):
                raise ValueError('Every distinct original held question required')
            if benchmark=='GSM8K':rows.sort(key=lambda r:hashlib.sha256(r['question'].encode()).digest())
            for row in rows:
                identity=benchmark+':'+str(row['id']);exposure=outcomes[identity]
                scopes_=exposure['inspected_exact_match_scopes']
                if bool(scopes_)!=exposure['exact_question_found_in_inspected_inputs']:
                    raise ValueError('Exact question exposure flags must agree')
                self.cases.append(dict(audit_id=identity,benchmark=benchmark,original=row,
                    split='held',optimization_forbidden=True,
                    previously_sampled_for_local_evaluation=exposure['previously_sampled_for_local_evaluation'],
                    inspected_exact_match_scopes=scopes_,potential_exact_question_exposure=bool(scopes_),
                    fresh_MMLU_not_previously_sampled_and_no_exact_match=(benchmark=='MMLU-Pro' and
                        not exposure['previously_sampled_for_local_evaluation'] and not scopes_)))
        if len(self.cases)!=16961 or len({r['audit_id'] for r in self.cases})!=16961:
            raise ValueError('Every original full QA identity required')
        self.bindings={str(p):file_hash(p) for p in [complete_path,plan_path,results_path,preflight_path,input_path,token_path,*paths.values(),Path(__file__)]}
        self.identity_sha256=hashlib.sha256(json.dumps(self.cases,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def score(case,generated,choice_scores=None):
    row=case['original'];benchmark=case['benchmark']
    if benchmark=='MMLU-Pro':
        gold=row['answer_index'];n=len(row['options'])
        if type(gold) is not int or not 0<=gold<n:raise ValueError('Original ordered gold choice index required')
        prediction=letter(generated,n)
        result=dict(correct=prediction==gold,answer_parseable=prediction is not None,generated_index=prediction,
            original_gold_index=gold,chance_accuracy=1/n)
        if choice_scores is not None:
            for name in ['spaced_letter','direct_letter']:
                values=choice_scores[name]
                if len(values)!=n or any(not math.isfinite(v) for v in values):raise ValueError('Every original option needs a finite complete native letter likelihood')
                index=max(range(n),key=lambda i:values[i])
                result[name+'_likelihood_index']=index;result[name+'_likelihood_correct']=index==gold
        return result
    if benchmark=='GSM8K':
        predicted=final_answer(generated);gold=Fraction(row['answer_numerator'],row['answer_denominator'])
        return dict(correct=predicted==gold,answer_parseable=predicted is not None,
            predicted_numerator=None if predicted is None else predicted.numerator,
            predicted_denominator=None if predicted is None else predicted.denominator,
            original_gold_numerator=gold.numerator,original_gold_denominator=gold.denominator)
    if benchmark=='NQ-open':
        return dict(correct=nq_correct(generated,row['reference_answers']),answer_parseable=bool(generated.strip()),
            original_reference_answers=row['reference_answers'])
    raise ValueError('Original supported benchmark required')


@torch.inference_mode()
def evaluate_case(model,tokenizer,case,*,expected_input=None):
    parameters=list(model.parameters());device=parameters[0].device
    if (model.training or any(p.grad is not None for p in parameters)
            or tokenizer.get_vocab_size()!=model.config.vocab_size
            or [tokenizer.token_to_id(t) for t in ['<pad>','<bos>','<eos>']]!=[0,1,2]):
        raise ValueError('Exact own evaluation policy, zero gradients and native tokenizer required')
    prompt=native_prompt(case);tokens=[1,*tokenizer.encode(prompt).ids];budget=BUDGETS[case['benchmark']]
    digest=hashlib.sha256(np.asarray(tokens,dtype='<u2').tobytes()).hexdigest()
    if len(tokens)+budget>model.config.max_text_length:raise ValueError('Whole original question plus unchanged generation budget must fit')
    if expected_input is not None and (expected_input['native_input_ids_sha256']!=digest or expected_input['prompt_tokens']!=len(tokens)
            or expected_input['fixed_generation_budget']!=budget):raise ValueError('Exact already preflighted original native input required')
    ids=torch.tensor([tokens],device=device);choice_scores=None;candidate_evidence=None
    amp=lambda:torch.autocast('cuda',dtype=torch.bfloat16) if device.type=='cuda' else nullcontext()
    with amp():
        if case['benchmark']=='MMLU-Pro':
            initial=model.prefill(ids);choice_scores={};candidate_evidence={}
            for name,prefix in [('spaced_letter',' '),('direct_letter','')]:
                values=[];evidence=[]
                for index in range(len(case['original']['options'])):
                    candidate=tokenizer.encode(prefix+chr(65+index)).ids
                    if not candidate:raise ValueError('Every complete original candidate letter needs actual native tokens')
                    state=initial;logps=[]
                    for position,token in enumerate(candidate):
                        logps.append(float(state['logits'][0,-1].float().log_softmax(-1)[token]))
                        if position+1<len(candidate):state=model.decode(torch.tensor([[token]],device=device),state['cache'])
                    values.append(math.fsum(logps));evidence.append(dict(candidate_ids=candidate,per_token_logprobs=logps))
                choice_scores[name]=values;candidate_evidence[name]=evidence
            del initial,state
        generated=model.generate(ids,max_new_tokens=budget,eos_token_id=2,use_cache=True)
    if generated.ndim!=2 or len(generated)!=1 or not torch.equal(generated[0,:len(tokens)],ids[0]):
        raise ValueError('Every complete original prompt token must be retained by generation')
    actions=generated[0,len(tokens):].tolist()
    if (not 1<=len(actions)<=budget or 2 in actions[:-1]
            or any(type(t) is not int or not 0<=t<model.config.vocab_size for t in actions)):
        raise ValueError('Only actual own-vocabulary first-EOS or full-budget greedy deliveries permitted')
    text=tokenizer.decode(actions);verdict=score(case,text,choice_scores)
    return dict(audit_id=case['audit_id'],benchmark=case['benchmark'],original_id=str(case['original']['id']),
        domain=case['original'].get('domain'),original_prompt=prompt,original_native_prompt_ids=tokens,
        original_native_prompt_ids_sha256=digest,actual_generated_ids=actions,actual_generated_response=text,
        genuine_generated_EOS=actions[-1]==2,generation_budget=budget,
        choice_scores=choice_scores,candidate_token_evidence=candidate_evidence,verdict=verdict,
        previously_sampled_for_local_evaluation=case['previously_sampled_for_local_evaluation'],
        potential_exact_question_exposure=case['potential_exact_question_exposure'],
        inspected_exact_match_scopes=case['inspected_exact_match_scopes'],
        fresh_MMLU_not_previously_sampled_and_no_exact_match=case['fresh_MMLU_not_previously_sampled_and_no_exact_match'],
        inference_device_type=device.type,inference_precision='BF16_autocast' if device.type=='cuda' else 'FP32',
        optimization_forbidden=True,model_updates=0)


def verify_record(tokenizer,case,record,expected_input=None):
    prompt=native_prompt(case);tokens=[1,*tokenizer.encode(prompt).ids]
    digest=hashlib.sha256(np.asarray(tokens,dtype='<u2').tobytes()).hexdigest()
    budget=BUDGETS[case['benchmark']];actions=record['actual_generated_ids']
    if (record['audit_id']!=case['audit_id'] or record['original_id']!=str(case['original']['id'])
            or record['original_prompt']!=prompt or record['original_native_prompt_ids']!=tokens
            or record['original_native_prompt_ids_sha256']!=digest or record['generation_budget']!=budget
            or not 1<=len(actions)<=budget or 2 in actions[:-1]
            or any(type(t) is not int or not 0<=t<tokenizer.get_vocab_size() for t in actions)
            or record['actual_generated_response']!=tokenizer.decode(actions)
            or record['genuine_generated_EOS']!=(actions[-1]==2)
            or record['optimization_forbidden'] is not True or record['model_updates']!=0):
        raise ValueError('Every original complete native prompt and actual delivered-token/text pair must rederive')
    if expected_input is not None and (expected_input['native_input_ids_sha256']!=digest
            or expected_input['prompt_tokens']!=len(tokens) or expected_input['fixed_generation_budget']!=budget):
        raise ValueError('Every saved complete native input must match its original preflight')
    if case['benchmark']=='MMLU-Pro':
        for name,prefix in [('spaced_letter',' '),('direct_letter','')]:
            rows=record['candidate_token_evidence'][name]
            if len(rows)!=len(case['original']['options']):raise ValueError('Every unchanged ordered answer candidate required')
            for index,row in enumerate(rows):
                ids=tokenizer.encode(prefix+chr(65+index)).ids;logps=row['per_token_logprobs']
                if (row['candidate_ids']!=ids or len(logps)!=len(ids) or any(not math.isfinite(v) for v in logps)
                        or math.fsum(logps)!=record['choice_scores'][name][index]):
                    raise ValueError('Every complete native answer-letter score must match its retained token likelihoods')
    elif record['choice_scores'] is not None or record['candidate_token_evidence'] is not None:
        raise ValueError('Open-answer QA cannot silently use a multiple-choice answer list')
    if record['verdict']!=score(case,record['actual_generated_response'],record['choice_scores']):
        raise ValueError('Actual raw answer and unchanged original gold must reproduce every verdict')


def summarize(cases,records):
    if len(records)!=len(cases) or len({r['audit_id'] for r in records})!=len(records):raise ValueError('Exactly one actual result per original selected question required')
    originals={r['audit_id']:r for r in cases};groups=defaultdict(list)
    for record in records:
        if record['audit_id'] not in originals:raise ValueError('Unexpected original QA identity')
        case=originals[record['audit_id']]
        if record['verdict']!=score(case,record['actual_generated_response'],record['choice_scores']):
            raise ValueError('Every QA verdict must rederive from actual generated text and original gold')
        for key in ['benchmark','previously_sampled_for_local_evaluation','potential_exact_question_exposure',
                'inspected_exact_match_scopes','fresh_MMLU_not_previously_sampled_and_no_exact_match']:
            if record[key]!=case[key]:raise ValueError('Unchanged original benchmark and exposure strata required')
        groups[case['benchmark']].append(record)
    def metrics(rows):
        total=len(rows);result=dict(questions=total,correct=sum(r['verdict']['correct'] for r in rows),
            answer_parseable=sum(r['verdict']['answer_parseable'] for r in rows))
        result['accuracy']=result['correct']/total if total else None
        if rows and rows[0]['benchmark']=='MMLU-Pro':
            result.update({key:sum(r['verdict'][key] for r in rows) for key in ['spaced_letter_likelihood_correct','direct_letter_likelihood_correct']})
            result['mean_chance_accuracy']=math.fsum(r['verdict']['chance_accuracy'] for r in rows)/total
        return result
    result={}
    for benchmark,rows in groups.items():
        item=dict(all_original=metrics(rows),potential_exact_question_exposure=metrics([r for r in rows if r['potential_exact_question_exposure']]),
            no_exact_question_match_in_inspected_pools=metrics([r for r in rows if not r['potential_exact_question_exposure']]))
        if benchmark=='MMLU-Pro':
            item['fresh_not_previously_sampled_and_no_exact_match']=metrics([r for r in rows if r['fresh_MMLU_not_previously_sampled_and_no_exact_match']])
            domains=defaultdict(list)
            for row in rows:domains[row['domain']].append(row)
            item['by_original_domain']={k:dict(all_original=metrics(v),
                fresh_not_previously_sampled_and_no_exact_match=metrics([r for r in v if r['fresh_MMLU_not_previously_sampled_and_no_exact_match']]))
                for k,v in domains.items()}
        result[benchmark]=item
    return result


def evaluate_shard(model,tokenizer,population,output,*,rank,world,checkpoint_sha256,checked=lambda:None,record=lambda _:None):
    if type(population) is not FullHeldQA or type(world) is not int or not 1<=world<=4 or type(rank) is not int or not 0<=rank<world:
        raise ValueError('Complete audited population and exact declared native shard required')
    if Path(output).exists():raise ValueError('Preserve every existing native QA shard')
    if file_hash(population.root/'data/tokenizer-corpus/tokenizer.json')!=population.tokenizer_sha256:
        raise ValueError('Exact own tokenizer bytes required')
    if json.loads(tokenizer.to_str())!=json.loads(population.tokenizer_path.read_text()):
        raise ValueError('Actual complete native tokenizer configuration must match the audited bytes')
    count=sum(p.numel() for p in model.parameters())
    if not 0<count<10**9:raise ValueError('Every actually loaded student value must fit the requested under1B budget')
    selected=population.cases[rank::world];records=[]
    for case in selected:
        checked();result=evaluate_case(model,tokenizer,case,expected_input=population.preflight[(case['benchmark'],str(case['original']['id']))])
        verify_record(tokenizer,case,result,population.preflight[(case['benchmark'],str(case['original']['id']))])
        record(result);records.append(result)
    for name,digest in population.bindings.items():
        if file_hash(Path(name))!=digest:raise ValueError('Complete original QA evidence changed during generation')
    report=dict(format=FORMAT,rank=rank,world=world,checkpoint_sha256=checkpoint_sha256,
        original_population_sha256=population.identity_sha256,tokenizer_sha256=population.tokenizer_sha256,
        actual_student_parameter_values=count,inference_device_type=next(model.parameters()).device.type,
        inference_precision='BF16_autocast' if next(model.parameters()).device.type=='cuda' else 'FP32',fixed_generation_budgets=BUDGETS,
        original_selected_ids=[c['audit_id'] for c in selected],results=records,summary=summarize(selected,records),
        full_original_population_counts=POPULATIONS,source_files_sha256=population.bindings,
        original_question_no_clipping_or_skips=True,optimization_forbidden=True,
        custom_native_zero_shot_protocol_not_an_official_leaderboard_score=True,
        no_exact_match_is_not_a_semantic_independence_or_full_ancestry_certificate=True,model_updates=0,goal_achieved=False)
    atomic_json(Path(output),report);return report


def reconcile(population,paths,output):
    if type(population) is not FullHeldQA or Path(output).exists():raise ValueError('Complete real QA population and fresh reconciliation output required')
    reports=[json.loads(Path(path).read_text()) for path in paths]
    world,checkpoint=require_shards(reports)
    indexed={}
    for report in reports:
        selected=population.cases[report['rank']::world]
        selected_by_id={c['audit_id']:c for c in selected}
        if (report['format']!=FORMAT or report['world']!=world or report['checkpoint_sha256']!=checkpoint
                or report['original_population_sha256']!=population.identity_sha256 or report['tokenizer_sha256']!=population.tokenizer_sha256
                or report['source_files_sha256']!=population.bindings or report['fixed_generation_budgets']!=BUDGETS
                or any(report[k]!=reports[0][k] for k in ['actual_student_parameter_values','inference_device_type','inference_precision'])
                or report['original_selected_ids']!=[c['audit_id'] for c in selected]
                or report['summary']!=summarize(selected,report['results'])):
            raise ValueError('Every original complete QA identity, checkpoint, stratum and verdict required')
        for record in report['results']:
            if record['audit_id'] in indexed:raise ValueError('Duplicate original QA result')
            case=selected_by_id[record['audit_id']]
            verify_record(population.tokenizer,case,record,population.preflight[(case['benchmark'],str(case['original']['id']))])
            indexed[record['audit_id']]=record
    records=[indexed[c['audit_id']] for c in population.cases]
    result=dict(format=FORMAT,complete=True,checkpoint_sha256=checkpoint,original_population_sha256=population.identity_sha256,
        actual_student_parameter_values=reports[0]['actual_student_parameter_values'],
        inference_device_type=reports[0]['inference_device_type'],inference_precision=reports[0]['inference_precision'],
        fixed_generation_budgets=BUDGETS,
        actual_original_question_count=len(records),full_original_population_counts=POPULATIONS,summary=summarize(population.cases,records),
        results=records,original_question_no_clipping_or_skips=True,source_files_sha256={str(p):file_hash(p) for p in paths},
        optimization_forbidden=True,custom_native_zero_shot_protocol_not_an_official_leaderboard_score=True,
        no_exact_match_is_not_a_semantic_independence_or_full_ancestry_certificate=True,model_updates=0,goal_achieved=False)
    atomic_json(Path(output),result);return result


def require_shards(reports):
    if not reports:raise ValueError('All declared original shards required')
    world=reports[0]['world'];checkpoint=reports[0]['checkpoint_sha256']
    if (type(world) is not int or not 1<=world<=4 or len(reports)!=world
            or any(type(r['rank']) is not int for r in reports)
            or {r['rank'] for r in reports}!=set(range(world))
            or any(r['world']!=world or r['checkpoint_sha256']!=checkpoint for r in reports)):
        raise ValueError('Every distinct original QA shard from the same exact checkpoint required')
    return world,checkpoint
