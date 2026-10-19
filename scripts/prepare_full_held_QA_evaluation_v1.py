"""Wait for the exact source auditor, then seal all real native QA cases.

No neural model, generated answer or optimizer is created. Full input lengths,
source-overlap strata, shard assignments and reference-format scoring controls
are verified against all16961 original questions before assessment is enabled.
"""
from collections import Counter,defaultdict
from datetime import datetime,timezone
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/full-held-QA-evaluation-v1-preparation'
AUDITOR=ROOT/'reports/full-held-question-audit-v2'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists():raise ValueError('Fresh CPU-only full native QA preparation required')
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    OUT.mkdir();started=time.monotonic();pid=os.getpid()
    fields=(Path('/proc')/str(pid)/'stat').read_text().split(') ',1)[1].split()
    owner=dict(pid=pid,ticks=fields[19],CUDA_VISIBLE_DEVICES='')
    parent_plan=json.loads((AUDITOR/'plan.json').read_text());audit_owner=parent_plan['owner']
    files=[Path(__file__),AUDITOR/'plan.json',ROOT/'src/slm_lab/full_held_QA_native_v1.py',
        ROOT/'tests/test_full_held_QA_native_v1.py',ROOT/'reports/full-held-QA-native-v1-final-tests.log',
        ROOT/'reports/full-held-native-context-preflight-v1/complete.json']
    bindings={str(p):file_hash(p) for p in files}
    atomic_json(OUT/'plan.json',dict(at=datetime.now(timezone.utc).isoformat(),owner=owner,exact_auditor_owner=audit_owner,
        source_files_sha256=bindings,planned_original_QA_questions=16961,neural_generations=0,model_updates=0,
        no_held_generation_before_complete_source_audit=True,automatic_retry=False,goal_achieved=False))
    try:
        while not (AUDITOR/'complete.json').exists():
            if (AUDITOR/'failure.json').exists():raise ValueError('The exact required source audit failed; no fresh QA admission')
            path=Path('/proc')/str(audit_owner['pid'])/'stat'
            if not path.exists():raise ValueError('Required source auditor is absent without a completed audit')
            current=path.read_text().split(') ',1)[1].split()
            if current[19]!=audit_owner['ticks'] or current[0] in ['Z','X']:
                raise ValueError('Required exact source auditor is terminal without complete evidence')
            atomic_json(OUT/'waiting.json',dict(at=datetime.now(timezone.utc).isoformat(),owner=owner,
                verified_live_exact_auditor=dict(pid=audit_owner['pid'],ticks=current[19],state=current[0]),
                elapsed_seconds=time.monotonic()-started,neural_generations=0,model_updates=0))
            time.sleep(10)
        import numpy as np
        import torch
        from slm_lab.full_held_QA_native_v1 import FullHeldQA,BUDGETS,native_prompt,score
        torch.set_num_threads(1)
        population=FullHeldQA(ROOT);cases=population.cases
        registry=[];counts=Counter();matched=Counter();fresh=Counter();shards=defaultdict(list)
        for position,case in enumerate(cases):
            benchmark=case['benchmark'];row=case['original'];prompt=native_prompt(case)
            ids=[1,*population.tokenizer.encode(prompt).ids];budget=BUDGETS[benchmark]
            digest=hashlib.sha256(np.asarray(ids,dtype='<u2').tobytes()).hexdigest()
            expected=population.preflight[(benchmark,str(row['id']))]
            if (digest!=expected['native_input_ids_sha256'] or len(ids)!=expected['prompt_tokens']
                    or budget!=expected['fixed_generation_budget'] or len(ids)+budget>4096):
                raise ValueError('Every original full native input must retain its preflighted bytes and complete budget')
            if benchmark=='MMLU-Pro':reference_format=chr(65+row['answer_index'])
            elif benchmark=='GSM8K':reference_format='#### '+str(Fraction(row['answer_numerator'],row['answer_denominator']))
            else:reference_format=row['reference_answers'][0]
            if not score(case,reference_format)['correct'] or score(case,'')['correct']:
                raise ValueError('Reference-format positive and blank negative scoring controls must pass for every original QA row')
            record=dict(audit_id=case['audit_id'],benchmark=benchmark,id=str(row['id']),domain=row.get('domain'),
                original_native_prompt_ids_sha256=digest,prompt_tokens=len(ids),generation_budget=budget,
                potential_exact_question_exposure=case['potential_exact_question_exposure'],
                inspected_exact_match_scopes=case['inspected_exact_match_scopes'],
                previously_sampled_for_local_evaluation=case['previously_sampled_for_local_evaluation'],
                fresh_MMLU_not_previously_sampled_and_no_exact_match=case['fresh_MMLU_not_previously_sampled_and_no_exact_match'],
                original_four_GPU_shard=position%4,optimization_forbidden=True)
            registry.append(record);counts[benchmark]+=1;matched[benchmark]+=int(case['potential_exact_question_exposure'])
            if case['fresh_MMLU_not_previously_sampled_and_no_exact_match']:fresh[row['domain']]+=1
            shards[position%4].append(case['audit_id'])
        if (dict(counts)!={'MMLU-Pro':12032,'GSM8K':1319,'NQ-open':3610}
                or len({i for values in shards.values() for i in values})!=16961 or sum(map(len,shards.values()))!=16961):
            raise ValueError('The complete original question population and every exact four-GPU shard required')
        registry_path=OUT/'original-native-case-registry.jsonl'
        registry_path.write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in registry))
        atomic_json(OUT/'four-GPU-original-shards.json',{str(k):v for k,v in shards.items()})
        for name,digest in bindings.items():
            if file_hash(Path(name))!=digest:raise ValueError('The bound native evaluator or original proof changed while waiting')
        report=dict(at=datetime.now(timezone.utc).isoformat(),complete=True,owner=owner,
            original_population_sha256=population.identity_sha256,original_benchmark_populations=dict(counts),
            exact_question_matches_in_any_inspected_input_pool=dict(matched),
            fresh_MMLU_not_previously_sampled_no_exact_match=sum(fresh.values()),fresh_MMLU_by_domain=dict(fresh),
            all16961_whole_native_prompt_ID_hashes_lengths_and_budgets_rederived=True,
            all16961_reference_format_positive_and_blank_negative_score_controls_passed=True,
            original_four_GPU_shard_counts={str(k):len(v) for k,v in shards.items()},
            all_original_question_ids_exactly_once_across_shards=True,
            source_audit_complete_sha256=file_hash(AUDITOR/'complete.json'),native_evaluator_sha256=file_hash(ROOT/'src/slm_lab/full_held_QA_native_v1.py'),
            source_files_sha256={**population.bindings,**bindings,str(registry_path):file_hash(registry_path),
                str(OUT/'four-GPU-original-shards.json'):file_hash(OUT/'four-GPU-original-shards.json')},
            evaluation_inputs_prepared_but_no_neural_assessments_completed=True,
            no_source_independence_certificate_or_official_leaderboard_score=True,
            no_neural_network_constructed=True,neural_generations=0,model_updates=0,candidate_promoted=False,goal_achieved=False)
        atomic_json(OUT/'complete.json',report);print(json.dumps({k:report[k] for k in ['original_benchmark_populations',
            'exact_question_matches_in_any_inspected_input_pool','fresh_MMLU_not_previously_sampled_no_exact_match','original_four_GPU_shard_counts']}),flush=True)
    except BaseException as error:
        atomic_json(OUT/'failure.json',dict(owner=owner,type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),
            automatic_retry=False,neural_generations=0,model_updates=0,goal_achieved=False));raise


if __name__=='__main__':main()
