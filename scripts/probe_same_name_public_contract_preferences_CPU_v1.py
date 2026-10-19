"""All qualified same-name/different-contract public program pairs.

Both implementations are correct for their own qualified finite contracts.
The question is whether one is wrong under the other's explicit task. This
prepares candidate machine preference data; it performs no model training and
does not automatically admit a new optimization corpus.
"""
import ast
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import gzip
import json
import os
from pathlib import Path

from tokenizers import Tokenizer
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.contract_source_contrasts_v2 import encode
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.local_source_closure_v2 import python_sources
from slm_lab.public_function_contract_census_v2 import family,identity,observe
from slm_lab.public_function_rollout_observation_v1 import observe as public_observe

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/same-name-public-contract-preferences-CPU-v1'
DATA=ROOT/'data/contract-qualified-public-direct-admitted-v3'
ORIGINAL=ROOT/'data/direct-algorithm-function-contracts-admitted-v1'
RECEIPT=ROOT/'reports/public-contract-census-v3-readback-admission/complete.json'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists():raise ValueError('Fresh CUDA-hidden complete pair probe required')
    done=json.loads(RECEIPT.read_text());files=dict(done['source_files_sha256'])
    files.update({str(p):file_hash(p) for p in [RECEIPT,*python_sources(ROOT,[Path(__file__)])]});guard=FileBindings(files)
    rows=list(map(json.loads,gzip.open(ORIGINAL/'original-rows.jsonl.gz','rt')));by={r['id']:r for r in rows}
    qualified=list(map(json.loads,gzip.open(DATA/'original-rows.jsonl.gz','rt')))
    source=[by[r['original_source_function_id']] for r in qualified];jobs=[]
    for target in source:
        for other in source:
            a=target['source_candidate']['function'];b=other['source_candidate']['function']
            if a==b and family(target)!=family(other):
                arity=lambda row:len(ast.parse(row['response']).body[0].args.args)
                if arity(target)==arity(other):jobs.append((target,other))
    if len(jobs)!=4:raise ValueError('All four current same-name/compatible-interface/different-contract directed pairs required')
    OUT.mkdir();atomic_json(OUT/'plan.json',dict(source_files_sha256=files,all_qualified_sources=len(source),all_directed_candidate_pairs=len(jobs),
        fixed_pairs=[dict(target_id=a['id'],other_id=b['id'],target_contract=family(a),other_contract=family(b)) for a,b in jobs],
        both_originals_qualified_for_own_contract=True,negative_is_only_wrong_for_this_task_not_bad_in_own_context=True,
        all_pairs_selected_before_cross_contract_runtime_results=True,workers=2,CPU_nice=os.nice(0),neural_updates=0,goal_achieved=False))
    tokenizer=Tokenizer.from_file(str(ORIGINAL/'tokenizer.json'))
    def work(job):
        target,other=job
        semantic=observe(target,other['response']);published=public_observe(target,other['response'])
        ids,labels=encode(target['prompt'],target['response'],tokenizer);negative,negative_labels=encode(target['prompt'],other['response'],tokenizer)
        start=next(i for i,v in enumerate(labels) if v!=-100)
        if (ids[:start]!=negative[:start] or negative_labels[:start]!=[-100]*start
                or not semantic['all_cases_observed'] or semantic['all_cases_passed']
                or not published['all_original_examples_observed'] or published['all_original_examples_passed']
                or any(r['entry_body_calls']<1 for r in semantic['raw_results'])):
            raise ValueError('Every complete wrong-context program must execute its actual body and fail both finite and original public task checks')
        return dict(target_source_id=target['id'],other_source_id=other['id'],target_identity=identity(target),other_identity=identity(other),
            target_contract=family(target),other_contract=family(other),native_prompt_ids=ids[:start],
            chosen_original_complete_response_EOS_ids=ids[start:],rejected_other_contract_complete_response_EOS_ids=negative[start:],
            wrong_context_actual_semantic_run=semantic,wrong_context_actual_public_run=published,
            chosen_rights=target['rights'],rejected_rights=other['rights'],original_source_roles=['train','train'],
            unchanged_other_source_program_has_correct_entrypoint_and_runs=True,
            rejected_program_is_correct_for_own_qualified_contract=True,wrong_context_only=True,
            human_preferences=False,on_policy_RL=False,held_benchmark=False,optimization_admitted=False)
    with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(work,jobs))
    guard.check(full=True);atomic_json(OUT/'all-four-complete-native-preference-candidates.json',results)
    artifact=OUT/'all-four-complete-native-preference-candidates.json'
    atomic_json(OUT/'complete.json',dict(complete=True,all_four_qualified_same_name_different_contract_pairs_retained=True,
        actual_fresh_confined_program_runs=8,complete_both_response_EOS_native_tokens_and_original_prompt_preserved=True,
        all_wrong_context_programs_exercised_real_entrypoint_and_failed_both_task_checks=True,
        independent_readback_and_training_admission_still_required=True,source_files_sha256={**files,str(artifact):file_hash(artifact)},
        actual_neural_forwards=0,actual_neural_updates=0,candidate_promoted=False,goal_achieved=False))
    print(json.dumps(dict(complete_native_candidates=4,fresh_program_runs=8,training_admitted=False,neural_updates=0)),flush=True)


if __name__=='__main__':main()
