"""Fresh own-contract and wrong-context checks for all six source preferences."""
import copy
import gzip
import json
import os
from pathlib import Path
import shutil
import traceback

from tokenizers import Tokenizer
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.codenet_token_pack_v1 import CodePackWriter,CompleteCodePack,FILES
from slm_lab.contract_source_contrasts_v2 import encode
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.local_source_closure_v2 import python_sources
from slm_lab.public_function_contract_census_v2 import digest,family,observe
from slm_lab.public_function_rollout_observation_v1 import observe as public_observe

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'reports/same-name-public-contract-preferences-CPU-v2'
ORIGINAL=ROOT/'data/direct-algorithm-function-contracts-admitted-v1'
QUALIFIED=ROOT/'data/contract-qualified-public-direct-admitted-v3'
OUT=ROOT/'reports/same-name-contract-preferences-v1-readback-admission'
DATA=ROOT/'data/same-name-contract-preferences-admitted-v1'


def read(path):return json.loads(Path(path).read_text())
def effect(value):return {k:v for k,v in value.items() if k!='execution'}


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists() or DATA.exists():raise ValueError('Fresh CUDA-hidden complete preference admission required')
    done=read(SOURCE/'complete.json');plan=read(SOURCE/'plan.json');candidates=read(SOURCE/'all-six-complete-native-preference-candidates.json')
    if not done.get('complete') or len(candidates)!=6 or len(plan['fixed_pairs'])!=6:raise ValueError('All six frozen complete source preference candidates required')
    files=dict(done['source_files_sha256'])
    for path in [SOURCE/'complete.json',SOURCE/'plan.json',*python_sources(ROOT,[Path(__file__)])]:files[str(path)]=file_hash(path)
    guard=FileBindings(files);original=list(map(json.loads,gzip.open(ORIGINAL/'original-rows.jsonl.gz','rt')))
    by={row['id']:row for row in original};pack=CompleteCodePack(ORIGINAL,'train','training')
    native_position={original[int(index)]['id']:i for i,index in enumerate(pack.positions)}
    qualified={row['original_source_function_id'] for row in map(json.loads,gzip.open(QUALIFIED/'original-rows.jsonl.gz','rt'))}
    tokenizer=Tokenizer.from_file(str(ORIGINAL/'tokenizer.json'));OUT.mkdir();DATA.mkdir()
    for name in ['tokenizer.json','model-config.json','DATASET-LICENSE.txt','publisher-README.md']:shutil.copyfile(ORIGINAL/name,DATA/name)
    atomic_json(OUT/'plan.json',dict(source_files_sha256=files,all_six_candidates_required=True,
        both_public_implementations_must_pass_own_finite_and_whole_public_contracts=True,
        wrong_context_programs_must_execute_actual_body_and_violate_explicit_target_contract=True,
        original_public_examples_may_not_detect_container_type_mismatch=True,
        same_bound_contracts_and_execution_implementation=True,CPU_nice=os.nice(0),neural_updates=0,goal_achieved=False))
    writer=CodePackWriter(DATA,maximum_bytes=16*1024**2,minimum_free_gib=26);pairs=[];readbacks=[];expected_native=[]
    try:
        for index,(candidate,fixed) in enumerate(zip(candidates,plan['fixed_pairs'])):
            target=by[candidate['target_source_id']];other=by[candidate['other_source_id']]
            if (candidate['target_source_id']!=fixed['target_id'] or candidate['other_source_id']!=fixed['other_id']
                    or target['id'] not in qualified or other['id'] not in qualified
                    or target['source_candidate']['function']!=other['source_candidate']['function'] or family(target)==family(other)
                    or target['id'] not in native_position or other['id'] not in native_position):
                raise ValueError('All fixed distinct qualified train sources with the same entrypoint and different contracts required')
            own_target=observe(target);public_target=public_observe(target,target['response'])
            own_other=observe(other);public_other=public_observe(other,other['response'])
            cross=observe(target,other['response']);cross_public=public_observe(target,other['response'])
            if (not all(v['all_cases_passed'] for v in [own_target,own_other])
                    or not all(v['all_original_examples_passed'] for v in [public_target,public_other])
                    or not cross['all_cases_observed'] or cross['all_cases_passed']
                    or any(v['entry_body_calls']<1 for v in cross['raw_results'])
                    or effect(cross)!=effect(candidate['wrong_context_actual_semantic_run'])
                    or effect(cross_public)!=effect(candidate['wrong_context_actual_public_run'])):
                raise ValueError('Every own-context success and actual wrong-context violation must freshly reproduce')
            ids,labels=encode(target['prompt'],target['response'],tokenizer);negative,negative_labels=encode(target['prompt'],other['response'],tokenizer)
            start=len(candidate['native_prompt_ids'])
            if (ids!=candidate['native_prompt_ids']+candidate['chosen_original_complete_response_EOS_ids']
                    or negative!=candidate['native_prompt_ids']+candidate['rejected_other_contract_complete_response_EOS_ids']
                    or labels!=[-100]*start+ids[start:] or negative_labels!=[-100]*start+negative[start:]
                    or pack.sequence(native_position[target['id']])!=(ids,labels)):
                raise ValueError('Both full native programs/EOS and original target prefix/positive masks must stay exact')
            row=copy.deepcopy(target);row.update(id='same-name-contract:'+digest([target['id'],other['id']]),
                original_source_function_id=target['id'],other_source_function_id=other['id'],
                target_contract=family(target),other_contract=family(other),other_rights=other['rights'],
                other_public_URL=other['original_public_URL'],unchanged_original_positive_prompt_and_response=True,
                negative_correct_for_own_contract_but_wrong_for_this_context=True,
                human_preferences=False,on_policy_RL=False,dataset_admitted_to_training=False)
            writer.add(row,ids,labels);expected_native.append((ids,labels))
            pairs.append(dict(id=row['id'],native_prompt_ids=ids[:start],chosen_complete_response_EOS_ids=ids[start:],
                rejected_complete_response_EOS_ids=negative[start:],original_source_role='train',donor_source_role='train',
                original_source_function_id=target['id'],other_source_function_id=other['id'],target_contract=family(target),other_contract=family(other),
                both_originals_pass_own_finite_contracts=True,rejection_applies_only_to_this_task_context=True,
                all_complete_source_programs_and_true_EOS_kept=True,human_preferences=False,on_policy_RL=False))
            readbacks.append(dict(position=index,target_id=target['id'],other_id=other['id'],fresh_target_own_contract=own_target,
                fresh_target_public=public_target,fresh_other_own_contract=own_other,fresh_other_public=public_other,
                fresh_wrong_context_contract=cross,fresh_wrong_context_public=cross_public,all_native_masks_and_both_complete_responses_exact=True))
        manifest=writer.finish(dict(source_files_sha256=files,kind='same_name_different_explicit_contract_machine_preferences',
            all_six_qualified_directed_pairs=True,original_positive_sources_unchanged=True,both_programs_good_for_own_contract=True,
            finite_qualification_not_general_correctness=True,human_preferences=False,on_policy_RL=False,
            original_training_and_benchmark_scopes_unchanged=True))
        pair_path=DATA/'all-complete-source-preference-pairs.jsonl'
        with pair_path.open('x') as stream:
            for pair in pairs:stream.write(json.dumps(pair)+'\n')
        audit=CompleteCodePack(DATA,'train','audit')
        if len(audit)!=6 or any(audit.sequence(i)!=value for i,value in enumerate(expected_native)):
            raise ValueError('Every actual written complete native candidate must read back from disk before admission')
        guard.check(full=True);receipt=OUT/'readback-complete.json'
        atomic_json(receipt,dict(complete=True,all_six_native_source_preferences_and_masks_rebuilt=True,
            actual_fresh_confined_program_runs=36,actual_readbacks=readbacks,
            both_originals_pass_own_finite_contract_and_all_public_examples=True,
            wrong_context_only_rejections_not_global_wrong_code_labels=True,
            same_bound_contracts_and_execution_implementation=True,actual_neural_forwards=0,actual_neural_updates=0,goal_achieved=False))
        atomic_json(DATA/'admission.json',dict(admitted_to_training=True,dataset_manifest_sha256=file_hash(DATA/'manifest.json'),
            preference_pairs_sha256=file_hash(pair_path),actual_complete_machine_preference_pairs=6,
            explicit_task_contract_preferences_not_human_or_on_policy_RL=True,both_sources_correct_for_own_finite_contract=True,
            no_original_source_relabelled_globally_wrong=True,source_valid_or_test_records=0,
            full_readback=str(receipt),full_readback_sha256=file_hash(receipt),source_files_sha256=files,
            finite_qualification_not_general_correctness=True,training_started=False,new_neural_updates=0,goal_achieved=False))
        train=CompleteCodePack(DATA,'train','training')
        if len(train)!=6:raise ValueError('All six actual positive native training rows must load')
        atomic_json(OUT/'complete.json',dict(complete=True,actual_admitted_machine_source_preferences=6,
            actual_positive_response_EOS_targets=manifest['counts']['train']['target_tokens'],actual_fresh_confined_program_runs=36,
            source_files_sha256={**files,**{str(p):file_hash(p) for p in [receipt,DATA/'manifest.json',DATA/'admission.json',pair_path,*[DATA/name for name in FILES]]}},
            candidate_promoted=False,actual_neural_forwards=0,actual_neural_updates=0,goal_achieved=False))
        print(json.dumps(dict(admitted_machine_preferences=6,fresh_program_runs=36,actual_neural_updates=0)),flush=True)
    except BaseException as error:
        if not writer.tokens.closed:writer.close()
        atomic_json(OUT/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),automatic_retry=False,goal_achieved=False));raise


if __name__=='__main__':main()
