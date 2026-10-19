"""Inventory every original source; qualify exact reviewed finite contracts.

All826train and15validation records retain their original roles and exposures.
New qualification is additional evidence, never an edit to a live curriculum.
Passing whole public functions form a candidate CE pack; failing references may
form a separate corrected-source pack using a compatible public implementation.
"""
import ast
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import traceback

from tokenizers import Tokenizer
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.codenet_token_pack_v1 import CodePackWriter,CompleteCodePack,FILES
from slm_lab.contract_source_contrasts_v1 import direct,contrast,alternative
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.local_source_closure_v2 import python_sources
from slm_lab.public_function_contract_census_v1 import OLD_CASES,cases,digest,family,identity,observe,registry
from slm_lab.public_function_rollout_observation_v1 import observe as public_observe

ROOT=Path(__file__).resolve().parents[1]
ORIGINAL=ROOT/'data/direct-algorithm-function-contracts-admitted-v1'
OUT=ROOT/'reports/public-contract-census-CPU-v1'
DIRECT=ROOT/'data/contract-qualified-public-direct-candidate-v1'
CONTRAST=ROOT/'data/contract-qualified-public-contrast-candidate-v1'


def effect(value):return {k:v for k,v in value.items() if k!='execution'}
def read(path):return json.loads(Path(path).read_text())


def body_digest(program):
    tree=ast.parse(program)
    for node in ast.walk(tree):
        if isinstance(node,(ast.Module,ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
            node.body=[v for v in node.body if not isinstance(v,ast.Expr) or not isinstance(v.value,ast.Constant) or not isinstance(v.value.value,str)]
    return digest(ast.dump(tree))


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or any(p.exists() for p in [OUT,DIRECT,CONTRAST]):raise ValueError('Fresh CUDA-hidden full census and candidate paths required')
    tests={'public-function-contract-census-v1-tests':4,'contract-source-contrasts-v1-tests':2}
    for name,count in tests.items():
        if str(count)+' passed' not in (ROOT/'reports'/(name+'.log')).read_text():raise ValueError('Actual complete return/type/mutation/contract/native controls required')
    pack=CompleteCodePack(ORIGINAL,'train','training');original=list(map(json.loads,gzip.open(ORIGINAL/'original-rows.jsonl.gz','rt')))
    if len(original)!=841 or len(pack)!=826:raise ValueError('Every original sealed public source role must remain')
    positions={int(v):i for i,v in enumerate(pack.positions)};selected=[];inventory=[];keys=registry()
    for i,row in enumerate(original):
        train=i in positions
        if train!=(row['split']=='train') or row['split'] not in ['train','validation']:raise ValueError('Every native and document role must match')
        profile=family(row) if train else None
        record=dict(original_global_position=i,source_id=row['id'],source_group=row['group_id'],source_role=row['split'],
            source_function_identity=identity(row),whole_source_sha256=hashlib.sha256(row['response'].encode()).hexdigest(),
            body_AST_sha256=body_digest(row['response']),contract_family=profile,
            disposition='explicit_contract_selected' if profile else 'held_role_not_qualified_or_optimized' if not train else 'no_reviewed_semantic_contract_not_assumed_correct')
        inventory.append(record)
        if profile:selected.append((i,positions[i],row))
    if set(identity(r) for _,_,r in selected)!=set(keys):raise ValueError('Every exact registered contract source must be present in the train pool')
    inputs=[ORIGINAL/name for name in [*FILES,'manifest.json','admission.json']]+[OLD_CASES,
        ROOT/'reports/completed-code-public-function-rollouts-CPU-v1/all14-original-source-train-records.json',
        *[ROOT/'reports'/(name+'.log') for name in tests],
        *python_sources(ROOT,[Path(__file__),ROOT/'tests/test_public_function_contract_census_v1.py',ROOT/'tests/test_contract_source_contrasts_v1.py'])]
    files={str(p):file_hash(p) for p in set(inputs)};guard=FileBindings(files)
    OUT.mkdir()
    for folder in [DIRECT,CONTRAST]:
        folder.mkdir()
        for name in ['tokenizer.json','model-config.json','DATASET-LICENSE.txt','publisher-README.md']:shutil.copyfile(ORIGINAL/name,folder/name)
    atomic_json(OUT/'all841-original-source-dispositions.json',inventory)
    fixed=[dict(original_global_position=i,source_train_position=p,source_id=row['id'],contract_family=family(row),cases=cases(row)) for i,p,row in selected]
    atomic_json(OUT/'all-fixed-contract-cases.json',fixed)
    atomic_json(OUT/'plan.json',dict(at=datetime.now(timezone.utc).isoformat(),owner=dict(pid=os.getpid(),ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]),
        source_files_sha256=files,all_original_train_sources=826,all_original_validation_sources_excluded=15,
        actual_selected_contract_sources=len(selected),actual_contract_families=dict(Counter(family(r) for _,_,r in selected)),
        exact_reviewed_contract_registry=keys,all_cases_sha256=file_hash(OUT/'all-fixed-contract-cases.json'),
        every_source_disposition_sha256=file_hash(OUT/'all841-original-source-dispositions.json'),
        first_passing_public_implementation_in_original_order_is_donor=True,
        donor_must_pass_own_and_target_original_public_examples_and_every_target_finite_contract_case=True,
        code_and_prompts_never_clipped=True,unsupported_sources_never_relabelled_as_correct_or_incorrect=True,
        qualification_is_additional_not_a_replacement_of_any_original_training_or_benchmark=True,
        source_train_post_hoc_diagnostic_not_independent_model_benchmark=True,workers=2,CPU_nice=os.nice(0),
        new_neural_forwards=0,new_neural_updates=0,goal_achieved=False))
    def work(item):
        i,p,row=item;semantic=[observe(row) for _ in range(2)];published=[public_observe(row,row['response']) for _ in range(2)]
        stable=effect(semantic[0])==effect(semantic[1]) and effect(published[0])==effect(published[1])
        full_public=all(r['all_original_examples_passed'] for r in published)
        disposition=('unstable_preserved_no_qualification' if not stable else 'current_public_examples_fail_no_semantic_wrong_code_label' if not full_public
            else 'finite_contract_qualified' if all(r['all_cases_passed'] for r in semantic)
            else 'current_runtime_or_context_qualification_not_wrong_code' if not semantic[0]['all_cases_observed'] else 'verified_contract_counterexample')
        return dict(original_global_position=i,source_train_position=p,source_id=row['id'],source_function_identity=identity(row),
            contract_family=family(row),disposition=disposition,actual_semantic_runs=semantic,actual_whole_public_runs=published)
    direct_writer=CodePackWriter(DIRECT,maximum_bytes=64*1024**2,minimum_free_gib=26)
    contrast_writer=CodePackWriter(CONTRAST,maximum_bytes=32*1024**2,minimum_free_gib=26)
    try:
        outcomes=[]
        with (OUT/'all-selected-original-public-and-contract-runs.jsonl').open('x') as stream,ThreadPoolExecutor(max_workers=2) as pool:
            for value in pool.map(work,selected):
                outcomes.append(value);stream.write(json.dumps(value)+'\n');stream.flush()
                atomic_json(OUT/'progress.json',dict(qualified_source_observations=len(outcomes),planned=len(selected),new_neural_updates=0,goal_achieved=False))
        lookup={v['source_id']:v for v in outcomes};qualified=[row for _,_,row in selected if lookup[row['id']]['disposition']=='finite_contract_qualified']
        tokenizer=Tokenizer.from_file(str(ORIGINAL/'tokenizer.json'));native_position={row['id']:p for _,p,row in selected}
        for row in qualified:
            derived,ids,labels=direct(row,tokenizer)
            if pack.sequence(native_position[row['id']])!=(ids,labels):raise ValueError('Every qualified direct original native input, target mask and EOS must remain exact')
            direct_writer.add(derived,ids,labels)
        corrections=[];all_trials=[]
        for _,_,row in selected:
            outcome=lookup[row['id']]
            if outcome['disposition']!='verified_contract_counterexample':continue
            attempts=[];chosen=None
            for donor in qualified:
                if family(row)!=family(donor) or row['id']==donor['id']:continue
                program=alternative(row,donor);semantic=[observe(row,program) for _ in range(2)];published=[public_observe(row,program) for _ in range(2)]
                candidate=dict(donor_source_id=donor['id'],actual_target_semantic_runs=semantic,actual_target_original_public_runs=published)
                attempts.append(candidate)
                if (any(not v['all_cases_passed'] for v in semantic) or any(not v['all_original_examples_passed'] for v in published)
                        or effect(semantic[0])!=effect(semantic[1]) or effect(published[0])!=effect(published[1])):continue
                derived,ids,labels,pair=contrast(row,donor,outcome['actual_semantic_runs'][0],tokenizer)
                contrast_writer.add(derived,ids,labels);chosen=dict(candidate_id=derived['id'],source_id=row['id'],donor_source_id=donor['id'],
                    candidate=derived,pair=pair,actual_positive_target_semantic_runs=semantic,actual_positive_target_original_public_runs=published)
                corrections.append(chosen);break
            all_trials.append(dict(source_id=row['id'],contract_family=family(row),all_actual_ordered_donor_trials=attempts,
                selected_candidate_id=None if chosen is None else chosen['candidate_id'],
                disposition='complete_public_contract_correction_candidate' if chosen else 'no_compatible_fully_passing_public_donor_preserved'))
        common=dict(source_files_sha256=files,all841source_inventory_preserved=True,all826train_and15validation_roles_unchanged=True,
            every_selected_source_full_finite_and_whole_public_execution_record_retained=True,
            original_live_training_and_benchmark_scopes_unchanged=True,finite_contract_qualification_not_general_correctness=True,
            source_train_data_not_independent_benchmark=True,machine_source_contrasts_not_human_or_on_policy_feedback=True,
            independent_native_source_and_runtime_readback_still_required=True)
        direct_manifest=direct_writer.finish(dict(**common,kind='unchanged_whole_qualified_public_functions'))
        contrast_manifest=contrast_writer.finish(dict(**common,kind='unchanged_public_donor_with_static_alias_and_whole_original_negative'))
        with (CONTRAST/'all-complete-source-preference-pairs.jsonl').open('x') as stream:
            for value in corrections:stream.write(json.dumps(value['pair'])+'\n')
        atomic_json(OUT/'all-ordered-correction-donor-trials.json',all_trials);atomic_json(OUT/'all-complete-correction-candidates.json',corrections)
        guard.check(full=True)
        extra=[OUT/'plan.json',OUT/'all841-original-source-dispositions.json',OUT/'all-fixed-contract-cases.json',
            OUT/'all-selected-original-public-and-contract-runs.jsonl',OUT/'all-ordered-correction-donor-trials.json',OUT/'all-complete-correction-candidates.json',
            CONTRAST/'all-complete-source-preference-pairs.jsonl']
        for folder in [DIRECT,CONTRAST]:extra += [folder/name for name in [*FILES,'manifest.json']]
        atomic_json(OUT/'complete.json',dict(complete=True,at=datetime.now(timezone.utc).isoformat(),all_original_sources_inventoried=841,
            source_role_counts=dict(Counter(r['source_role'] for r in inventory)),all_selected_contract_sources_observed=len(outcomes),
            source_dispositions=dict(Counter(v['disposition'] for v in outcomes)),
            actual_qualified_direct_functions=len(qualified),actual_complete_public_correction_candidates=len(corrections),
            direct_native_counts=direct_manifest['counts'],contrast_native_counts=contrast_manifest['counts'],
            actual_confined_program_runs=sum(2+sum(v['actual_program_launched'] for v in row['actual_semantic_runs']) for row in outcomes)+
                sum(sum(v['actual_program_launched'] for v in trial['actual_target_semantic_runs'])+2 for row in all_trials for trial in row['all_actual_ordered_donor_trials']),
            source_files_sha256={**files,**{str(p):file_hash(p) for p in extra}},training_admitted=False,independent_readback_still_required=True,
            candidate_promoted=False,new_neural_forwards=0,new_neural_updates=0,goal_achieved=False))
        print(json.dumps(dict(observed_sources=len(outcomes),dispositions=dict(Counter(v['disposition'] for v in outcomes)),
            qualified_direct=len(qualified),public_corrections=len(corrections),training_admitted=False)),flush=True)
    except BaseException as error:
        for writer in [direct_writer,contrast_writer]:
            if not writer.tokens.closed:writer.close()
        atomic_json(OUT/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),automatic_retry=False,goal_achieved=False));raise


if __name__=='__main__':main()
