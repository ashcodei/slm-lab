"""Read back every source disposition, finite contract and complete native target.

Both additional packs use original train sources only. Each fixed public donor
trial is freshly observed, including any rejected trial. This is separate-process
readback using the same bound contracts/runtime, not a new model benchmark.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import traceback

from tokenizers import Tokenizer
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.codenet_token_pack_v1 import CompleteCodePack,FILES
from slm_lab.contract_source_contrasts_v1 import direct,contrast,alternative
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.exact_waiting_process_retirement_v1 import terminal
from slm_lab.local_source_closure_v2 import python_sources
from slm_lab.public_function_contract_census_v1 import cases,digest,family,identity,observe,registry
from slm_lab.public_function_rollout_observation_v1 import observe as public_observe
from collect_public_contract_census_CPU_v1 import body_digest,effect

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'reports/public-contract-census-CPU-v1'
ORIGINAL=ROOT/'data/direct-algorithm-function-contracts-admitted-v1'
OUT=ROOT/'reports/public-contract-census-v1-readback-admission'
FOLDERS={kind:(ROOT/('data/contract-qualified-public-'+kind+'-candidate-v1'),ROOT/('data/contract-qualified-public-'+kind+'-admitted-v1'))
    for kind in ['direct','contrast']}


def read(path):return json.loads(Path(path).read_text())
def records(path):return list(map(json.loads,Path(path).read_text().splitlines()))


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists() or any(b.exists() for _,b in FOLDERS.values()):
        raise ValueError('Fresh CUDA-hidden complete-source admission required')
    plan=read(SOURCE/'plan.json')
    while not terminal(plan['owner']):time.sleep(2)
    if (SOURCE/'failure.json').exists() or not (SOURCE/'complete.json').exists():raise ValueError('Actual complete whole source census required')
    done=read(SOURCE/'complete.json')
    if (not done.get('complete') or done['all_original_sources_inventoried']!=841
            or done['source_role_counts']!={'train':826,'validation':15}
            or done['all_selected_contract_sources_observed']!=len(registry())):
        raise ValueError('Every original source/role and every fixed contract source required')
    files=dict(done['source_files_sha256'])
    for path in [SOURCE/'complete.json',*python_sources(ROOT,[Path(__file__)])]:files[str(path)]=file_hash(path)
    guard=FileBindings(files);original=list(map(json.loads,gzip.open(ORIGINAL/'original-rows.jsonl.gz','rt')))
    pack=CompleteCodePack(ORIGINAL,'train','training');positions={int(p):i for i,p in enumerate(pack.positions)}
    inventory=read(SOURCE/'all841-original-source-dispositions.json');fixed=read(SOURCE/'all-fixed-contract-cases.json')
    old=records(SOURCE/'all-selected-original-public-and-contract-runs.jsonl');selected=[]
    if len(original)!=841 or len(inventory)!=841 or len(old)!=len(registry()) or len(fixed)!=len(old):
        raise ValueError('Whole inventory and every selected observation/case list required')
    for i,(row,stored) in enumerate(zip(original,inventory)):
        train=i in positions;profile=family(row) if train else None
        expected=dict(original_global_position=i,source_id=row['id'],source_group=row['group_id'],source_role=row['split'],
            source_function_identity=identity(row),whole_source_sha256=hashlib.sha256(row['response'].encode()).hexdigest(),
            body_AST_sha256=body_digest(row['response']),contract_family=profile,
            disposition='explicit_contract_selected' if profile else 'held_role_not_qualified_or_optimized' if not train else 'no_reviewed_semantic_contract_not_assumed_correct')
        if stored!=expected or train!=(row['split']=='train'):raise ValueError('Every original AST/source/file/role/disposition must reconstruct')
        if profile:selected.append((i,positions[i],row))
    for (i,p,row),chosen,observed in zip(selected,fixed,old):
        expected=dict(original_global_position=i,source_train_position=p,source_id=row['id'],contract_family=family(row),cases=cases(row))
        if chosen!=expected or observed['source_id']!=row['id'] or observed['source_train_position']!=p:
            raise ValueError('Every exact complete fixed contract and source order must reconstruct')
    OUT.mkdir();atomic_json(OUT/'plan.json',dict(owner=dict(pid=os.getpid(),ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]),
        source_files_sha256=files,all841original_source_dispositions_reconstructed=True,all_selected_sources=len(selected),
        workers=2,CPU_nice=os.nice(0),same_bound_contracts_and_execution_engine=True,
        no_replacement_of_original_sources_exposures_or_benchmarks=True,new_neural_forwards=0,new_neural_updates=0,goal_achieved=False))
    def work(item):
        index,(i,p,row)=item;stored=old[index];semantic=observe(row);published=public_observe(row,row['response'])
        if any(effect(semantic)!=effect(v) for v in stored['actual_semantic_runs']) or any(effect(published)!=effect(v) for v in stored['actual_whole_public_runs']):
            raise ValueError('Every fixed source finite-contract and whole public observation must reproduce')
        return dict(source_id=row['id'],original_global_position=i,source_train_position=p,
            actual_fresh_semantic_run=semantic,actual_fresh_public_run=published,original_disposition=stored['disposition'])
    try:
        fresh=[]
        with (OUT/'every-selected-source-fresh-readback.jsonl').open('x') as stream,ThreadPoolExecutor(max_workers=2) as pool:
            for item in pool.map(work,enumerate(selected)):fresh.append(item);stream.write(json.dumps(item)+'\n');stream.flush()
        lookup={row['source_id']:row for row in old};fresh_lookup={row['source_id']:row for row in fresh};by_id={row['id']:row for row in original}
        qualified=[row for _,_,row in selected if lookup[row['id']]['disposition']=='finite_contract_qualified'];tokenizer=Tokenizer.from_file(str(ORIGINAL/'tokenizer.json'))
        direct_candidate=FOLDERS['direct'][0];direct_pack=CompleteCodePack(direct_candidate,'train','audit')
        direct_rows=list(map(json.loads,gzip.open(direct_candidate/'original-rows.jsonl.gz','rt')))
        if len(qualified)!=len(direct_pack) or len(direct_pack)!=done['actual_qualified_direct_functions']:
            raise ValueError('All and only frozen finite-qualified original functions may become direct targets')
        for i,(row,stored) in enumerate(zip(qualified,direct_rows)):
            rebuilt,ids,labels=direct(row,tokenizer)
            expected={k:v for k,v in stored.items() if k not in ['token_pack_index','complete_input_ids_sha256','target_tokens']}
            source_position=next(p for _,p,source in selected if source['id']==row['id'])
            if rebuilt!=expected or direct_pack.sequence(i)!=(ids,labels) or pack.sequence(source_position)!=(ids,labels):
                raise ValueError('Every original direct source/code/rights/native prompt mask and gold/EOS target must stay byte-exact')
        trials=read(SOURCE/'all-ordered-correction-donor-trials.json');evidence=read(SOURCE/'all-complete-correction-candidates.json')
        negative_rows=[row for _,_,row in selected if lookup[row['id']]['disposition']=='verified_contract_counterexample']
        if [v['source_id'] for v in trials]!=[r['id'] for r in negative_rows]:raise ValueError('Every fixed actual source failure needs its complete donor disposition')
        actual_runs=sum(int(v['actual_fresh_semantic_run']['actual_program_launched'])+1 for v in fresh);new_trials=[];selected_candidates=[]
        for row,stored in zip(negative_rows,trials):
            eligible=[d for d in qualified if family(d)==family(row) and row['id']!=d['id']]
            if [v['donor_source_id'] for v in stored['all_actual_ordered_donor_trials']]!=[d['id'] for d in eligible[:len(stored['all_actual_ordered_donor_trials'])]]:
                raise ValueError('Every and only fixed original-order donor prefix must remain')
            observed_trials=[];selected_id=None
            for attempt in stored['all_actual_ordered_donor_trials']:
                donor=by_id[attempt['donor_source_id']];program=alternative(row,donor)
                semantic=observe(row,program);published=public_observe(row,program);actual_runs+=int(semantic['actual_program_launched'])+1
                if any(effect(semantic)!=effect(v) for v in attempt['actual_target_semantic_runs']) or any(effect(published)!=effect(v) for v in attempt['actual_target_original_public_runs']):
                    raise ValueError('Every selected and rejected donor runtime observation must reproduce')
                observed_trials.append(dict(donor_source_id=donor['id'],fresh_semantic=semantic,fresh_public=published))
                if semantic['all_cases_passed'] and published['all_original_examples_passed']:
                    built=contrast(row,donor,fresh_lookup[row['id']]['actual_fresh_semantic_run'],tokenizer)
                    if selected_id is not None:raise ValueError('Only the fixed first fully passing public donor is eligible')
                    selected_id=built[0]['id'];selected_candidates.append((row,donor,built))
            if selected_id!=stored['selected_candidate_id']:raise ValueError('No score-selected new donor or dropped candidate may enter admission')
            new_trials.append(dict(source_id=row['id'],selected_candidate_id=selected_id,all_fresh_ordered_donor_trials=observed_trials))
        contrast_candidate=FOLDERS['contrast'][0];contrast_pack=CompleteCodePack(contrast_candidate,'train','audit')
        contrast_rows=list(map(json.loads,gzip.open(contrast_candidate/'original-rows.jsonl.gz','rt')))
        preference_rows=records(contrast_candidate/'all-complete-source-preference-pairs.jsonl')
        if not len(selected_candidates)==len(contrast_pack)==len(contrast_rows)==len(preference_rows)==len(evidence)==done['actual_complete_public_correction_candidates']:
            raise ValueError('Every complete frozen public contrast/native preference pair required')
        for i,((row,donor,(candidate,ids,labels,pair)),stored,pref,extra) in enumerate(zip(selected_candidates,contrast_rows,preference_rows,evidence)):
            expected={k:v for k,v in stored.items() if k not in ['token_pack_index','complete_input_ids_sha256','target_tokens']}
            if candidate!=expected or pair!=pref or candidate!=extra['candidate'] or pair!=extra['pair'] or contrast_pack.sequence(i)!=(ids,labels):
                raise ValueError('Every whole public donor/negative program, complete positive/negative EOS, mask and rights byte must rebuild')
        guard.check(full=True);receipt=OUT/'readback-complete.json'
        atomic_json(receipt,dict(complete=True,all841source_dispositions_rebuilt=True,all826train_and15held_source_roles_unchanged=True,
            every_selected_source_contract_and_whole_public_run_reobserved=True,actual_selected_source_functions=len(selected),
            all_direct_whole_source_native_masks_and_EOS_exact=True,all_contrast_positive_negative_and_rights_native_bytes_exact=True,
            actual_direct_targets=len(direct_pack),actual_contrast_targets=len(contrast_pack),actual_fresh_confined_program_runs=actual_runs,
            all_ordered_donor_trials_reobserved=new_trials,same_bound_contract_and_execution_implementation=True,
            finite_qualified_source_data_not_model_benchmark=True,new_neural_forwards=0,new_neural_updates=0,goal_achieved=False))
        inventories={}
        for kind,(candidate,admitted) in FOLDERS.items():
            admitted.mkdir();names=[*FILES,'manifest.json']+(['all-complete-source-preference-pairs.jsonl'] if kind=='contrast' else [])
            for name in names:
                shutil.copyfile(candidate/name,admitted/name)
                if file_hash(candidate/name)!=file_hash(admitted/name):raise ValueError('Every native/source/rights/pair byte must remain exact')
            count=len(direct_pack) if kind=='direct' else len(contrast_pack)
            atomic_json(admitted/'admission.json',dict(admitted_to_training=True,dataset_manifest_sha256=file_hash(admitted/'manifest.json'),
                kind=kind,finite_contract_qualified_public_CE_admitted=True,finite_qualified_machine_source_contrasts_admitted=kind=='contrast',
                original_source_train_records_only=True,source_valid_or_test_records=0,actual_examples=count,
                original_public_donor_source_bytes_unchanged=True,static_alias_only_for_contrasts=kind=='contrast',
                full_readback=str(receipt),full_readback_sha256=file_hash(receipt),source_files_sha256=files,
                human_preferences=False,on_policy_RL=False,finite_qualification_not_general_correctness=True,
                original_training_exposures_and_benchmarks_unmodified=True,training_started=False,new_neural_updates=0,goal_achieved=False))
            loaded=CompleteCodePack(admitted,'train','training')
            if len(loaded)!=count:raise ValueError('Every actual admitted complete native training target must load')
            inventories[kind]=dict(folder=str(admitted),examples=len(loaded),response_EOS_targets=loaded.manifest['counts']['train']['target_tokens'])
        extra=[receipt]
        for kind,(_,folder) in FOLDERS.items():extra += [folder/name for name in [*FILES,'manifest.json','admission.json']]+([folder/'all-complete-source-preference-pairs.jsonl'] if kind=='contrast' else [])
        atomic_json(OUT/'complete.json',dict(complete=True,at=datetime.now(timezone.utc).isoformat(),actual_admitted= inventories,
            actual_fresh_confined_program_runs=actual_runs,all841source_inventory_and_every_selected_contract_and_native_target_verified=True,
            source_files_sha256={**files,**{str(p):file_hash(p) for p in extra}},candidate_promoted=False,
            actual_neural_forwards=0,actual_neural_updates=0,goal_achieved=False))
        print(json.dumps(dict(admitted=inventories,fresh_confined_program_runs=actual_runs,actual_neural_updates=0)),flush=True)
    except BaseException as error:
        atomic_json(OUT/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),automatic_retry=False,goal_achieved=False));raise


if __name__=='__main__':main()
