"""Fresh source/native/runtime readback for every diagnostic and public contrast.

This reuses the bound numeric specifications and confined execution engine in
new processes; it does not claim a separate oracle implementation or benchmark.
Admission covers only the frozen three public-source correction examples.
"""
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
from slm_lab.code_quality import instruction_code
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.exact_waiting_process_retirement_v1 import terminal
from slm_lab.function_semantic_stress_v1 import NAMES,DONORS,cases,observe,public_alternative
from slm_lab.local_source_closure_v2 import python_sources
from slm_lab.own_function_result_readback_v1 import reconstruct
from slm_lab.own_function_trajectory_preferences_v1 import OwnFunctionTrajectoryPreferences
from slm_lab.public_function_rollout_observation_v1 import observe as public_observe
from slm_lab.public_semantic_source_contrast_v1 import make

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'reports/function-semantic-stress-CPU-v1'
CANDIDATE=ROOT/'data/public-semantic-source-contrast-candidate-v1'
ORIGINAL=ROOT/'data/direct-algorithm-function-contracts-admitted-v1'
OUT=ROOT/'reports/public-semantic-source-contrast-v1-readback-admission'
ADMITTED=ROOT/'data/public-semantic-source-contrast-admitted-v1'


def read(path):return json.loads(Path(path).read_text())
def records(path):return list(map(json.loads,Path(path).read_text().splitlines()))
def effect(value):return {k:v for k,v in value.items() if k!='execution'}
def public_effect(value):return {k:v for k,v in value.items() if k!='execution'}


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists() or ADMITTED.exists():raise ValueError('Fresh CUDA-hidden independent readback required')
    plan=read(SOURCE/'plan.json')
    while not terminal(plan['owner']):time.sleep(2)
    if (SOURCE/'failure.json').exists() or not (SOURCE/'complete.json').exists():raise ValueError('Actual complete original diagnostic required')
    done=read(SOURCE/'complete.json')
    if (done.get('complete') is not True or done.get('all56source_and_model_program_pairs_retained') is not True
            or done.get('actual_complete_public_correction_candidates')!=3):
        raise ValueError('All original source/model programs and fixed three public corrections required')
    files=dict(done['source_files_sha256'])
    for path in [SOURCE/'complete.json',*python_sources(ROOT,[Path(__file__)])]:files[str(path)]=file_hash(path)
    guard=FileBindings(files);fixed=read(SOURCE/'fixed-cases.json');observations=records(SOURCE/'all-original-and-model-fresh-input-runs.jsonl')
    sources=read(ROOT/'reports/completed-code-public-function-rollouts-CPU-v1/all14-original-source-train-records.json')
    rows=[v['row'] for v in sources];pairs=OwnFunctionTrajectoryPreferences(ROOT/'data/own-function-trajectory-preferences-v1','audit')
    tokenizer=Tokenizer.from_file(str(ORIGINAL/'tokenizer.json'));programs={}
    previous=ROOT/'reports/own-function-trajectory-preferences-full-CPU-v4'
    for endpoint in ['initial','chosen_CE','trajectory_DPO']:
        events=previous/('initial-actual-native-actions.jsonl' if endpoint=='initial' else endpoint+'/all14-actual-native-actions.jsonl')
        deliveries=previous/('initial-actual-programs-and-public-executions.jsonl' if endpoint=='initial' else endpoint+'/all14-actual-generated-programs-and-public-executions.jsonl')
        programs[endpoint]=reconstruct(pairs.rows,records(events),records(deliveries),tokenizer.decode,instruction_code)
    if (len(fixed)!=14 or [r['source_candidate']['function'] for r in rows]!=NAMES or len(observations)!=56
            or [r['id'] for r in rows]!=[r['id'] for r in pairs.rows]):raise ValueError('Every fixed source/role/action population required')
    for i,row in enumerate(rows):
        if (fixed[i]['position']!=i or fixed[i]['source_id']!=row['id'] or fixed[i]['cases']!=cases(row)
                or fixed[i]['optimization_role']!=pairs.rows[i]['optimization_role']):
            raise ValueError('Every numerical case and original optimization/control role must reconstruct')
    jobs=[(endpoint,i) for endpoint in ['original_public',*programs] for i in range(14)]
    if [(r['endpoint'],r['position']) for r in observations]!=jobs:raise ValueError('Every source/model outcome must remain in original order')
    OUT.mkdir();atomic_json(OUT/'plan.json',dict(owner=dict(pid=os.getpid(),ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]),
        source_files_sha256=files,all56programs_rebuilt_and_reobserved=True,every_syntactically_valid_program_reexecuted=True,workers=2,CPU_nice=os.nice(0),
        no_model_forwards_or_optimization=True,same_runtime_and_bound_specification_implementation=True,goal_achieved=False))
    def work(item):
        endpoint,i=item;program=rows[i]['response'] if endpoint=='original_public' else programs[endpoint][i]['program']
        actual=observe(program,NAMES[i],fixed[i]['cases']);stored=observations[jobs.index(item)]
        if (hashlib.sha256(program.encode()).hexdigest()!=stored['program_sha256']
                or any(effect(actual)!=effect(r) for r in stored['actual_runs'])):
            raise ValueError('Every unchanged complete native program and fresh numerical outcome must reproduce')
        return dict(endpoint=endpoint,position=i,source_id=rows[i]['id'],actual_fresh_run=actual)
    try:
        reruns=[]
        with (OUT/'all56fresh-source-and-model-readbacks.jsonl').open('x') as stream,ThreadPoolExecutor(max_workers=2) as pool:
            for value in pool.map(work,jobs):reruns.append(value);stream.write(json.dumps(value)+'\n');stream.flush()
        pack=CompleteCodePack(CANDIDATE,'train','audit');derived=list(map(json.loads,gzip.open(CANDIDATE/'original-rows.jsonl.gz','rt')))
        stored_pairs=records(CANDIDATE/'all-complete-source-preference-pairs.jsonl');candidate_evidence=read(SOURCE/'all-public-correction-candidates-and-executions.json')
        original=list(map(json.loads,gzip.open(ORIGINAL/'original-rows.jsonl.gz','rt')))
        by_id={r['id']:r for r in original};proofs=[];fresh_runs=sum(v['actual_fresh_run']['actual_program_launched'] for v in reruns)
        if len(pack)!=len(derived) or len(pack)!=len(stored_pairs) or len(pack)!=len(candidate_evidence) or len(pack)!=3:
            raise ValueError('Every and only frozen public correction row must remain')
        for i,(candidate,pair,evidence) in enumerate(zip(derived,stored_pairs,candidate_evidence)):
            row=by_id[candidate['original_source_function_id']];donor=by_id[candidate['donor_source_function_id']];name=row['source_candidate']['function']
            position=NAMES.index(name);negative=reruns[position]['actual_fresh_run']
            rebuilt,ids,labels,rebuilt_pair=make(row,donor,negative,tokenizer)
            expected={k:v for k,v in candidate.items() if k not in ['token_pack_index','complete_input_ids_sha256','target_tokens']}
            if rebuilt!=expected or pair!=rebuilt_pair or pack.sequence(i)!=(ids,labels):
                raise ValueError('All original donor/negative bytes, native masks, both complete EOS responses and rights must reconstruct')
            program=public_alternative(row,donor);good=[observe(program,name,fixed[position]['cases']) for _ in range(2)]
            public_original=[public_observe(row,row['response']) for _ in range(2)]
            public_positive=[public_observe(row,program) for _ in range(2)]
            public_donor=[public_observe(donor,donor['response']) for _ in range(2)]
            if any(not r['all_cases_passed'] for r in good) or any(not r['all_original_examples_passed'] for r in [*public_original,*public_positive,*public_donor]):
                raise ValueError('Every corrected numerical case and every original/target/donor public example must freshly pass')
            for actual,key in [(good,'actual_positive_numerical_runs'),(public_original,'actual_original_public_runs'),
                    (public_positive,'actual_repair_original_public_runs'),(public_donor,'actual_donor_public_runs')]:
                if [effect(v) for v in actual]!=[effect(v) for v in evidence[key]]:
                    raise ValueError('Every frozen source and public correction observation must reproduce')
            proofs.append(dict(id=candidate['id'],function=name,all_native_source_rights_and_complete_pair_bytes_rebuilt=True,
                fresh_positive_numerical_runs=good,fresh_original_public_runs=public_original,
                fresh_positive_public_runs=public_positive,fresh_donor_public_runs=public_donor));fresh_runs+=8
        guard.check(full=True)
        atomic_json(OUT/'readback-complete.json',dict(complete=True,all56source_and_model_programs_rebuilt_and_reobserved=True,
            every_syntactically_valid_program_reexecuted=True,
            actual_fresh_confined_program_runs=fresh_runs,all_three_public_contrasts_rebuilt_and_reexecuted=True,
            public_contrast_readbacks=proofs,source_train_diagnostic_not_independent_benchmark=True,
            same_bound_numeric_specifications_and_execution_implementation=True,new_neural_forwards=0,new_neural_updates=0,goal_achieved=False))
        ADMITTED.mkdir()
        for name in [*FILES,'manifest.json','all-complete-source-preference-pairs.jsonl']:
            shutil.copyfile(CANDIDATE/name,ADMITTED/name)
            if file_hash(CANDIDATE/name)!=file_hash(ADMITTED/name):raise ValueError('Every admitted candidate/rights/pair/native byte must stay exact')
        receipt=OUT/'readback-complete.json'
        atomic_json(ADMITTED/'admission.json',dict(admitted_to_training=True,dataset_manifest_sha256=file_hash(ADMITTED/'manifest.json'),
            verified_public_source_correction_CE_admitted=True,verified_machine_source_contrasts_admitted=True,
            human_preferences=False,on_policy_RL=False,source_valid_or_test_records=0,original_positive_public_donor_rows=3,
            positive_source_bytes_unchanged_plus_static_alias_only=True,negative_is_complete_original_public_code=True,
            finite_numerical_qualification_not_general_correctness=True,post_hoc_source_train_diagnostic_not_benchmark=True,
            candidate_pair_optimization_admitted_field_false_is_historical_pre_readback_state=True,
            full_independent_readback=str(receipt),full_independent_readback_sha256=file_hash(receipt),source_files_sha256=files,
            training_started=False,neural_updates=0,goal_achieved=False))
        training=CompleteCodePack(ADMITTED,'train','training')
        if len(training)!=3:raise ValueError('All three admitted native correction rows must actually load for training')
        atomic_json(OUT/'complete.json',dict(complete=True,at=datetime.now(timezone.utc).isoformat(),actual_fresh_confined_program_runs=fresh_runs,
            original_all56programs_and_three_public_corrections_verified=True,actual_admitted_correction_examples=len(training),
            actual_response_EOS_targets=training.manifest['counts']['train']['target_tokens'],
            source_files_sha256={**files,**{str(p):file_hash(p) for p in [receipt,ADMITTED/'admission.json',ADMITTED/'manifest.json',
                ADMITTED/'all-complete-source-preference-pairs.jsonl',*[ADMITTED/name for name in FILES]]}},
            candidate_promoted=False,actual_neural_forwards=0,actual_neural_updates=0,goal_achieved=False))
        print(json.dumps(dict(admitted_examples=len(training),fresh_confined_program_runs=fresh_runs,actual_neural_updates=0)),flush=True)
    except BaseException as error:
        atomic_json(OUT/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),automatic_retry=False,goal_achieved=False));raise


if __name__=='__main__':main()
