"""Fresh-input readback of all source-learning outcomes and public repair candidates.

All fourteen original functions, three model endpoints and the original public
references remain. This post-hoc diagnostic does not alter any live curriculum,
benchmark population, source label or generated program.
"""
import ast
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import gzip
import json
import os
from pathlib import Path
import shutil
import traceback

from tokenizers import Tokenizer
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.codenet_token_pack_v1 import CodePackWriter,FILES
from slm_lab.code_quality import instruction_code
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.function_semantic_stress_v1 import NAMES,DONORS,DOMAINS,cases,digest,observe,public_alternative
from slm_lab.local_source_closure_v2 import python_sources
from slm_lab.own_function_result_readback_v1 import reconstruct
from slm_lab.own_function_trajectory_preferences_v1 import OwnFunctionTrajectoryPreferences
from slm_lab.public_function_rollout_observation_v1 import observe as public_observe
from slm_lab.public_semantic_source_contrast_v1 import FORMAT,make

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'reports/own-function-trajectory-preferences-full-CPU-v4'
ORIGINAL=ROOT/'reports/completed-code-public-function-rollouts-CPU-v1/all14-original-source-train-records.json'
PACK=ROOT/'data/direct-algorithm-function-contracts-admitted-v1'
OUT=ROOT/'reports/function-semantic-stress-CPU-v1'
DATA=ROOT/'data/public-semantic-source-contrast-candidate-v1'


def read(path):return json.loads(Path(path).read_text())
def records(path):return list(map(json.loads,Path(path).read_text().splitlines()))
def effect(value):return {k:v for k,v in value.items() if k!='execution'}


def code_digest(program):
    try:tree=ast.parse(program)
    except (SyntaxError,ValueError,RecursionError):return None
    for node in ast.walk(tree):
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef,ast.Module)):
            node.body=[v for v in node.body if not isinstance(v,ast.Expr) or not isinstance(v.value,ast.Constant) or not isinstance(v.value.value,str)]
    return digest(ast.dump(tree))


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists() or DATA.exists():raise ValueError('Fresh CUDA-hidden readback required')
    tests=ROOT/'reports/function-semantic-stress-v1-tests.log'
    if '4 passed' not in tests.read_text():raise ValueError('Actual source/domain/type and confined bad/good-reference controls must pass')
    contrast_tests=ROOT/'reports/public-semantic-source-contrast-v1-tests.log'
    if '2 passed' not in contrast_tests.read_text():raise ValueError('Both full native programs and source-role/counterexample rejection controls required')
    original=read(ORIGINAL);rows=[v['row'] for v in original]
    if [r['source_candidate']['function'] for r in rows]!=NAMES:raise ValueError('All original fourteen source functions in their original order required')
    data=OwnFunctionTrajectoryPreferences(ROOT/'data/own-function-trajectory-preferences-v1','audit')
    if [p['id'] for p in data.rows]!=[r['id'] for r in rows]:raise ValueError('All original fixed optimization/control source identities required')
    tokenizer=Tokenizer.from_file(str(PACK/'tokenizer.json'));programs={};paths=[Path(__file__),ORIGINAL,tests,contrast_tests,
        ROOT/'reports/own-function-training-results-CPU-v1-readback/complete.json',SOURCE/'complete.json']
    for endpoint in ['initial','chosen_CE','trajectory_DPO']:
        events=SOURCE/('initial-actual-native-actions.jsonl' if endpoint=='initial' else endpoint+'/all14-actual-native-actions.jsonl')
        deliveries=SOURCE/('initial-actual-programs-and-public-executions.jsonl' if endpoint=='initial' else endpoint+'/all14-actual-generated-programs-and-public-executions.jsonl')
        actual=reconstruct(data.rows,records(events),records(deliveries),tokenizer.decode,instruction_code)
        programs[endpoint]=actual;paths.extend([events,deliveries])
    public=list(map(json.loads,gzip.open(PACK/'original-rows.jsonl.gz','rt')))
    donors={r['source_candidate']['original_source_git_path']+':'+r['source_candidate']['function']:r for r in public}
    fixed=[dict(position=i,source_id=r['id'],function=NAMES[i],optimization_role=data.rows[i]['optimization_role'],domain=DOMAINS[NAMES[i]],cases=cases(r)) for i,r in enumerate(rows)]
    paths += [PACK/name for name in [*FILES,'manifest.json','admission.json']]
    paths += python_sources(ROOT,[Path(__file__),ROOT/'tests/test_function_semantic_stress_v1.py',ROOT/'tests/test_public_semantic_source_contrast_v1.py'])
    files={**data.source_files_sha256,**{str(p):file_hash(p) for p in set(paths)}};guard=FileBindings(files)
    OUT.mkdir();DATA.mkdir()
    for name in ['tokenizer.json','model-config.json','DATASET-LICENSE.txt','publisher-README.md']:shutil.copyfile(PACK/name,DATA/name)
    atomic_json(OUT/'fixed-cases.json',fixed)
    atomic_json(OUT/'plan.json',dict(owner=dict(pid=os.getpid(),ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]),
        at=datetime.now(timezone.utc).isoformat(),source_files_sha256=files,all_original_source_functions=14,
        actual_model_endpoints=['initial','chosen_CE','trajectory_DPO'],fixed_case_file_sha256=file_hash(OUT/'fixed-cases.json'),
        fixed_before_fresh_runtime_observations=True,designed_after_original_source_and_model_output_inspection=True,
        not_predeclared_before_original_model_outputs=True,source_train_diagnostic_not_independent_benchmark=True,
        not_gate_binary_domain_already_exhausted_no_fresh_arguments_claimed=True,workers=2,CPU_nice=os.nice(0),
        no_original_training_or_benchmark_population_replaced=True,new_neural_forwards=0,new_neural_updates=0,goal_achieved=False))
    jobs=[('original_public',i,r['response']) for i,r in enumerate(rows)]
    jobs += [(endpoint,i,r['program']) for endpoint,values in programs.items() for i,r in enumerate(values)]
    def work(job):
        endpoint,i,program=job;selected=fixed[i]['cases'];actual=[observe(program,NAMES[i],selected) for _ in range(2)]
        if effect(actual[0])!=effect(actual[1]):raise ValueError('Every original source/model program must reproduce both complete fresh-input observations')
        return dict(endpoint=endpoint,position=i,source_id=rows[i]['id'],function=NAMES[i],optimization_role=data.rows[i]['optimization_role'],
            fresh_argument_cases=sum(c['arguments_not_in_original_public_examples'] for c in selected),program_sha256=actual[0]['program_sha256'],
            program_body_AST_sha256=code_digest(program),body_AST_matches_original_public=code_digest(program)==code_digest(rows[i]['response']),
            actual_runs=actual)
    outputs=[];candidates=[];writer=CodePackWriter(DATA,maximum_bytes=32*1024**2,minimum_free_gib=26)
    try:
        with (OUT/'all-original-and-model-fresh-input-runs.jsonl').open('x') as stream,ThreadPoolExecutor(max_workers=2) as pool:
            for result in pool.map(work,jobs):
                outputs.append(result);stream.write(json.dumps(result)+'\n');stream.flush()
                atomic_json(OUT/'progress.json',dict(complete_program_pairs=len(outputs),planned_program_pairs=56,actual_neural_updates=0,goal_achieved=False))
        negatives={r['function']:r for r in outputs if r['endpoint']=='original_public' and not r['actual_runs'][0]['all_cases_passed']}
        for i,row in enumerate(rows):
            name=NAMES[i]
            if name not in negatives:continue
            if name not in DONORS:raise ValueError('An unexpected source failure needs an explicit new review; never silently omit it')
            donor=donors[DONORS[name]];program=public_alternative(row,donor)
            good=[observe(program,name,fixed[i]['cases']) for _ in range(2)]
            published_original=[public_observe(row,row['response']) for _ in range(2)]
            published_new=[public_observe(row,program) for _ in range(2)]
            published_donor=[public_observe(donor,donor['response']) for _ in range(2)]
            if (any(not r['all_cases_passed'] for r in good)
                    or any(not r['all_original_examples_passed'] for r in [*published_original,*published_new,*published_donor])):
                raise ValueError('Original and donor full public examples plus every fresh numerical case must qualify')
            candidate,ids,labels,pair=make(row,donor,negatives[name]['actual_runs'][0],tokenizer)
            writer.add(candidate,ids,labels)
            candidates.append(dict(function=name,source_id=row['id'],donor_id=donor['id'],candidate=candidate,pair=pair,
                actual_positive_numerical_runs=good,actual_original_public_runs=published_original,
                actual_repair_original_public_runs=published_new,actual_donor_public_runs=published_donor))
        manifest=writer.finish(dict(format=FORMAT,source_files_sha256=files,original_fourteen_functions_preserved=True,
            actual_original_source_failures=len(negatives),actual_complete_public_correction_candidates=len(candidates),
            same_pinned_MIT_public_repository=True,chosen_is_unchanged_public_donor_plus_static_alias=True,
            rejected_is_whole_original_public_wrong_implementation=True,finite_numerical_qualification_only=True,
            machine_source_contrasts_not_human_or_on_policy_RL=True,independent_native_source_and_execution_readback_required=True))
        with (DATA/'all-complete-source-preference-pairs.jsonl').open('x') as stream:
            for value in candidates:stream.write(json.dumps(value['pair'])+'\n')
        atomic_json(OUT/'all-public-correction-candidates-and-executions.json',candidates)
        summary={}
        for endpoint in ['original_public',*programs]:
            selected=[r for r in outputs if r['endpoint']==endpoint];summary[endpoint]={}
            for role in ['optimization_source_train','unoptimized_source_train_control']:
                members=[r for r in selected if r['optimization_role']==role]
                summary[endpoint][role]=dict(functions=len(members),syntax_valid=sum(r['actual_runs'][0]['syntax_valid'] for r in members),
                    all_domain_cases_passed=sum(r['actual_runs'][0]['all_cases_passed'] for r in members),
                    functions_with_fresh_arguments=sum(r['fresh_argument_cases']>0 for r in members),
                    all_fresh_argument_cases_passed=sum(r['fresh_argument_cases']>0 and r['actual_runs'][0]['all_cases_passed'] for r in members),
                    fresh_arguments=sum(r['fresh_argument_cases'] for r in members),
                    body_AST_matches_original_public=sum(r['body_AST_matches_original_public'] for r in members))
        actual_runs=sum(run['actual_program_launched'] for row in outputs for run in row['actual_runs'])+8*len(candidates)
        guard.check(full=True)
        atomic_json(OUT/'complete.json',dict(complete=True,at=datetime.now(timezone.utc).isoformat(),all56source_and_model_program_pairs_retained=True,
            actual_matched_post_hoc_source_train_results=summary,original_public_implementations_with_counterexamples=list(negatives),
            actual_fresh_confined_program_runs=actual_runs,actual_complete_public_correction_candidates=len(candidates),
            native_candidate_pack_counts=manifest['counts'],full_independent_readback_still_required=True,training_admitted=False,
            original_live_neural_jobs_and_benchmark_populations_unchanged=True,source_train_diagnostic_not_independent_benchmark=True,
            source_files_sha256={**files,**{str(p):file_hash(p) for p in [OUT/'fixed-cases.json',OUT/'plan.json',
                OUT/'all-original-and-model-fresh-input-runs.jsonl',OUT/'all-public-correction-candidates-and-executions.json',
                DATA/'manifest.json',DATA/'all-complete-source-preference-pairs.jsonl',*[DATA/name for name in FILES]]}},
            candidate_promoted=False,new_neural_forwards=0,new_neural_updates=0,goal_achieved=False))
        print(json.dumps(dict(summary=summary,public_correction_candidates=len(candidates),training_admitted=False)),flush=True)
    except BaseException as error:
        if not writer.tokens.closed:writer.close()
        atomic_json(OUT/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),automatic_retry=False,goal_achieved=False));raise


if __name__=='__main__':main()
