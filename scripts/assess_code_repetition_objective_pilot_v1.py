"""Queue both frozen objective endpoints after the earlier full assessment."""
from datetime import datetime,timezone
import gc
import json
from pathlib import Path
import shutil
import subprocess
import time

import torch

import assess_codenet_model_v2 as code_assessment
from assess_code_residual_lr_pilot_v1 import compare_fixed_results
from slm_lab import knowledge_evaluation
from slm_lab.code_repetition_inference_v1 import load_objective_policy,PROFILES
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT=Path(__file__).resolve().parents[1]


def live(receipt):
    try:
        fields=(Path('/proc')/str(receipt['pid'])/'stat').read_text().rsplit(')',1)[1].split()
        return fields[19]==str(receipt['start_ticks']) and fields[0] not in {'Z','X'}
    except FileNotFoundError:return False


def full_reservation():
    return dict(credited_bytes=0,verified_files=[],full_required_bytes=26*1024**3,
        policy='entire original26GiB writer reserve; zero occupied-file credit')


def main():
    reports=ROOT/'reports/code-repetition-objective-assessment-v1'
    if reports.exists():raise ValueError('Preserve previous assessment; no implicit restart')
    launch_path=ROOT/'reports/code-repetition-objective-pilot-v1-launch.json'
    predecessor_path=ROOT/'reports/code-residual-repeated-source-assessment-v1-launch.json'
    launch=json.loads(launch_path.read_text());predecessor=json.loads(predecessor_path.read_text())
    names=['scripts/assess_codenet_model_v2.py','scripts/assess_code_residual_lr_pilot_v1.py',
        'scripts/train_code_repetition_objective_pilot_v1.py','src/slm_lab/code_repetition_inference_v1.py',
        'src/slm_lab/code_repetition_objective_v1.py','src/slm_lab/code_residual_branch_v1.py',
        'src/slm_lab/code_residual_checkpoint_v1.py','src/slm_lab/model.py','src/slm_lab/training.py',
        'src/slm_lab/knowledge_evaluation.py','src/slm_lab/codenet_model_assessment_v1.py',
        'tests/test_code_repetition_inference_v1.py','reports/code-repetition-inference-v1-focused-tests.log',
        'data/codenet-algorithm-assessment-v1/manifest.json','data/knowledge-eval-v1/questions.jsonl',
        'data/knowledge-eval-v1/questions.audit.json']
    bound={str(ROOT/n):file_hash(ROOT/n) for n in names}
    for p in [Path(__file__),launch_path,predecessor_path]:bound[str(p)]=file_hash(p)
    guard=FileBindings(bound);reports.mkdir()
    def status(message,**fields):
        atomic_json(reports/'status.json',dict(status=message,updated_at=datetime.now(timezone.utc).isoformat(),
            candidate_promoted=False,goal_achieved=False,**fields))
    atomic_json(reports/'plan.json',dict(created_at=datetime.now(timezone.utc).isoformat(),source_files_sha256=bound,
        exact_training_actor=launch,exact_preceding_assessment_actor=predecessor,profiles=list(PROFILES),
        fixed_final_endpoints_without_selection=True,code_questions_per_profile=28,code_generation_budget=2048,
        raw_knowledge_questions_per_profile=559,raw_knowledge_generation_budget=32,
        primary_scoring_or_greedy_decoding_changes=False,assessment_optimizer_updates=0,
        minimum_home_free_bytes=26*1024**3+256*1024**2,
        resource_policy='Entire original26GiB writer budget, zero occupied-file credit, plus256MiB total code/QA reports',
        GPU='cuda:0',maximum_GPU_memory_fraction=.14,minimum_initial_GPU_free_bytes=12*1024**3,
        all_candidate_assessments_sequential_after_exact_previous_assessment=True,
        automatic_restarts_or_promotion=False,goal_achieved=False))
    try:
        while live(launch):
            guard.check();status('waiting for exact live matched-objective training actor',
                preceding_pid=launch['pid'],preceding_start_ticks=launch['start_ticks']);time.sleep(5)
        training=ROOT/'reports/code-repetition-objective-pilot-v1'
        result_path=training/'results.json';trained=json.loads(result_path.read_text())
        plan_path=training/'plan.json';plan=json.loads(plan_path.read_text())
        if (trained['actual_Adam_updates_total']!=256 or trained['complete_original_response_target_exposures']!=609150
                or trained['held_source_rows_used_for_updates'] or not trained['fixed_final_states_without_selection']):
            raise ValueError('Completed predeclared matched objective endpoints required')
        bound.update({str(result_path):file_hash(result_path),str(plan_path):trained['plan_sha256']})
        for result in trained['profiles']:
            profile=result['profile'];checkpoint=Path(result['checkpoint']);frozen=training/profile['name']/'frozen-final-state.json'
            if profile not in PROFILES or result['actual_updates']!=128 or result['response_target_exposures']!=304575:
                raise ValueError('Every complete fixed128-update candidate required')
            if json.loads(frozen.read_text())['checkpoint_sha256']!=result['checkpoint_sha256']:
                raise ValueError('Candidate must have been frozen before generation feedback')
            bound[str(checkpoint)]=result['checkpoint_sha256'];bound[str(frozen)]=file_hash(frozen)
            bound[str(checkpoint.parent/'tokenizer.json')]=file_hash(checkpoint.parent/'tokenizer.json')
        base=Path(plan['original_base_checkpoint']);bound[str(base)]=plan['original_base_checkpoint_sha256']
        guard=FileBindings(bound);guard.check(full=True)
        while live(predecessor):
            guard.check();status('waiting for exact earlier full assessment before evaluating new endpoints',
                preceding_pid=predecessor['pid'],preceding_start_ticks=predecessor['start_ticks']);time.sleep(15)
        prior_comparison=ROOT/'reports/code-residual-repeated-source-assessment-v1/matched-comparison.json'
        prior=json.loads(prior_comparison.read_text())
        if not prior['every_original559_question_gold_and_score_recomputed']:
            raise ValueError('Inspect incomplete preceding assessment before proceeding')
        bound[str(prior_comparison)]=file_hash(prior_comparison);guard=FileBindings(bound)
        comparisons=[]
        for result in trained['profiles']:
            profile=result['profile'];checkpoint=Path(result['checkpoint']);token=checkpoint.parent/'tokenizer.json'
            while True:
                guard.check();free=shutil.disk_usage(ROOT).free
                gpu=[int(v) for v in subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()]
                if free>=26*1024**3+256*1024**2 and len(gpu)==4 and gpu[0]*1024**2>=12*1024**3:break
                status('waiting for funded original reserve and shared GPU capacity',profile=profile,
                    free_home_bytes=free,required_home_bytes=26*1024**3+256*1024**2,GPU_free_MiB=gpu)
                time.sleep(15)
            def loader(path,device='cpu',training_state=True):
                if Path(path).resolve()!=checkpoint.resolve() or training_state is not False:
                    raise ValueError('Only the exact frozen objective inference state may load')
                return load_objective_policy(checkpoint,base,token,
                    expected_checkpoint_sha256=result['checkpoint_sha256'],expected_base_sha256=plan['original_base_checkpoint_sha256'],
                    expected_plan_sha256=trained['plan_sha256'],expected_starting_policy_sha256=plan['starting_policy_sha256'],
                    expected_selection_sha256=plan['original_selection_sha256'],expected_profile=profile,device=device)
            code_assessment.load_checkpoint=loader;code_assessment.occupied_writer_reservation=full_reservation
            sub=reports/profile['name'];sub.mkdir();code_folder=sub/'code'
            status('generating all28 unchanged raw code questions',profile=profile)
            code_assessment.evaluate(checkpoint,code_folder,'cuda:0',threads=1,memory_fraction=.14,
                expected_metadata=dict(stage='code_repetition_objective_pilot_v1',steps=128,planned_steps=128,
                    data_sha256=file_hash(ROOT/'data/codenet-durable-pack-v2/manifest.json')),
                label='own857M_fixed_objective_'+profile['name'],expected_checkpoint_sha256=result['checkpoint_sha256'])
            guard.check(full=True);gc.collect();torch.cuda.empty_cache()
            knowledge_evaluation.load_checkpoint=loader;knowledge=sub/'code-route-raw-knowledge.json'
            status('evaluating all559 unchanged raw knowledge questions',profile=profile)
            knowledge_evaluation.evaluate(checkpoint,ROOT/'data/knowledge-eval-v1/questions.jsonl',knowledge,
                device='cuda:0',max_examples=559,max_new_tokens=32)
            guard.check(full=True)
            temporary=Path('/tmp')/f'slm-objective-compare-{__import__("os").getpid()}.json'
            comparison=compare_fixed_results(code_folder,knowledge,temporary);temporary.unlink()
            comparison.update(objective_profile=profile,training_checkpoint_sha256=result['checkpoint_sha256'],
                scope='One fixed128-update endpoint of a matched own857M CE versus CE+repetition-unlikelihood experiment on512 original programs and128 original-train-prompt rollouts. Unchanged28 code/2048 tokens and559 raw QA/32 tokens. Original reused development/earlier-exposure qualifications remain; no automatic routing, model selection, promotion or frontier claim.')
            atomic_json(sub/'matched-comparison.json',comparison);comparisons.append(comparison)
            gc.collect();torch.cuda.empty_cache()
        atomic_json(reports/'matched-objective-comparisons.json',dict(completed_at=datetime.now(timezone.utc).isoformat(),
            profiles=comparisons,no_candidate_selected_or_promoted=True,primary_scores_recomputed=True,goal_achieved=False))
        status('both objective endpoints completed unchanged code and knowledge assessment')
    except Exception as error:
        status('objective assessment stopped; preserve evidence and inspect',error=str(error));raise


if __name__=='__main__':main()
