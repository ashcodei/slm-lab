"""Assess one fixed-end repeated-source candidate, retaining original resource guards."""
from datetime import datetime, timezone
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
from slm_lab.code_residual_repeated_inference_v1 import load_repeated_source_policy
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT=Path(__file__).resolve().parents[1]


def live(launch):
    try:
        fields=(Path('/proc')/str(launch['pid'])/'stat').read_text().rsplit(')',1)[1].split()
        return fields[19]==str(launch['start_ticks']) and fields[0] not in {'Z','X'}
    except FileNotFoundError:
        return False


def full_writer_reservation():
    # The historical helper is tied to an old writer PID. This reserves all
    # 26GiB and grants zero occupied-file credit, as in the completed v2 audit.
    return dict(credited_bytes=0, verified_files=[], full_required_bytes=26*1024**3,
        policy='entire original26GiB writer reserve; zero occupied-file credit')


def main():
    reports=ROOT/'reports/code-residual-repeated-source-assessment-v1'
    if reports.exists():
        raise ValueError('Preserve earlier assessment; no implicit restart')
    launch_path=ROOT/'reports/code-residual-repeated-source-pilot-v1-launch.json'
    launch=json.loads(launch_path.read_text())
    names=['scripts/assess_codenet_model_v2.py','scripts/assess_code_residual_lr_pilot_v1.py',
        'scripts/train_code_residual_repeated_source_pilot_v1.py',
        'src/slm_lab/code_residual_repeated_inference_v1.py','src/slm_lab/code_residual_branch_v1.py',
        'src/slm_lab/code_residual_checkpoint_v1.py','src/slm_lab/model.py','src/slm_lab/training.py',
        'src/slm_lab/knowledge_evaluation.py','src/slm_lab/codenet_model_assessment_v1.py',
        'tests/test_code_residual_repeated_inference_v1.py',
        'reports/code-residual-repeated-inference-v1-focused-tests.log',
        'reports/code-residual-repeated-source-pilot-v1-reused-mechanics-tests.log',
        'data/codenet-algorithm-assessment-v1/manifest.json',
        'data/knowledge-eval-v1/questions.jsonl','data/knowledge-eval-v1/questions.audit.json']
    bound={str(ROOT/n):file_hash(ROOT/n) for n in names}
    bound[str(Path(__file__))]=file_hash(Path(__file__))
    bound[str(launch_path)]=file_hash(launch_path)
    guard=FileBindings(bound);reports.mkdir()

    def status(message,**fields):
        atomic_json(reports/'status.json',dict(status=message,updated_at=datetime.now(timezone.utc).isoformat(),
            candidate_promoted=False,goal_achieved=False,**fields))

    atomic_json(reports/'plan.json',dict(created_at=datetime.now(timezone.utc).isoformat(),
        source_files_sha256=bound,preceding_exact_training_process=launch,
        candidate_selection='Only predeclared final512-update/four-additional-pass state, before raw generation or tests',
        fixed_code_questions=28,code_generation_budget=2048,raw_QA_questions=559,raw_QA_generation_budget=32,
        evaluation_optimizer_updates=0,primary_generation_or_scoring_sources_changed=False,
        GPU='cuda:0',maximum_GPU_memory_fraction=.14,minimum_initial_GPU_free_bytes=12*1024**3,
        minimum_persistent_free_bytes=26*1024**3+128*1024**2,
        resource_policy='Entire original26GiB reserve, zero occupied-file credit; plus128MiB for code/QA reports',
        storage_wait_does_not_lower_original_preflight=True,maximum_assessment_launches=1,
        automatic_restarts_or_promotion=False,goal_achieved=False))
    try:
        while live(launch):
            guard.check();status('waiting for exact live repeated-source training actor',
                preceding_pid=launch['pid'],preceding_start_ticks=launch['start_ticks'])
            time.sleep(5)
        folder=ROOT/'reports/code-residual-repeated-source-pilot-v1'
        result_path=folder/'results.json'
        trained=json.loads(result_path.read_text())
        plan_path=folder/'plan.json';plan=json.loads(plan_path.read_text())
        frozen_path=folder/'frozen-final-state.json';frozen=json.loads(frozen_path.read_text())
        if (trained['actual_updates'] != 512 or trained['original_training_sources'] != 512
                or trained['response_target_exposures'] != 1218300
                or not trained['fixed_end_model_selected_without_generation_or_test_feedback']
                or trained['validation_or_test_rows_used_for_updates']
                or trained['checkpoint_sha256'] != frozen['checkpoint_sha256']):
            raise ValueError('Actual complete predeclared fixed-end source experiment required')
        checkpoint=Path(trained['checkpoint']);base=Path(plan['original_base_checkpoint'])
        token=checkpoint.parent/'tokenizer.json'
        bound.update({str(result_path):file_hash(result_path),str(plan_path):trained['plan_sha256'],
            str(frozen_path):file_hash(frozen_path),str(checkpoint):trained['checkpoint_sha256'],
            str(base):plan['original_base_checkpoint_sha256'],str(token):file_hash(token)})
        guard=FileBindings(bound);guard.check(full=True)
        atomic_json(reports/'frozen-candidate-binding.json',dict(recorded_at=datetime.now(timezone.utc).isoformat(),
            source_files_sha256=bound,candidate_selected_before_generation_and_test_feedback=True,
            checkpoint=str(checkpoint),training_actor_terminal=True,goal_achieved=False))
        while True:
            guard.check()
            free=shutil.disk_usage(ROOT).free
            gpu_free=[int(v) for v in subprocess.check_output(['nvidia-smi','--query-gpu=memory.free',
                '--format=csv,noheader,nounits'],text=True).splitlines()]
            if free >= 26*1024**3+128*1024**2 and len(gpu_free)==4 and gpu_free[0]*1024**2 >= 12*1024**3:
                break
            status('waiting for funded original26GiB writer reserve and12GiB free on evaluation GPU',
                free_home_bytes=free,required_home_bytes=26*1024**3+128*1024**2,
                GPU_free_MiB=gpu_free,checkpoint_sha256=trained['checkpoint_sha256'],
                assessment_started=False)
            time.sleep(15)

        def loader(path,device='cpu',training_state=True):
            if Path(path).resolve()!=checkpoint.resolve() or training_state is not False:
                raise ValueError('Only the exact frozen own repeated-source inference state may be loaded')
            return load_repeated_source_policy(checkpoint,base,token,
                expected_checkpoint_sha256=trained['checkpoint_sha256'],
                expected_base_sha256=plan['original_base_checkpoint_sha256'],expected_plan_sha256=trained['plan_sha256'],
                expected_selection_sha256=plan['original_selection_sha256'],
                expected_starting_policy_sha256=plan['starting_policy_sha256'],device=device)

        code_assessment.load_checkpoint=loader
        code_assessment.occupied_writer_reservation=full_writer_reservation
        code_folder=ROOT/'reports/code-residual-repeated-source-held-code-v1'
        status('generating all28 unchanged fixed raw code questions for frozen candidate')
        code_assessment.evaluate(checkpoint,code_folder,'cuda:0',threads=1,memory_fraction=.14,
            expected_metadata=dict(stage='code_residual_repeated_source_pilot_v1',steps=512,planned_steps=512,
                data_sha256=file_hash(ROOT/'data/codenet-durable-pack-v2/manifest.json')),
            label='own857M_code_residual_same512_sources_four_additional_passes_fixed_end',
            expected_checkpoint_sha256=trained['checkpoint_sha256'])
        guard.check(full=True);gc.collect();torch.cuda.empty_cache()
        knowledge_evaluation.load_checkpoint=loader
        knowledge_path=reports/'code-route-raw-knowledge.json'
        status('assessing all559 unchanged raw broad-domain questions on frozen code route')
        knowledge_evaluation.evaluate(checkpoint,ROOT/'data/knowledge-eval-v1/questions.jsonl',knowledge_path,
            device='cuda:0',max_examples=559,max_new_tokens=32)
        guard.check(full=True)
        # Reuse complete original prompt/gold/budget/seal/execution validation.
        # Its historical description is replaced with this actual experiment's
        # metadata before a durable comparison is exposed in the report folder.
        temporary=Path('/tmp')/f'slm-repeated-source-comparison-{__import__("os").getpid()}.json'
        comparison=compare_fixed_results(code_folder,knowledge_path,temporary)
        temporary.unlink()
        comparison.update(training_checkpoint_sha256=trained['checkpoint_sha256'],
            training_updates_this_experiment=512,original_train_sources=512,additional_complete_passes=4,
            scope='Fixed-end own857M branch after four additional passes over512 original programs, compared with its exact own706M completed-language base. Unchanged28 code questions/2048 tokens and559 raw QA/32 tokens. Reused development samples with original earlier-corpus/semantic-exposure qualifications, not certified independent final benchmarks. No automatic routing, promotion or frontier capability claim.')
        atomic_json(reports/'matched-comparison.json',comparison)
        status('complete fixed code/QA assessment; inspect unchanged matched capability scores',
            code=comparison['candidate_code'],knowledge=comparison['candidate_code_route_raw_knowledge'])
    except Exception as error:
        status('assessment stopped; preserve evidence and inspect before any continuation',error=str(error))
        raise


if __name__=='__main__':
    main()
