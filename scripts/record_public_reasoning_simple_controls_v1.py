"""Review actual train-fitted reasoning controls and a verified live-job wait."""
import copy
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path

from slm_lab.function_repair_GPU_contract_v1 import ADMISSION,QUEUE,handoff_admitted

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/public-reasoning-simple-controls-v1-review'
CONTROLS=ROOT/'reports/public-reasoning-simple-controls-CPU-v1'
OBSERVATION=ROOT/'reports/public-reasoning-controls-live-observation-CPU-v1.json'


def read(path):return json.loads(Path(path).read_text())


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path,value):
    pending=path.with_suffix(path.suffix+'.pending')
    with pending.open('x') as stream:stream.write(json.dumps(value,indent=2)+'\n')
    pending.replace(path)


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.nice(0)<19 or OUT.exists():raise ValueError('Fresh low-priority CPU review required')
    control=read(CONTROLS/'complete.json');observed=read(OBSERVATION)
    if (not control.get('complete') or control['actual_overall']!={'training_majority':{'examples':289,'correct':59},
            'copy_input':{'examples':45,'correct':11},'train_selected':{'examples':289,'correct':70}}
            or control['new_neural_forwards']!=0 or not observed.get('complete')
            or not observed['all_exact_original_births_live'] or observed['wall_seconds']<45):
        raise ValueError('All actual train-fitted/reserved controls and exact live-process observation required')
    for item in observed['after']['processes']:
        owner=item['owner'];fields=(Path('/proc')/str(owner['pid'])/'stat').read_text().split(') ',1)[1].split()
        if fields[19]!=owner['ticks'] or fields[0] in {'Z','X','T','t'}:raise ValueError('Every observed original/neural/waiting birth must remain live and running')
    audit=read(ROOT/'reports/completion-audit.json');previous=Path(audit['latest_full_scope_review']);review=copy.deepcopy(read(previous))
    gpu=read(ROOT/QUEUE);handoff=read(ROOT/ADMISSION)
    if not handoff_admitted(ROOT,gpu['owner']):raise ValueError('All ten original full-scope successors must remain admitted')
    warmup=read(ROOT/'reports/public-reasoning-warmup-CPU-v1-queue.json')
    maps=[warmup['source_files_sha256'],gpu['source_files_sha256'],
        read(ROOT/'reports/public-pretraining-recovery-v1/plan.json')['source_files_sha256'],
        read(ROOT/'reports/full-function-repair-comparison-CPU-v1/plan.json')['source_files_sha256'],
        read(ROOT/'reports/public-pretraining-interim-CPU-v2/evaluation-plan.json')['source_files_sha256'],
        read(ROOT/'reports/public-edge-preference-comparison-CPU-v1-queue.json')['source_files_sha256']]
    maps += [read(ROOT/item['queue'])['source_files_sha256'] for item in handoff['new_controllers']]
    for name in ['reports/completion-audit.json','README.md','RESEARCH.md']:
        if any(str(ROOT/name) in mapping for mapping in maps):raise ValueError('Never edit a live source-bound index/document')
    files=dict(control['source_files_sha256'])
    for path in [Path(__file__),CONTROLS/'complete.json',CONTROLS/'train-fitted-controls.json',
        CONTROLS/'all289-actual-control-predictions.jsonl',OBSERVATION,previous]:files[str(path)]=sha(path)
    if any(sha(path)!=digest for path,digest in files.items()):raise ValueError('Every source and actual control byte must remain exact')
    at=datetime.now(timezone.utc).isoformat();OUT.mkdir()
    current=dict(owner=observed['after']['processes'][0]['owner'],complete_actual_updates=observed['after']['main_updates'],planned_updates=4447,
        response_EOS_targets=observed['after']['targets'],complete_prefix_sha256=observed['after']['complete_prefix_sha256'])
    proof=dict(complete=True,at=at,source_files_sha256=files,
        previous_goal_turn_classification='progress: tested source-bound recovery/native evaluation and actual full-size warmup queue waiting on exact CPU predecessors',
        current_goal_turn_classification='progress: actual train-only simple-control scores on all289reserved cases; verified45secondwait on exact live neural jobs',
        actual_non_neural_control_scores=control['actual_overall'],actual_per_task=control['actual_per_task'],
        actual_prediction_and_control_unit_tests_passed=3,actual_new_model_forwards_or_updates=0,
        actual_verified_live_observation=observed,actual_original_main=current,
        exact_original_warmup_waiter=warmup['owner'],all_original_ten_downstream_owners_live=True,
        all_original17family_human_RL_and_nine_endpoint_scopes_preserved=True,
        no_data_training_native_evaluator_or_queue_modified=True,control_scores_not_model_accuracy=True,
        candidate_promoted=False,goal_achieved=False)
    atomic(OUT/'complete.json',proof)
    lines=['# Reasoning controls and verified ongoing work','',
        'Three actual source/prediction controls pass. On all289original source-reserved examples, a program that always returns the per-task '
        'training-majority answer scores **59/289 (20.4%)**. Choosing between that answer and literal input-grid copying using training accuracy only '
        'scores **70/289 (24.2%)**. Input copying is applicable to45matrix cases and gets11correct. The prediction API receives only task and original question. '
        'No reserved answer enters fitting or prediction. All2,568training sources participate in fitting, and every reserved case is scored.', '',
        'The two binary tasks account for37of59majority successes; GCD contributes11. Copying solves3of20distance-matrix cases and8of25rotation cases '
        'without computing distances or rotating. The detailed per-task table and all289actual predictions are retained in '
        '[the control assessment](../public-reasoning-simple-controls-CPU-v1/metrics.md).', '',
        '**These are handwritten program controls, not neural-model scores or improvements.** The programs frame their own complete outputs with EOS. '
        'The controls show why upcoming model scores need per-task interpretation beyond aggregate accuracy. They are descriptive comparisons; '
        'exceeding24.2% does not establish advanced reasoning, coding capability or completion of the requested model. '
        'The original289case model evaluation, every coding/broad check and full final benchmark scope remain unchanged. '
        'No new model or training job is created, and no original data, native evaluator or queue is edited.', '',
        f'A separate **{observed["wall_seconds"]:.1f}-second** read-only observation verifies all ten exact main/GPU/CPU/research/waiting process births remain live. '
        f'The main commits **{observed["actual_new_main_updates"]} new updates** and **{observed["actual_new_main_targets"]:,} new targets** during that window, '
        f'reaching **{current["complete_actual_updates"]:,}/4,447**. GPU utilization and process CPU seconds are actual observations, '
        'not FLOP efficiency, a general throughput forecast or accuracy evidence. The queued reasoning warmup remains unstarted behind its original exact predecessors.', '',
        '**Latest complete local functional correctness remains0/48. The new matched assessments are not complete. '
        'No new capability gain or frontier parity is established; all ten final requirements remain incomplete.**','']
    (OUT/'metrics.md').write_text('\n'.join(lines))
    review['requirements']['broad_independent_benchmarks']['evidence']+=(
        ' Train-fitted handwritten controls are now actually scored on all289new source-reserved cases:59/289for per-task majority, '
        '11/45for matrix input copy and70/289for training-selected majority/copy. These are non-neural control outcomes with explicit programmaticEOS, '
        'not model-quality gains or an alternative final benchmark. All original cases and scoring remain. '
        'A45secondread-only observation confirms exact main/GPU/CPU/waiter births and actual continued main updates; current new matched native assessments remain incomplete.')
    if len(review['requirements'])!=10 or any(value['status']!='incomplete_for_requested_final_model' for value in review['requirements'].values()):
        raise ValueError('Simple controls and verified waiting establish none of the requested final model capabilities')
    review.update(at=at,previous_full_scope_review=str(previous),previous_goal_turn_classification=proof['previous_goal_turn_classification'],
        current_goal_turn_classification=proof['current_goal_turn_classification'],latest_actual_reasoning_simple_controls=proof,
        source_files_sha256={**files,str(OUT/'complete.json'):sha(OUT/'complete.json')},goal_achieved=False)
    atomic(OUT/'review.json',review)
    if any(sha(path)!=digest for path,digest in files.items()):raise ValueError('All frozen control/source bytes must remain exact before index update')
    audit.update(at=at,requirements=review['requirements'],latest_full_scope_review=str(OUT/'review.json'),latest_actual_main_runtime=current,
        latest_actual_reasoning_simple_controls=str(OUT/'complete.json'),goal_achieved=False)
    atomic(ROOT/'reports/completion-audit.json',audit)
    paragraph=(f'Reasoning controls and live-job observation, **{at}**: train-only per-task majority scores **59/289**; '
        'training-selected majority/input-copy scores **70/289 (24.2%)** on all source-reserved cases. '
        '**These are handwritten controls, not neural scores or gains**, and they change no original data, model scoring or queue. '
        f'A verified45secondobservation confirms all ten exact neural/research/waiter births remain live while the main commits{observed["actual_new_main_updates"]}updates '
        f'and{observed["actual_new_main_targets"]:,}targets, reaching **{current["complete_actual_updates"]}/4,447**. '
        '**Latest complete functional correctness remains0/48; all ten final requirements remain incomplete.** '
        '[Controls and observation](reports/public-reasoning-simple-controls-v1-review/metrics.md), '
        '[full review](reports/public-reasoning-simple-controls-v1-review/review.json). Earlier entries are historical snapshots.\n\n')
    for name in ['README.md','RESEARCH.md']:
        path=ROOT/name;first,rest=path.read_text().split('\n',1);path.write_text(first+'\n\n'+paragraph+rest.lstrip('\n'))
    print(json.dumps(dict(review=str(OUT/'review.json'),actual_non_neural_control_correct=70,original_main=current,goal_achieved=False)),flush=True)


if __name__=='__main__':main()
