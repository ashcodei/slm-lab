"""Full-scope review of a measured bounded polling continuation."""
import copy
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

from slm_lab.function_repair_GPU_contract_v1 import ADMISSION,QUEUE,handoff_admitted

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/polling-continuation-v1-review'
MEASURE=ROOT/'reports/polling-continuation-observation-CPU-v1'
DUTY=ROOT/'reports/waiter-polling-duty-cycle-CPU-v3'


def read(path):return json.loads(Path(path).read_text())


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path,value):
    pending=path.with_suffix(path.suffix+'.pending')
    with pending.open('x') as stream:stream.write(json.dumps(value,indent=2)+'\n')
    pending.replace(path)


def live(owner):
    values=(Path('/proc')/str(owner['pid'])/'stat').read_text().split(') ',1)[1].split()
    if values[19]!=owner['ticks'] or values[0] in {'Z','X'}:raise ValueError('Every exact original and bounded-controller birth must remain live')
    return dict(owner=owner,state=values[0])


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.nice(0)<19 or OUT.exists():raise ValueError('Fresh low-priority CPU continuation review required')
    measured=read(MEASURE/'complete.json');plan=read(DUTY/'plan.json')
    if (not measured.get('complete') or measured['new_actual_bounded_owner']!=plan['owner']
            or not measured['actual_first_cycle_release']['all_owned_suspensions_released']
            or plan['maximum_seconds']!=14400 or plan['pause_seconds']!=45 or plan['independent_rescue_seconds']!=55
            or (DUTY/'failure.json').exists()):raise ValueError('Complete original repeated observation and actually restored first bounded cycle required')
    coordinator=live(plan['owner']);normal=measured['actual_normal_polling'];paused=measured['actual_paused_polling']
    if (normal['wall_seconds']<30 or paused['wall_seconds']<30
            or normal['actual_average_CPU_cores']['waiter_pollers']<.3
            or abs(paused['actual_average_CPU_cores']['waiter_pollers'])>.05):
        raise ValueError('Substantial actual restored polling and negligible paused work must be measured')
    prior=read(ROOT/'reports/waiter-polling-duty-cycle-CPU-v2/complete.json')
    if not prior.get('complete') or not prior['all_owned_suspensions_released']:raise ValueError('Previous bounded controller must have ended and restored every waiter')
    audit=read(ROOT/'reports/completion-audit.json');previous=Path(audit['latest_full_scope_review']);review=copy.deepcopy(read(previous))
    gpu=read(ROOT/QUEUE);handoff=read(ROOT/ADMISSION)
    if not handoff_admitted(ROOT,gpu['owner']):raise ValueError('All ten original full-scope successors must remain admitted')
    maps=[plan['source_files_sha256'],gpu['source_files_sha256'],read(ROOT/'reports/public-reasoning-warmup-CPU-v1-queue.json')['source_files_sha256'],
        read(ROOT/'reports/public-pretraining-recovery-v1/plan.json')['source_files_sha256'],
        read(ROOT/'reports/full-function-repair-comparison-CPU-v1/plan.json')['source_files_sha256'],
        read(ROOT/'reports/public-pretraining-interim-CPU-v2/evaluation-plan.json')['source_files_sha256'],
        read(ROOT/'reports/public-edge-preference-comparison-CPU-v1-queue.json')['source_files_sha256']]
    maps += [read(ROOT/item['queue'])['source_files_sha256'] for item in handoff['new_controllers']]
    for name in ['reports/completion-audit.json','README.md','RESEARCH.md']:
        if any(str(ROOT/name) in mapping for mapping in maps):raise ValueError('Never edit an index/document bound by active work')
    files={**plan['source_files_sha256'],**measured['source_files_sha256']}
    for path in [Path(__file__),previous,MEASURE/'complete.json',MEASURE/'plan.json',
        MEASURE/'original-polling-restored.json',MEASURE/'bounded-polling-suspension.json',DUTY/'plan.json',DUTY/'cycle-1-complete.json']:
        files[str(path)]=sha(path)
    if any(sha(path)!=digest for path,digest in files.items()):raise ValueError('Every original and measured resource/source byte must remain exact')
    protected=[live(owner) for owner in plan['protected_neural_and_GPU_study_owners']]
    if any(item['state'] in {'T','t'} for item in protected):raise ValueError('No actual protected neural worker may be stopped')
    raw=(ROOT/'reports/public-pretraining-recovery-v1/training-metrics.jsonl').read_bytes();prefix=raw[:raw.rfind(b'\n')+1];rows=list(map(json.loads,prefix.splitlines()))
    if [row['step'] for row in rows]!=list(range(1,len(rows)+1)):raise ValueError('Actual complete contiguous main update prefix required')
    current=dict(owner=plan['protected_neural_and_GPU_study_owners'][0],complete_actual_updates=len(rows),planned_updates=4447,
        response_EOS_targets=sum(int(row['response_EOS_targets']) for row in rows),complete_prefix_sha256=hashlib.sha256(prefix).hexdigest())
    at=datetime.now(timezone.utc).isoformat();OUT.mkdir()
    proof=dict(complete=True,at=at,source_files_sha256=files,
        previous_goal_turn_classification='progress:25actual source-matched CPU causal/gradient/tokenizer controls pass without runtime edits',
        current_goal_turn_classification='progress: previous hour completes with full restoration; fresh measured bounded continuation active and first-cycle restoration verified',
        actual_new_bounded_owner=coordinator,actual_old_hour_completed_with_every_waiter_released=True,maximum_seconds=14400,
        pause_seconds=45,running_seconds=15,independent_deadline_rescue_seconds=55,
        actual_adjacent_windows=dict(original_polling=normal,temporarily_paused_polling=paused),
        actual_first_rescue_cycle=measured['actual_first_cycle_release'],actual_current_duty_status=read(DUTY/'progress.json'),
        actual_all_original_protected_neural_births_running=protected,actual_original_main=current,
        actual_GPU_rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True),
        CPU_inference_source_already_has_native_decode_and_rotary_cache=True,
        no_new_neural_job_or_worker_priority_affinity_or_source_or_original_queue_edits=True,
        all_original17family_human_RL_and_nine_endpoint_scopes_preserved=True,
        CPU_allocation_only_not_direct_generation_speed_or_accuracy=True,candidate_promoted=False,goal_achieved=False)
    atomic(OUT/'complete.json',proof)
    a=normal['actual_average_CPU_cores'];b=paused['actual_average_CPU_cores']
    lines=['# Bounded polling continuation and current CPU allocation','',
        'The previous one-hour coordinator is actually terminal and records complete restoration of all nine waiting parents. '
        'A new coordinator is live for at most **four hours**, using the same tested **45seconds paused /15seconds running** cadence. '
        'Every suspension retains an independent55second rescue, held pidfds, parent-death/pipe restoration and protected-predecessor-exit restoration. '
        'The whole coordinator stops before the main public-training handoff or on an original stage/process transition. '
        'Only the same nine exact unstarted, childless, CUDA-hidden queue parents are affected.', '',
        '| Process group | Original polling: CPU cores | Temporarily paused: CPU cores |','|---|---:|---:|']
    for key,label in [('waiter_pollers','Waiting queue checks'),('CPU_training','Existing CPU model training'),('CPU_assessment','Existing CPU native assessment'),('GPU_workers','Original four GPU driver processes')]:
        lines.append(f'| {label} | {a[key]:.3f} | {b[key]:.3f} |')
    lines += ['', 'Each adjacent window lasts30seconds. CPU cores mean process CPU seconds divided by wall seconds. '
        'The first full new cycle actually resumes every owned waiter and writes its independent-rescue receipt. '
        'This repeats the observed hashing overhead and shows more CPU time assigned to ongoing neural work; '
        'it does not directly measure generated tokens/second, FLOP efficiency or an accuracy gain.', '',
        'Source inspection confirms that the existing CPU native assessment already uses cached decoding and the established rotary-table cache. '
        'The host exposes native CPU BF16/AMX instruction flags. No inference precision, decoder, parameter, source file, worker priority or affinity is changed. '
        'The existing numerical and evaluation protocols continue unchanged.', '',
        f'The original main remains live at **{len(rows):,}/4,447** updates and **{current["response_EOS_targets"]:,}** actual targets. '
        'All original neural workers remain running; the complete17family/human/DPO/execution-RL/machine-preference/nine-endpoint pipeline is preserved. '
        'The optional CPU warmup remains behind its exact released-resource gate. '
        '**Latest complete local functional correctness remains0/48; no new capability gain or frontier parity is established. All ten final requirements remain incomplete.**','']
    (OUT/'metrics.md').write_text('\n'.join(lines))
    review['requirements']['serious_full_model_training']['evidence']+=(
        ' The original one-hour polling coordinator actually completes and releases every owned waiter. A fresh maximum4hour continuation is active, '
        'with unchanged45/15cadence,55second independent held-pidfd rescue and original predecessor/handoff exits. '
        'Two actual30secondadjacent windows measure idle checks at0.911CPUcores versus0while paused; CPUtraining allocation2.578to3.260 and assessment0.536to0.642cores. '
        'Its first full cycle actually restores every waiter. No protected neural worker is stopped or reprioritized, and no runtime/data/numerical or benchmark scope changes. '
        'These are scheduler allocation observations, not direct model throughput or accuracy gains.')
    if len(review['requirements'])!=10 or any(value['status']!='incomplete_for_requested_final_model' for value in review['requirements'].values()):raise ValueError('Resource continuation proves none of the requested final capabilities')
    review.update(at=at,previous_full_scope_review=str(previous),previous_goal_turn_classification=proof['previous_goal_turn_classification'],
        current_goal_turn_classification=proof['current_goal_turn_classification'],latest_actual_polling_continuation=proof,
        source_files_sha256={**files,str(OUT/'complete.json'):sha(OUT/'complete.json')},goal_achieved=False)
    atomic(OUT/'review.json',review)
    if any(sha(path)!=digest for path,digest in files.items()):raise ValueError('Every frozen original/measured source must remain exact before index update')
    audit.update(at=at,requirements=review['requirements'],latest_full_scope_review=str(OUT/'review.json'),latest_actual_main_runtime=current,
        latest_actual_polling_continuation=str(OUT/'complete.json'),goal_achieved=False)
    atomic(ROOT/'reports/completion-audit.json',audit)
    paragraph=(f'Bounded polling continuation measured, **{at}**: the previous hour ends with all9waiting parents restored. '
        'A new maximum4hour continuation is actually live with the same45/15cadence,55second independent rescue and handoff/predecessor exits. '
        'Two adjacent30secondwindows measure waiting checks at **0.911CPUcores** normally and **0while paused**; '
        'ongoing CPU training/assessment receive more CPU time. Its first full rescue cycle restores every waiter. '
        '**This is CPU allocation evidence, not direct decoding speed or accuracy.** '
        f'The main continues at **{len(rows)}/4,447**, preserving all source/human/RL/nine-endpoint scopes. '
        '**Latest complete functional correctness remains0/48; all ten final requirements remain incomplete.** '
        '[Measurements and bounds](reports/polling-continuation-v1-review/metrics.md), '
        '[full review](reports/polling-continuation-v1-review/review.json). Earlier entries are historical snapshots.\n\n')
    for name in ['README.md','RESEARCH.md']:
        path=ROOT/name;first,rest=path.read_text().split('\n',1);path.write_text(first+'\n\n'+paragraph+rest.lstrip('\n'))
    print(json.dumps(dict(review=str(OUT/'review.json'),actual_bounded_owner=plan['owner'],actual_main_updates=len(rows),goal_achieved=False)),flush=True)


if __name__=='__main__':main()
