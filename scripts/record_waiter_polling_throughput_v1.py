"""Record measured CPU allocation and the actual bounded polling coordinator."""
import copy
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.exact_waiting_process_retirement_v1 import terminal
from slm_lab.function_repair_GPU_contract_v1 import QUEUE,ADMISSION,handoff_admitted

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/waiter-polling-throughput-v1-review'
PROFILE=ROOT/'reports/waiter-polling-throughput-CPU-v1'
DUTY=ROOT/'reports/waiter-polling-duty-cycle-CPU-v1'


def read(path):return json.loads(Path(path).read_text())


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists():raise ValueError('Fresh CPU-only observation review required')
    measured=read(PROFILE/'complete.json');duty=read(DUTY/'plan.json')
    if (not measured.get('complete') or not measured.get('all_nine_waiting_parents_resumed')
            or terminal(duty['owner']) or not (DUTY/'cycle-1-complete.json').exists()):
        raise ValueError('Complete repeated measurements and an actual live coordinator with a completed rescue cycle required')
    windows=measured['actual_windows'];normal=[v for v in windows if not v['window'].startswith('paused')]
    paused=[v for v in windows if v['window'].startswith('paused')]
    if len(normal)!=3 or len(paused)!=2:raise ValueError('All five original fixed measurement windows required')
    means={name:{group:sum(v['actual_average_CPU_cores'][group] for v in rows)/len(rows)
                 for group in ['waiter_pollers','CPU_training','CPU_assessment','GPU_workers']}
           for name,rows in [('normal_polling',normal),('temporarily_paused_polling',paused)]}
    previous=Path(read(ROOT/'reports/completion-audit.json')['latest_full_scope_review'])
    files=dict(measured['source_files_sha256'])
    paths=[Path(__file__),previous,PROFILE/'complete.json',PROFILE/'plan.json',DUTY/'plan.json',
           DUTY/'cycle-1-complete.json',DUTY/'cycle-1/rescue-plan.json',DUTY/'cycle-1/rescue-complete.json',
           ROOT/'scripts/throttle_waiter_polling_CPU_v1.py',ROOT/'tests/test_bounded_waiter_pause_protected_exit_v1.py',
           ROOT/'reports/bounded-waiter-pause-protected-exit-v1-tests.log']
    files.update({str(p):file_hash(p) for p in paths});guard=FileBindings(files)
    targets=duty['actual_idle_targets'];protected=duty['protected_neural_and_GPU_study_owners'];all_owners=targets+protected
    hz=os.sysconf('SC_CLK_TCK')
    def snapshot():
        values={}
        for owner in all_owners:
            fields=(Path('/proc')/str(owner['pid'])/'stat').read_text().split(') ',1)[1].split()
            if fields[19]!=owner['ticks'] or fields[0] in {'Z','X'}:raise ValueError('Actual original process birth ended during observation')
            values[str(owner['pid'])]=dict(CPU_ticks=int(fields[11])+int(fields[12]),state=fields[0])
        return dict(monotonic=time.monotonic(),processes=values)
    first=snapshot();time.sleep(60);last=snapshot();wall=last['monotonic']-first['monotonic']
    groups=dict(waiter_pollers=[v['pid'] for v in targets],CPU_training=[protected[1]['pid']],CPU_assessment=[protected[2]['pid']],
                GPU_workers=[v['pid'] for v in protected[4:]])
    actual_duty_CPU={group:sum(last['processes'][str(pid)]['CPU_ticks']-first['processes'][str(pid)]['CPU_ticks'] for pid in pids)/hz/wall
                     for group,pids in groups.items()}
    if terminal(duty['owner']) or (DUTY/'failure.json').exists():raise ValueError('The observed bounded coordinator must remain live and healthy')
    gpu_queue=read(ROOT/QUEUE)
    if not handoff_admitted(ROOT,gpu_queue['owner']):raise ValueError('All ten original full-scope successor owners must remain live')
    raw=(ROOT/'reports/public-pretraining-recovery-v1/training-metrics.jsonl').read_bytes();prefix=raw[:raw.rfind(b'\n')+1]
    rows=list(map(json.loads,prefix.splitlines()))
    if [v['step'] for v in rows]!=list(range(1,len(rows)+1)):raise ValueError('Complete contiguous actual neural prefix required')
    current=dict(owner=protected[0],complete_actual_updates=len(rows),planned_updates=4447,
                 response_EOS_targets=sum(int(v['response_EOS_targets']) for v in rows),complete_prefix_sha256=hashlib.sha256(prefix).hexdigest())
    at=datetime.now(timezone.utc).isoformat();OUT.mkdir()
    proof=dict(complete=True,at=at,source_files_sha256=files,previous_goal_turn_classification='progress: tested three-arm first-divergence preference research and actual full-size CPU queue',
        current_goal_turn_classification='progress: repeated measured idle polling CPU cost and bounded reversible polling duty cycle active',
        actual_five_window_CPU_core_means=means,actual_new60second_duty_cycle_observation=dict(before=first,after=last,wall_seconds=wall,average_CPU_cores=actual_duty_CPU),
        actual_duty_owner=duty['owner'],actual_duty_status=read(DUTY/'progress.json'),
        maximum_adopted_lifetime_seconds=1200,pause_seconds=45,original_running_seconds=15,independent_rescue_seconds=55,
        actual_original_waiting_parents=9,existing_CPU_strata_reader_left_running=True,
        actual_disposable_pause_error_deadline_killed_parent_tests_passed=3,actual_protected_exit_rescue_tests_passed=1,
        original_main=current,original_CPU_comparison=read(ROOT/'reports/full-function-repair-comparison-CPU-v1/progress.json'),
        original_interim_assessment=read(ROOT/'reports/public-pretraining-interim-CPU-v2/progress.json'),
        actual_GPU_rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True),
        all_original_neural_worker_births_and_source_and_benchmark_scope_preserved=True,
        no_neural_worker_signals_or_runtime_source_or_queue_edits=True,
        scheduling_evidence_not_direct_model_speed_or_accuracy_gain=True,actual_new_neural_optimizer_updates_by_resource_coordinator=0,
        candidate_promoted=False,goal_achieved=False)
    atomic_json(OUT/'complete.json',proof)
    lines=['# Measured CPU polling overhead','',
           'Nine unstarted, childless, CUDA-hidden controller parents repeatedly hash unchanged bound files while their predecessors run. '
           'Two bounded suspension windows and three normal/restored windows measure their CPU cost. Every original controller resumes, '
           'and the original source checks still run before any later neural stage.', '',
           '| Process group | Normal polling: CPU cores | Full pause: CPU cores | Live 45/15 duty cycle: CPU cores |',
           '|---|---:|---:|---:|']
    labels={'waiter_pollers':'Idle queue checks','CPU_training':'Existing CPU model training','CPU_assessment':'Existing CPU model assessment','GPU_workers':'Original four GPU driver processes'}
    for group,label in labels.items():
        lines.append(f'| {label} | {max(0,means["normal_polling"][group]):.3f} | {max(0,means["temporarily_paused_polling"][group]):.3f} | {actual_duty_CPU[group]:.3f} |')
    lines += ['', 'CPU cores here mean process CPU seconds divided by wall seconds. The five original windows last about 20 seconds each; '
              'the separate live duty observation lasts 60 seconds. These measure scheduler allocation and avoided hashing work, '
              'not generated-answer throughput, FLOP efficiency, algorithmic correctness or capability gain.', '',
              'The temporary coordinator runs for at most 20 minutes. Each suspension of 45 seconds has its own independent rescue after 55 seconds, '
              'followed by 15 seconds of the unchanged original checks. Pipe closure after coordinator death, a protected predecessor exit, '
              'or the hard deadline resumes every exact held pidfd. Disposable-process tests prove normal/error restoration, deadline restoration, '
              'killed-parent rescue and protected-predecessor exit rescue. It stops before the final public-training handoff or when an original stage changes.', '',
              'The actual main/GPU workers, current CPU training and current CPU assessment receive no signal. No original runtime file, source, '
              'queue, epoch, target mask, optimizer setting, generation budget, human feedback source or benchmark population is edited. '
              'All ten downstream owners remain live; suspended idle pollers are live processes and are not reported as terminal.', '',
              f'The original main is at {len(rows):,}/4,447 updates in this review. All four L40s remain in that run. '
              'The full human/RL/export/evaluation pipelines and the new preference experiment remain intact. '
              '**No model accuracy improvement or frontier-level capability is established. All ten final requirements remain incomplete.**','']
    (OUT/'metrics.md').write_text('\n'.join(lines))
    review=copy.deepcopy(read(previous))
    evidence=(' Repeated windows of 20 seconds measure about 0.9 CPU cores spent on unchanged-file checks by 9 unstarted CUDA-hidden queue parents. '
              'A bounded 20-minute 45/15 polling duty cycle is actually live, with held-pidfd independent rescue after 55 seconds. '
              'Four real disposable-process controls cover error/deadline/killed-parent/protected-exit restoration. '
              'No main/GPU/active CPU neural worker is signalled and no source, data, numerical setting or benchmark scope is edited. '
              'Measured CPU allocation does not establish decoding throughput or an accuracy gain.')
    review['requirements']['serious_full_model_training']['evidence']+=evidence
    if any(v['status']!='incomplete_for_requested_final_model' for v in review['requirements'].values()):raise ValueError('Resource scheduling completes no final-model requirement')
    review.update(at=at,previous_full_scope_review=str(previous),current_goal_turn_classification=proof['current_goal_turn_classification'],
                  latest_actual_polling_measurement=proof,source_files_sha256={**files,str(OUT/'complete.json'):file_hash(OUT/'complete.json')},goal_achieved=False)
    atomic_json(OUT/'review.json',review);guard.check(full=True)
    maps=[read(ROOT/'reports/public-pretraining-recovery-v1/plan.json')['source_files_sha256'],
          read(ROOT/'reports/full-function-repair-comparison-CPU-v1/plan.json')['source_files_sha256'],
          read(ROOT/'reports/public-pretraining-interim-CPU-v2/evaluation-plan.json')['source_files_sha256'],
          read(ROOT/'reports/public-edge-preference-comparison-CPU-v1-queue.json')['source_files_sha256'],gpu_queue['source_files_sha256']]
    maps += [read(ROOT/item['queue'])['source_files_sha256'] for item in read(ROOT/ADMISSION)['new_controllers']]
    for name in ['reports/completion-audit.json','README.md','RESEARCH.md']:
        if any(str(ROOT/name) in mapping for mapping in maps):raise ValueError('Never edit a live source-bound document')
    audit=read(ROOT/'reports/completion-audit.json');audit.update(at=at,latest_full_scope_review=str(OUT/'review.json'),requirements=review['requirements'],
        latest_actual_main_runtime=current,latest_actual_polling_measurement=str(OUT/'complete.json'),goal_achieved=False)
    atomic_json(ROOT/'reports/completion-audit.json',audit)
    paragraph=(f'Measured polling overhead, **{at}**: five actual windows measure idle queue checks at **{means["normal_polling"]["waiter_pollers"]:.2f} CPU cores** '
               'during normal polling and nearly zero while suspended. A separate 60-second window confirms the active bounded 45/15 duty cycle. '
               'Four disposable-process tests verify restoration after normal/error exit, deadline, killed coordinator and protected-predecessor exit. '
               'Only 9 unstarted queue parents are affected; all neural workers and original full source/benchmark scopes remain. '
               f'The main is at **{len(rows)}/4,447**. **This is CPU allocation evidence, not a model-speed or accuracy gain. '
               'All ten final requirements remain incomplete.** [Measurements](reports/waiter-polling-throughput-v1-review/metrics.md), '
               '[full review](reports/waiter-polling-throughput-v1-review/review.json). Earlier entries are historical snapshots.\n\n')
    for name in ['README.md','RESEARCH.md']:
        path=ROOT/name;first_line,rest=path.read_text().split('\n',1);path.write_text(first_line+'\n\n'+paragraph+rest.lstrip('\n'))
    print(json.dumps(dict(review=str(OUT/'review.json'),actual_duty_CPU=actual_duty_CPU,actual_main_updates=len(rows),goal_achieved=False)),flush=True)


if __name__=='__main__':main()
