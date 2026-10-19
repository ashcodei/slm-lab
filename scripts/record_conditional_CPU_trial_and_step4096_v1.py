"""Record the real full-size conditional trial and verified4096 checkpoint scope."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/conditional-CPU-trial-and-step4096-v1-review'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists():raise ValueError('Fresh CPU-only full-scope review required')
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    previous=ROOT/'reports/full-QA-queued-and-prefix-signal-v1-review/review.json';old=json.loads(previous.read_text())
    dense_path=ROOT/'reports/phase-joint-step4096-readback-v1/complete.json';dense=json.loads(dense_path.read_text())
    diagnostic_path=ROOT/'reports/prefix-reward-density-full-CPU-v2-readback/complete.json';diagnostic=json.loads(diagnostic_path.read_text())
    mechanics_path=ROOT/'reports/conditional-prefix-CPU-v1/complete.json';mechanics=json.loads(mechanics_path.read_text())
    trial_root=ROOT/'reports/conditional-prefix-full-CPU-v1';trial_path=trial_root/'plan.json';trial=json.loads(trial_path.read_text())
    if (dense['actual_completed_updates']!=4096 or dense['actual_trained_targets']!=141639285
            or not diagnostic['all40_actual_candidate_programs_independently_rechecked_with_same_original_outcomes']
            or not mechanics['next_conditional_update_weights_Adam_Python_NumPy_TorchCPU_RNG_and_actual_trajectory_replay_exact']
            or trial['actual_exported_student_values_each_arm']!=999706442 or trial['planned_updates_each_arm']!=3
            or trial['CPU_mechanics_horizon'] is not None):raise ValueError('Actual completed full diagnostic, dense state proof and full-size unshortened trial required')
    for proof in [dense,diagnostic,mechanics]:
        for name,digest in proof['source_files_sha256'].items():
            if file_hash(Path(name))!=digest:raise ValueError('Immutable new evidence input changed')
    def identity(owner,hidden=False):
        path=Path('/proc')/str(owner['pid']);fields=(path/'stat').read_text().split(') ',1)[1].split()
        if fields[19]!=owner['ticks'] or fields[0] in ['Z','X']:raise ValueError('Exact live PID/start ticks required')
        children=(path/'task'/str(owner['pid'])/'children').read_text().strip();env=(path/'environ').read_bytes().decode().split('\0')
        visible=next((s.split('=',1)[1] for s in env if s.startswith('CUDA_VISIBLE_DEVICES=')),None)
        if hidden and (children or visible!=''):raise ValueError('Waiting controller must be CPU-hidden and childless')
        return dict(pid=owner['pid'],ticks=fields[19],state=fields[0],children=children,CUDA_VISIBLE_DEVICES=visible)
    phase_path=ROOT/'reports/phase-joint-v3/plan.json';phase=json.loads(phase_path.read_text())
    queues={name:json.loads((ROOT/'reports'/name).read_text()) for name in [
        'phase-human-preference-v4-queue.json','broad-anchored-code-v3-queue.json','full-held-QA-pipeline-v1-queue.json']}
    workers=[identity(p) for p in json.loads((ROOT/'reports/phase-joint-v3/children.json').read_text()) if p['subject'].startswith('train-rank-')]
    if len(workers)!=4:raise ValueError('All four exact current broad-training workers required')
    journal=[]
    for line in (ROOT/'reports/phase-joint-v3/training-metrics.jsonl').read_bytes().splitlines():
        try:journal.append(json.loads(line))
        except json.JSONDecodeError:break
    if [r['step'] for r in journal]!=list(range(1,len(journal)+1)):raise ValueError('Every original committed main update required')
    OUT.mkdir();at=datetime.now(timezone.utc).isoformat();atomic_json(OUT/'actual-main-journal-prefix.json',journal)
    updates={}
    for arm in ['anchor_CE','anchor_execution']:
        path=trial_root/(arm+'-progress.json');updates[arm]=json.loads(path.read_text())['actual_logical_updates'] if path.exists() else 0
    baseline=trial_root/'initial-actual-evaluation-prefix.json';baseline_rows=json.loads(baseline.read_text()) if baseline.exists() else []
    atomic_json(OUT/'CPU-trial-baseline-observation.json',baseline_rows)
    runtime=dict(at=at,actual_main_owner=identity(phase['owner']),actual_four_main_workers=workers,
        actual_main_updates=len(journal),planned_main_updates=35284,actual_main_targets=int(sum(r['response_EOS_targets'] for r in journal)),
        planned_main_targets=467581587,waiting_owners={k:identity(v['owner'],True) for k,v in queues.items()},
        conditional_full_size_CPU_trial_owner=identity(trial['owner']),conditional_full_size_CPU_trial_updates=updates,
        current_initial_CPU_trial_neural_evaluations=len(baseline_rows),
        GPU_observation=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used','--format=csv,noheader'],text=True),
        full_new_human_or_code_GPU_production_updates=0,new_full_QA_neural_assessments=0,goal_achieved=False)
    atomic_json(OUT/'runtime.json',runtime)
    requirements=old['requirements']
    requirements['own_architecture']['evidence']+=(' The qualified-prefix kernel now has an actual CPU update/recovery implementation. '
        'It differentiates only genuine sampled suffix actions, executes their reconstructed program with the explicit public prefix, '
        'and always supervises all original CE/EOS targets. Only16source problems and2/3-quarter views have qualified controls; '
        'this does not admit the whole3,721-source curriculum. The native CPU-trial export explicitly identifies the partial own '
        'step2048 base and cannot masquerade as complete human alignment or a complete source epoch.')
    requirements['under_1b_parameters']['evidence']+=(' The real CPU trial loaded two999,706,442-value policy arms sharing one984,399,690-value '
        'own base, with one extra frozen own starting-policy private snapshot. All-arm training resident unique values are explicitly '
        '1,030,319,946; each exported inference student is999,706,442. The control branch/reference are outside each student export. '
        'The complete full-size initial computations of both arms are bit exact, and neutral-actor/base parity is measured before training.')
    requirements['serious_full_model_training']['evidence']=(f'The original full-weight984,399,690-value broad model is active at '
        f'{len(journal):,}/35,284 actual updates and {runtime["actual_main_targets"]:,}/467,581,587 targets. The real4096 snapshot '
        'independently reconciles every model tensor, all317 language Adam states/clocks, original source positions and141,639,285 '
        'targets, finite moments and unchanged inactive vision/word/other heads. Its mutable large resume was fingerprinted without '
        'duplication; permanent native values plus compact actual hashes/private RNG/journal evidence remain retained. The new matched '
        f'full-size CPU trial is live with actual observed updates {updates}; it schedules3logical updates per arm,4fresh suffix samples '
        'per group, complete native action budgets, every original reference target and an extra exact replay of update2. It repeats '
        'one train problem, not a source epoch. Initial/matched endpoint source-only evaluations and student-only cold text/image '
        'reload are concrete in the driver, but endpoints/quality are still pending. Original complete human/code GPU epochs remain waiting.')
    requirements['gpu_runtime']['evidence']=('All four exact original broad-training GPU ranks are live; the human/code/full-QA queues '
        'remain CPU-hidden and childless. The new full-size conditional comparison is CPU only and waited for the exact40-episode '
        'diagnostic process to become terminal. It initializes no CUDA device. Its kernel explicitly rejects GPU execution. '
        'The small CPU recovery test proves the next update, full original166-target CE, all private RNG and deliberate probability '
        'fault rejection; this cannot admit full-size GPU conditional training. The original four-L40 full-source pipeline is unchanged.')
    requirements['any_code_correctness_and_efficiency']['evidence']=('NOT ACHIEVED. The same128 source-held packed windows/524,160 '
        'targets now give producer NLL0.8699451996362768(initial),0.8051151143154129(step2048),0.7935042203171179(step4096): '
        '8.786872937642376% lower than initial. Actual checkpoint weights and held positions/context/token bytes were independently '
        'read back; the reader did not repeat a neural loss forward. Complete old unassisted native scores remain0/24completion '
        'and1/24instruction in both old CE/RL arms. The completed40-episode real984,399,690-value CPU diagnostic has0/20positive '
        'unassisted rewards, versus one partial-positive hinted greedy attempt and one full-finite-case-positive hinted stochastic '
        'attempt. All40 actual candidate programs were independently rechecked under the original compile/scope/runtime protocol, '
        'with every aggregate and within-source reward group rederived. The hinted stochastic source0001 group is[1,0,0,0], '
        'while every unassisted stochastic group is all zero. This establishes available conditional training signal on one source, '
        'not a statistically established curriculum benefit, unassisted held gain, algorithmic efficiency or high-end parity. '
        'The live full-size CPU training comparison now tests whether an update helps; no new endpoint result is ready.')
    if len(requirements)!=10 or any(v['status']!='incomplete_for_requested_final_model' for v in requirements.values()):
        raise ValueError('All ten final requirements must remain incomplete')
    files=[Path(__file__),previous,dense_path,diagnostic_path,mechanics_path,trial_path,phase_path,
        OUT/'runtime.json',OUT/'actual-main-journal-prefix.json',OUT/'CPU-trial-baseline-observation.json',
        ROOT/'reports/conditional-prefix-full-CPU-v1-queue.json',ROOT/'reports/conditional-trial-policy-CPU-v1-final-tests.log',
        ROOT/'reports/conditional-prefix-full-CPU-v1-driver-tests.log',ROOT/'reports/admitted-public-prefix-episode-v1-final-tests.log']
    report=dict(at=at,requirements=requirements,actual_runtime=runtime,actual4096_checkpoint_and_held_loss_readback=dense,
        actual_complete40_episode_neural_execution_readback=diagnostic,actual_small_CPU_conditional_update_recovery=mechanics,
        actual_full_size_CPU_trial_plan=trial,previous_goal_turn_classification='progress',current_goal_turn_classification='progress',
        new_unassisted_held_coding_or_all_domain_gain_measured=False,all_ten_final_requirements_incomplete=True,no_remaining_scope_removed=True,
        source_files_sha256={str(p):file_hash(p) for p in files},candidate_promoted=False,goal_achieved=False)
    atomic_json(OUT/'review.json',report)
    master_path=ROOT/'reports/completion-audit.json';docs=[ROOT/'README.md',ROOT/'RESEARCH.md',master_path]
    protected=set(trial['source_files_sha256'])|set(phase['source_files_sha256'])
    for queue in queues.values():protected.update(queue['source_files_sha256'])
    if any(str(p) in protected for p in docs):
        print(json.dumps(dict(review=str(OUT/'review.json'),root_documents_source_bound_not_modified=True)));return
    master=json.loads(master_path.read_text())
    with (ROOT/'reports/completion-audit-before-conditional-CPU-trial4096-v1.json').open('xb') as f:f.write(master_path.read_bytes())
    master.update(updated_at=at,requirements={k:v['evidence'] for k,v in requirements.items()},
        latest_full_scope_review=str((OUT/'review.json').relative_to(ROOT)),live_joint_training=runtime,
        previous_goal_turn_classification='progress',current_goal_turn_classification='progress',
        latest_actual_checkpoint_held_loss_readback=str(dense_path.relative_to(ROOT)),
        actual_partial_program_neural_probe=dict(status='complete',neural_episodes=40,full_readback=str(diagnostic_path.relative_to(ROOT)),model_updates=0),
        current_conditional_prefix_CPU_training=dict(plan=str(trial_path.relative_to(ROOT)),owner=runtime['conditional_full_size_CPU_trial_owner'],
            planned_logical_updates_each_arm=3,actual_updates_at_review=updates,exported_student_values=999706442,
            full_new_GPU_admission=False,not_complete_curriculum_or_human_alignment=True),
        all_ten_final_requirements_incomplete=True,candidate_promoted=False,goal_achieved=False)
    atomic_json(master_path,master)
    paragraph=(f'Actual conditional CPU training trial and step4,096 readback, **{at}**: the retained **4,096-update** own native model '
        'and observed full training state reconcile: **317 active language Adam clocks**, **141,639,285 targets**, every original '
        'source cursor, finite moments, and unchanged inactive vision/word/other heads. On the **same128 held packed windows/524,160 '
        'targets**, producer NLL is **0.793504**, down **8.79%** from0.869945 initially (step2,048:0.805115). Held input bytes and '
        'protocol were independently checked; no new neural loss forward was performed by the reader. This is next-token prediction '
        'improvement, not an unassisted coding or all-domain capability result.\n\n'
        'The actual full-size **40-episode CPU sampling diagnostic is complete**. Independent replay of every candidate under the '
        'original runtime reproduces **0/20 positive unassisted rewards**, one hinted greedy partial reward and **one hinted stochastic '
        'full finite-case pass** with actual generated EOS. Its source0001 hinted stochastic group is **[1,0,0,0]**; all unassisted '
        'groups are zero. This is a conditional training signal on public train data, not a held/general or statistical efficacy claim.\n\n'
        'A separate real full-size CPU comparison is now live: two **999,706,442-value students**, sharing one own step2,048 base, '
        'compare always-on complete reference CE against the same CE plus fresh conditional execution policy gradients. One independently '
        'qualified train problem receives **three logical updates per arm**, with **four fresh suffix samples per group**, complete '
        'native action budgets and an exact update2 recovery test. Extra control/reference weights are training-only; all-arm resident '
        'unique values are explicitly1,030,319,946. The **1,110,722-value small CPU** mechanism/recovery/fault controls and native-policy '
        'IO checks pass. **No full-size conditional GPU admission exists.** Endpoint evaluations, full-size optimizer recovery and any '
        f'quality effect are pending. Actual trial updates in this snapshot: **{updates}**.\n\n'
        f'The original broad pipeline remains active at **{len(journal):,}/35,284 updates**, with human/code/full-QA successors waiting. '
        'The completed code suite remains0/24completion and1/24instruction in both old arms. **All ten final requirements remain incomplete.** '
        'Evidence: [review](reports/conditional-CPU-trial-and-step4096-v1-review/review.json), '
        '[step4096](reports/phase-joint-step4096-readback-v1/complete.json), '
        '[complete40-episode readback](reports/prefix-reward-density-full-CPU-v2-readback/complete.json), '
        '[full-size CPU trial plan](reports/conditional-prefix-full-CPU-v1/plan.json).\n\n')
    for path in docs[:2]:
        first,tail=path.read_text().split('\n',1);path.write_text(first+'\n\n'+paragraph+tail.lstrip('\n'))
    print(json.dumps(dict(review=str(OUT/'review.json'),actual_main_updates=len(journal),
        actual_conditional_CPU_updates=updates,completed_neural_diagnostic_episodes=40,held_NLL_reduction=dense['relative_held_NLL_reduction'],
        all_ten_final_requirements_incomplete=True)),flush=True)


if __name__=='__main__':main()
