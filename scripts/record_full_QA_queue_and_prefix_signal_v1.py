"""Record sealed full QA, the actual conditional signal and all final requirements."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/full-QA-queued-and-prefix-signal-v1-review'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists():raise ValueError('Fresh CPU-only complete-scope review required')
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    previous=ROOT/'reports/partial-program-curriculum-v1-review/review.json';old=json.loads(previous.read_text())
    audit_path=ROOT/'reports/full-held-question-audit-v2/complete.json';audit=json.loads(audit_path.read_text())
    prep_path=ROOT/'reports/full-held-QA-evaluation-v1-preparation/complete.json';prep=json.loads(prep_path.read_text())
    queue_path=ROOT/'reports/full-held-QA-pipeline-v1-queue.json';queue=json.loads(queue_path.read_text())
    signal_path=ROOT/'reports/prefix-reward-density-first16-CPU-v2-readback/complete.json';signal=json.loads(signal_path.read_text())
    probe_plan_path=ROOT/'reports/prefix-reward-density-full-CPU-v1/plan.json';probe=json.loads(probe_plan_path.read_text())
    if (not audit['all_six_declared_scopes_completed'] or prep['fresh_MMLU_not_previously_sampled_no_exact_match']!=11397
            or prep['exact_question_matches_in_any_inspected_input_pool']!={'MMLU-Pro':76,'GSM8K':0,'NQ-open':18}
            or queue['full_QA_questions_each_endpoint']!=16961 or len(queue['endpoints'])!=6
            or not signal['all16_actual_candidate_programs_independently_rechecked_with_same_original_outcomes']
            or signal['summary']['3:full_vocabulary_temperature1']['full_finite_case_credit_with_actual_EOS']!=1):
        raise ValueError('Actual sealed source population, fixed evaluation queue and independent real student execution evidence required')
    for proof in [prep,signal]:
        for name,digest in proof['source_files_sha256'].items():
            if file_hash(Path(name))!=digest:raise ValueError('Permanent new proof source changed')
    def identity(owner,childless=False):
        path=Path('/proc')/str(owner['pid']);f=(path/'stat').read_text().split(') ',1)[1].split()
        if f[19]!=owner['ticks'] or f[0] in ['Z','X']:raise ValueError('Exact live PID/start ticks required')
        children=(path/'task'/str(owner['pid'])/'children').read_text().strip();env=(path/'environ').read_bytes().decode().split('\0')
        visible=next((v.split('=',1)[1] for v in env if v.startswith('CUDA_VISIBLE_DEVICES=')),None)
        if childless and (children or visible!=''):raise ValueError('Queued controller must remain CPU-hidden and childless')
        return dict(pid=owner['pid'],ticks=f[19],state=f[0],children=children,CUDA_VISIBLE_DEVICES=visible)
    phase_path=ROOT/'reports/phase-joint-v3/plan.json';phase=json.loads(phase_path.read_text())
    human_path=ROOT/'reports/phase-human-preference-v4-queue.json';human=json.loads(human_path.read_text())
    code_path=ROOT/'reports/broad-anchored-code-v3-queue.json';code=json.loads(code_path.read_text())
    workers=[identity(p) for p in json.loads((ROOT/'reports/phase-joint-v3/children.json').read_text()) if p['subject'].startswith('train-rank-')]
    if len(workers)!=4:raise ValueError('All four original broad-training workers required')
    journal=[]
    for line in (ROOT/'reports/phase-joint-v3/training-metrics.jsonl').read_bytes().splitlines():
        try:journal.append(json.loads(line))
        except json.JSONDecodeError:break
    if [r['step'] for r in journal]!=list(range(1,len(journal)+1)):raise ValueError('Every actual committed main update required')
    OUT.mkdir();at=datetime.now(timezone.utc).isoformat()
    atomic_json(OUT/'actual-main-journal-prefix.json',journal)
    runtime=dict(at=at,actual_main_owner=identity(phase['owner']),actual_four_main_workers=workers,
        actual_main_updates=len(journal),planned_main_updates=35284,actual_main_targets=int(sum(r['response_EOS_targets'] for r in journal)),
        planned_main_targets=467581587,human_waiting_owner=identity(human['owner'],True),new_code_waiting_owner=identity(code['owner'],True),
        full_QA_waiting_owner=identity(queue['owner'],True),actual_CPU_probe_owner=identity(probe['owner']),
        GPU_observation=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used','--format=csv,noheader'],text=True),
        full_new_human_or_code_production_updates=0,new_full_QA_neural_assessments=0,goal_achieved=False)
    atomic_json(OUT/'runtime.json',runtime)
    progress=json.loads((probe_plan_path.parent/'progress.json').read_text());atomic_json(OUT/'CPU-probe-progress-observation.json',progress)
    requirements=old['requirements']
    requirements['public_data']['evidence']+=(' The explicit six-scope audit is now complete across3,250,975,085 native input positions '
        '(including repeated/current/future presentations and older possible source pools, not new unique training tokens). '
        'It flags76MMLU/18NQ question identities and no GSM8K question stem in the inspected pools; whole historical consumption, '
        'semantic independence and unannotated image-pixel text remain uncertified. All original source bytes and raw shard boundaries were preserved.')
    requirements['serious_full_model_training']['evidence']=(f'Actual broad full-weight training remains live at {len(journal):,}/35,284 '
        f'updates and {runtime["actual_main_targets"]:,}/467,581,587 targets on all four L40s. The earlier actual2048 native/model/Adam '
        'readback and held NLL improvement remain valid. Human-v4 and anchored-code-v3 are unchanged and waiting. '
        f'A separate984,399,690-value CPU/FP32 diagnostic has completed {progress["completed_actual_neural_episodes"]}/40 actual '
        'conditional/unassisted neural episodes in the observed runtime; there are no optimizer updates. The first16 deliveries '
        'and original runtime outcomes were independently reconstructed/rechecked. A prefix-capable policy-gradient kernel, '
        'actual conditional training and fresh full-size GPU admission are still required before this strategy can train.')
    requirements['gpu_runtime']['evidence']=('All four exact original broad-training GPU workers and owner are live. Human/code/full-QA '
        'waiting owners are CPU-hidden and childless. Actual full-model prefix sampling uses CPU only and does not preempt GPUs. '
        'The full-QA controller waits for complete code-v3 epochs, assessments, all registered workers terminal and restoration, '
        'then checks all four idle physical L40s under the existing global lease before holding/restoring the exact original photo waiter. '
        'No new full-QA GPU assessment or conditional-RL GPU admission has happened.')
    requirements['broad_independent_benchmarks']['evidence']=('All16,961 original native QA prompts (MMLU-Pro12,032, GSM8K1,319, '
        'NQ-open3,610) now have complete source-audit flags, unchanged prompt-ID/length/budget checks, source reference-format '
        'positive and blank-negative scoring controls, and exact four-GPU shard coverage. There are11,397 MMLU questions not '
        'previously sampled locally with no exact stem match in the inspected input pools. Native generation retains32/512/32 tokens, '
        'full prompt/output IDs, unchanged gold/options/aliases and separate generated versus direct/spaced-letter likelihood scores. '
        'Real small-network tests show gold indices do not influence generation; changed text/token likelihoods, changed preflight, '
        'missing/duplicate/checkpoint-mismatched shards and early source-audit use are rejected. The concrete live full-QA queue '
        'predeclares six endpoints: own broad initial/trained, complete human chosen-CE/DPO and complete code CE/execution arms. '
        'Each future assessment covers every original question, separates exposed/fresh/domain strata, and reports paired gains/losses '
        'without automatic selection. No new held QA neural score exists. Existing code24-task frames,64originalAPPS publisher-test '
        'problems,4,642held public human rankings and broad image checks remain required in the original pipeline. Full cross-language '
        'coding/efficiency, stronger multimodal and final independent testing remain incomplete. These protocols are not official '
        'leaderboard scores or semantic/full-ancestry independence certificates.')
    requirements['any_code_correctness_and_efficiency']['evidence']+=(' The actual own step2048 model now supplies genuine sampled '
        'conditional outputs. In the complete first16 training-source episode block: unassisted greedy0/4positive and stochastic0/4positive; '
        'hinted greedy1/4partial-positive (zero full) and hinted stochastic1/4full finite-case credit with actual sampled EOS. '
        'Every program was independently rechecked under the original compile/scope/execution protocol, including correct rejection '
        'before cases when qualification fails. Public reference prefixes are explicit and the generated suffix is real. This is '
        'conditional training-source reachability, not unassisted held coding, algorithmic efficiency or a proven curriculum effect. '
        'Only one stochastic sample per source/condition exists in that fixed block, so within-source4-sample reward variance is pending. '
        'The full40-episode diagnostic is still running. A v1 ledger wording ambiguity about cases after compile failures is preserved '
        'and corrected by the separately rechecked v2 evidence. Current complete native code scores remain0/24completion and1/24instruction '
        'in both older CE/RL arms; no new unassisted coding improvement is claimed.')
    if len(requirements)!=10 or any(v['status']!='incomplete_for_requested_final_model' for v in requirements.values()):
        raise ValueError('All ten incomplete final requirements must remain explicit')
    files=[Path(__file__),previous,audit_path,prep_path,queue_path,signal_path,probe_plan_path,phase_path,human_path,code_path,
        OUT/'runtime.json',OUT/'actual-main-journal-prefix.json',OUT/'CPU-probe-progress-observation.json',
        ROOT/'reports/full-held-QA-native-v1-final-tests.log',ROOT/'reports/full-held-QA-pipeline-v1-tests.log',
        ROOT/'reports/native-prefix-rollout-CPU-v1-corrected-tests.log',ROOT/'reports/prefix-reward-density-first16-CPU-v1-wording-correction.json']
    report=dict(at=at,requirements=requirements,actual_runtime=runtime,completed_full_source_audit=audit,
        complete_full_QA_input_preparation=prep,first16_actual_CPU_student_program_readback=signal,
        full_QA_queue=queue,previous_goal_turn_classification='progress',current_goal_turn_classification='progress',
        actual_conditional_student_generations_observed=progress['completed_actual_neural_episodes'],
        actual_conditional_training_updates=0,new_held_QA_or_unassisted_code_gain_measured=False,
        all_ten_final_requirements_incomplete=True,no_remaining_scope_removed=True,
        source_files_sha256={str(p):file_hash(p) for p in files},candidate_promoted=False,goal_achieved=False)
    atomic_json(OUT/'review.json',report)
    master_path=ROOT/'reports/completion-audit.json';docs=[ROOT/'README.md',ROOT/'RESEARCH.md',master_path]
    protected=set().union(*(set(p['source_files_sha256']) for p in [phase,human,code,queue,probe]))
    if any(str(p) in protected for p in docs):
        print(json.dumps(dict(review=str(OUT/'review.json'),root_documents_source_bound_not_modified=True)));return
    master=json.loads(master_path.read_text())
    with (ROOT/'reports/completion-audit-before-full-QA-queue-prefix-signal-v1.json').open('xb') as f:f.write(master_path.read_bytes())
    master.update(updated_at=at,requirements={k:v['evidence'] for k,v in requirements.items()},
        latest_full_scope_review=str((OUT/'review.json').relative_to(ROOT)),live_joint_training=runtime,
        previous_goal_turn_classification='progress',current_goal_turn_classification='progress',
        latest_full_held_source_audit=dict(status_at_review='complete',complete=str(audit_path.relative_to(ROOT)),fresh_MMLU_questions=11397),
        queued_full_QA_assessment=dict(queue=str(queue_path.relative_to(ROOT)),owner=runtime['full_QA_waiting_owner'],
            questions_each_endpoint=16961,predeclared_endpoints=list(queue['endpoints']),neural_assessments=0),
        actual_partial_program_neural_probe=dict(owner=runtime['actual_CPU_probe_owner'],completed_episodes_at_review=progress['completed_actual_neural_episodes'],
            planned_episodes=40,first16_readback=str(signal_path.relative_to(ROOT)),model_updates=0,unassisted_held_capability_gain=False),
        all_ten_final_requirements_incomplete=True,candidate_promoted=False,goal_achieved=False)
    atomic_json(master_path,master)
    paragraph=(f'Actual prefix-sampling signal and full QA queue, **{at}**: the complete source audit inspected **3,250,975,085 input positions** '
        '(including repetitions and older possible exposure, not new training tokens), flagging **76MMLU-Pro/18NQ** stems and zeroGSM8K. '
        '**11,397 MMLU questions** remain new to local evaluation with no exact stem match in these pools. This does not certify semantic '
        'independence, complete historical consumption or unannotated image text.\n\n'
        'All **16,961 original QA inputs** passed complete native input/budget and reference-format/blank scoring controls. A live '
        '**CPU-hidden, childless queue** now waits behind the original complete code experiment, then assesses all six fixed own-model '
        'endpoints on all four L40s. It retains raw token deliveries, flags exposed/fresh cases and reports subjects and paired gains/losses '
        'without choosing a winner. **New held QA neural assessments are zero.**\n\n'
        'The actual **984,399,690-value step2,048 model** is separately sampling complete512-token CPU/FP32 trajectories on four public '
        'training tasks, comparing unassisted versus explicit public-reference prefixes. The independently rechecked **first16** episode '
        'block yields **0/8 positive unassisted rewards**, **one hinted greedy partial reward**, and **one hinted stochastic full finite-case '
        'pass with actual generated EOS**. All real suffix tokens and original runtime outcomes are retained; compile/scope failures '
        'qualify before cases. This is conditional training-source reachability, **not unassisted coding improvement or a proven RL effect**. '
        'The full40 episodes and within-source4-sample reward variation remain pending. **Conditional-training optimizer updates are zero.**\n\n'
        f'Broad training remains active at **{len(journal):,}/35,284 updates** in this snapshot. The earlier **7.45% held-code NLL reduction** '
        'is retained; full human/code epochs, useful own-output human alignment, stronger vision and final independent capability remain '
        'unfinished. **All ten final requirements remain incomplete.** Evidence: '
        '[review](reports/full-QA-queued-and-prefix-signal-v1-review/review.json), '
        '[first16 actual-program readback](reports/prefix-reward-density-first16-CPU-v2-readback/complete.json), '
        '[full QA preparation](reports/full-held-QA-evaluation-v1-preparation/complete.json), '
        '[evaluation queue](reports/full-held-QA-pipeline-v1-queue.json).\n\n')
    for path in docs[:2]:
        first,tail=path.read_text().split('\n',1);path.write_text(first+'\n\n'+paragraph+tail.lstrip('\n'))
    print(json.dumps(dict(review=str(OUT/'review.json'),actual_main_updates=len(journal),
        actual_CPU_student_episodes=progress['completed_actual_neural_episodes'],full_QA_queue_live=True,
        fresh_MMLU_questions=11397,conditional_optimizer_updates=0,all_ten_final_requirements_incomplete=True)),flush=True)


if __name__=='__main__':main()
