"""Record actual held code loss, full QA reserve and the complete active scope."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/held-code-loss-and-full-QA-v1-review'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists():raise ValueError('Fresh CPU-only complete-scope review required')
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    previous_path=ROOT/'reports/broad-anchored-code-queued-v3-review/review.json'
    old=json.loads(previous_path.read_text());requirements=old['requirements']
    proof_path=ROOT/'reports/phase-joint-step2048-readback-v1/complete.json';proof=json.loads(proof_path.read_text())
    reserve_path=ROOT/'data/full-held-question-pool-v2/manifest.json';reserve=json.loads(reserve_path.read_text())
    preflight_path=ROOT/'reports/full-held-native-context-preflight-v1/complete.json';preflight=json.loads(preflight_path.read_text())
    audit_root=ROOT/'reports/full-held-question-audit-v2';audit_plan_path=audit_root/'plan.json'
    audit_plan=json.loads(audit_plan_path.read_text())
    if (not proof['complete'] or not proof['native_and_resume_all_model_tensors_bit_exact']
            or proof['actual_trained_targets']!=71028428 or proof['active_language_Adam_states']!=317
            or not preflight['all_16961_original_held_QA_inputs_fit_complete_unchanged_budgets']
            or reserve['new_to_local_evaluation_MMLU_questions']!=11472):
        raise ValueError('Actual retained checkpoint proof and every original held prompt preflight required')
    # Check every permanent proof binding. Its explicitly mutable raw resume
    # observation is deliberately absent from that permanent binding map.
    for name,digest in proof['source_files_sha256'].items():
        if file_hash(Path(name))!=digest:raise ValueError('Permanent step2048 evidence source changed')
    def identity(owner,hidden_childless=False):
        path=Path('/proc')/str(owner['pid']);fields=(path/'stat').read_text().split(') ',1)[1].split()
        if fields[19]!=owner['ticks'] or fields[0] in ['Z','X']:raise ValueError('Exact live PID/start ticks required')
        children=(path/'task'/str(owner['pid'])/'children').read_text().strip()
        env=(path/'environ').read_bytes().decode().split('\0')
        visible=next((s.split('=',1)[1] for s in env if s.startswith('CUDA_VISIBLE_DEVICES=')),None)
        if hidden_childless and (children or visible!=''):raise ValueError('Exact CPU-hidden childless queued/audit owner required')
        return dict(pid=owner['pid'],ticks=fields[19],state=fields[0],children=children,
            CUDA_VISIBLE_DEVICES=visible,argv=(path/'cmdline').read_bytes().replace(b'\0',b' ').decode())
    phase_path=ROOT/'reports/phase-joint-v3/plan.json';phase=json.loads(phase_path.read_text())
    human_path=ROOT/'reports/phase-human-preference-v4-queue.json';human=json.loads(human_path.read_text())
    code_path=ROOT/'reports/broad-anchored-code-v3-queue.json';code=json.loads(code_path.read_text())
    main_owner=identity(phase['owner']);human_owner=identity(human['owner'],True);code_owner=identity(code['owner'],True)
    workers=[identity(r) for r in json.loads((ROOT/'reports/phase-joint-v3/children.json').read_text()) if r['subject'].startswith('train-rank-')]
    if len(workers)!=4:raise ValueError('All four original broad-training ranks must still be live')
    journal=[]
    for line in (ROOT/'reports/phase-joint-v3/training-metrics.jsonl').read_bytes().splitlines():
        try:journal.append(json.loads(line))
        except json.JSONDecodeError:break
    if [r['step'] for r in journal]!=list(range(1,len(journal)+1)):raise ValueError('Contiguous actual committed training prefix required')
    OUT.mkdir();at=datetime.now(timezone.utc).isoformat()
    atomic_json(OUT/'actual-main-journal-prefix.json',journal)
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used','--format=csv,noheader'],text=True)
    targets=int(sum(r['response_EOS_targets'] for r in journal))
    runtime=dict(at=at,actual_main_owner=main_owner,actual_four_main_workers=workers,actual_main_updates=len(journal),
        planned_main_updates=phase['planned_steps'],actual_main_targets=targets,planned_main_targets=467581587,
        GPU_observation=gpu,human_waiting_owner=human_owner,new_code_waiting_owner=code_owner,
        full_new_human_or_code_production_updates=0,new_full_profile_code_GPU_admission_pending=True,goal_achieved=False)
    atomic_json(OUT/'runtime.json',runtime)
    if (audit_root/'failure.json').exists():raise ValueError('New held source audit failed; resolve before recording it as live')
    if (audit_root/'complete.json').exists():
        audit_snapshot=json.loads((audit_root/'complete.json').read_text());audit_status='complete'
        audit_owner=None
    else:
        audit_owner=identity(audit_plan['owner'],True)
        audit_snapshot=json.loads((audit_root/'progress.json').read_text());audit_status='running'
    atomic_json(OUT/'audit-runtime-observation.json',dict(status=audit_status,owner=audit_owner,observed_progress=audit_snapshot))
    requirements['serious_full_model_training']['evidence']=(f'Full-weight own984,399,690-value training is active at {len(journal):,}/35,284 updates '
        f'and {targets:,}/467,581,587 targets. The actual2048 checkpoint independently reconciles all317 language Adam clocks, '
        'all model values against the native export, complete source/target cursors, all finite moments and nonnegative second moments. '
        'All317 active language tensors changed; all inactive vision/word and other heads remain exact. All four private CPU RNGs are '
        'validated and actual CUDA bytes retained; this CPU reader does not claim a new CUDA replay. The large mutable resume was '
        'fingerprinted, not duplicated or retained; immutable native values, compact hashes, actual RNG states and the original journal are retained. '
        'Full epochs and final native capability measurements are pending; human-v4 and broad-code-v3 remain queued, with zero new production updates.')
    requirements['gpu_runtime']['evidence']=('All four exact current L40 training ranks and the main owner are live. Human/code queues '
        'remain GPU-hidden and childless. The held source audit is CPU-only and does not construct a neural model. '
        'The original full-size capacity/recovery gates passed; fresh full-size new code admission remains pending.')
    requirements['broad_independent_benchmarks']['evidence']+=(' A complete pinned12,032-question MMLU-Pro original test reserve '
        'now contains11,472 not previously sampled locally. All original16,961 QA prompts (MMLU12,032/GSM8K1,319/NQ3,610) '
        'fit the actual own4096 context with unchanged32/512/32 generation budgets and zero clipping/skips. An exact question-substring '
        'audit explicitly covers all577,662,698 current native input positions,31,147,346 future human response input positions, '
        '2,287,522 future code inputs, and2,639,877,519 positions in conservative older possible source pools. Counts include repetitions '
        'and possible historical exposure, not new unique training tokens or proof of exact ancestry consumption. No new held neural '
        'generation has occurred. A completed normalized substring audit cannot certify semantic independence or unannotated image text. '
        f'The v2 audit is {audit_status}; the v1 NQ metadata schema error is preserved with zero model updates and corrected by '
        'a separately tested source-specific extraction. Final independent broad capability remains unproven.')
    requirements['any_code_correctness_and_efficiency']['evidence']+=(' On exactly the same128 source-held packed windows '
        '(524,160 causal targets), producer-measured NLL falls0.8699451996362768to0.8051151143154129 at2048 updates '
        '(7.452203351196063% reduction; perplexity2.3867800535to2.2369539894). Input bytes, positions, budget and checkpoint '
        'weights were independently read back. This is improved next-token prediction, not a new code-correctness/efficiency score; '
        'the reader did not independently repeat the neural loss forward, and individual window losses were not retained for paired uncertainty. '
        'Latest completed native code scores remain0/24 completion and1/24 instruction for both old CE/RL arms; no RL coding advantage is established.')
    if len(requirements)!=10 or any(r['status']!='incomplete_for_requested_final_model' for r in requirements.values()):
        raise ValueError('All ten unachieved final requirements must remain explicit')
    files=[Path(__file__),previous_path,proof_path,reserve_path,preflight_path,audit_plan_path,phase_path,human_path,code_path,
        OUT/'actual-main-journal-prefix.json',OUT/'runtime.json',OUT/'audit-runtime-observation.json',
        ROOT/'reports/full-held-question-audit-v1/failure.json',ROOT/'reports/held-question-registry-v2-tests.log',
        ROOT/'reports/native-record-stream-v1-tests.log',ROOT/'reports/native-training-text-projection-v1-tests.log',
        ROOT/'reports/held-question-scanner-v1-tests.log']
    report=dict(at=at,previous_goal_turn_classification='progress',current_goal_turn_classification='progress',requirements=requirements,
        actual_runtime=runtime,held_code_loss_readback=proof,original_full_QA_input_preflight=preflight,
        source_overlap_audit_status=audit_status,source_overlap_audit_owner=audit_owner,
        source_overlap_audit_plan=str(audit_plan_path.relative_to(ROOT)),
        source_overlap_audit_completed_at_review=audit_status=='complete',
        new_native_code_correctness_or_all_domain_gain_measured=False,new_held_next_token_loss_gain_measured=True,
        original_completed_code_scores_unchanged=True,all_ten_final_requirements_incomplete=True,
        no_remaining_scope_removed=True,model_generations_by_this_review=0,model_updates_by_this_review=0,
        source_files_sha256={str(p):file_hash(p) for p in files},candidate_promoted=False,goal_achieved=False)
    atomic_json(OUT/'review.json',report)
    audit_path=ROOT/'reports/completion-audit.json';docs=[ROOT/'README.md',ROOT/'RESEARCH.md',audit_path]
    protected=set().union(*(set(p['source_files_sha256']) for p in [phase,human,code,audit_plan]))
    if any(str(p) in protected for p in docs):
        print(json.dumps(dict(review=str(OUT/'review.json'),root_documents_source_bound_not_modified=True)));return
    audit=json.loads(audit_path.read_text())
    with (ROOT/'reports/completion-audit-before-held-loss-full-QA-v1.json').open('xb') as f:f.write(audit_path.read_bytes())
    audit.update(updated_at=at,requirements={k:v['evidence'] for k,v in requirements.items()},
        previous_goal_turn_classification='progress',current_goal_turn_classification='progress',live_joint_training=runtime,
        latest_full_scope_review=str((OUT/'review.json').relative_to(ROOT)),
        latest_actual_checkpoint_held_loss_readback=str(proof_path.relative_to(ROOT)),
        latest_full_held_question_pool=str(reserve_path.relative_to(ROOT)),
        latest_full_held_source_audit=dict(status_at_review=audit_status,owner_at_review=audit_owner,
            plan=str(audit_plan_path.relative_to(ROOT)),future_complete_report='reports/full-held-question-audit-v2/complete.json'),
        latest_full_held_native_input_preflight=str(preflight_path.relative_to(ROOT)),
        all_ten_final_requirements_incomplete=True,candidate_promoted=False,goal_achieved=False)
    atomic_json(audit_path,audit)
    paragraph=(f'Actual held-code loss improvement and expanded QA reserve, **{at}**: the retained **step2,048** native checkpoint '
        'and observed full training state reconcile exactly: **984,399,690 unique exported values**, **317 active language Adam clocks '
        'at2,048**, **71,028,428 targets** in the saved committed prefix, exact original source positions/target masks, finite moments '
        'and unchanged inactive vision/word/head values. The large mutable resume was fingerprinted rather than duplicated; the native '
        'checkpoint and compact actual tensor hashes/private RNG/journal capture remain retained.\n\n'
        'On the **same128 source-held packed windows /524,160 targets**, producer-measured next-token NLL fell **0.869945→0.805115 '
        '(7.45%)**; perplexity fell **2.38678→2.23695**. Original held input bytes and protocol were independently checked. '
        '**The reader did not repeat a neural loss forward. This is not a new coding-correctness, broad-knowledge or image gain.** '
        'Latest completed code scores remain **0/24 completion,1/24 instruction** for both old CE/RL arms, without an established RL advantage.\n\n'
        'The full pinned **12,032-question MMLU-Pro** source reserve includes **11,472 not previously sampled locally**. '
        'All **16,961 original QA inputs** (MMLU12,032, GSM8K1,319, NQ3,610) fit unchanged native generation budgets '
        'without clipping or skipped questions. A CPU-only exact question-substring audit covers the complete current/future native '
        'inputs and three conservative older possible source pools (**3,250,975,085 input positions including repetitions/potential '
        f'exposure**, not new training tokens). Its status in this review is **{audit_status}**; no fresh QA generation has occurred. '
        'No-exact-match results will not certify semantic independence, complete historical consumption or image-pixel text. '
        'The first audit failed before scanning because NQ original-question metadata was an object; the unchanged question string '
        'is now extracted by a separately tested v2 reader, with all old artifacts preserved.\n\n'
        f'Current broad training is **{len(journal):,}/35,284 updates and{targets:,}/467,581,587 targets** in this snapshot, '
        'with all four L40 ranks live. Complete human-v4 and anchored-code-v3 remain queued; their new production updates are zero. '
        '**All ten final requirements remain incomplete.** Evidence: '
        '[review](reports/held-code-loss-and-full-QA-v1-review/review.json), '
        '[checkpoint/loss readback](reports/phase-joint-step2048-readback-v1/complete.json), '
        '[full QA input check](reports/full-held-native-context-preflight-v1/complete.json), '
        '[held source audit plan](reports/full-held-question-audit-v2/plan.json).\n\n')
    for path in docs[:2]:
        content=path.read_text();first,tail=content.split('\n',1)
        path.write_text(first+'\n\n'+paragraph+tail.lstrip('\n'))
    print(json.dumps(dict(review=str(OUT/'review.json'),actual_main_updates=len(journal),
        actual_main_targets=targets,held_NLL_reduction=proof['relative_held_NLL_reduction'],source_audit_status=audit_status,
        all_ten_final_requirements_incomplete=True,goal_achieved=False)),flush=True)


if __name__=='__main__':main()
