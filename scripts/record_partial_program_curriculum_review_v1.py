"""Preserve the complete goal scope alongside the prepared train-only curriculum."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/partial-program-curriculum-v1-review'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists():raise ValueError('Fresh CPU-only full-scope record required')
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    previous=ROOT/'reports/held-code-loss-and-full-QA-v1-review/review.json';old=json.loads(previous.read_text())
    prep_path=ROOT/'reports/public-reference-prefix-curriculum-v1/complete.json';prep=json.loads(prep_path.read_text())
    reader_path=ROOT/'reports/public-reference-prefix-curriculum-v1-independent-reader/complete.json';reader=json.loads(reader_path.read_text())
    matches_path=ROOT/'reports/current-held-question-matches-v1-readback/complete.json';matches=json.loads(matches_path.read_text())
    if (not reader['all14884_native_views_rederived_without_producer_view_function'] or
            not reader['all16_preselected_original_references_freshly_reexecuted_and_passed'] or
            not reader['all32_prefix_only_controls_do_not_earn_full_credit'] or prep['model_updates']!=0
            or prep['admitted_to_RL'] is not False or not matches['all_recorded_current_native_matches_verified']):
        raise ValueError('Actual complete input reconstruction and fresh independent train controls required')
    for proof in [prep,reader,matches]:
        for name,digest in proof['source_files_sha256'].items():
            if file_hash(Path(name))!=digest:raise ValueError('New immutable proof input changed')
    def identity(owner,hidden=False):
        path=Path('/proc')/str(owner['pid']);fields=(path/'stat').read_text().split(') ',1)[1].split()
        if fields[19]!=owner['ticks'] or fields[0] in ['Z','X']:raise ValueError('Exact live owner identity required')
        children=(path/'task'/str(owner['pid'])/'children').read_text().strip()
        env=(path/'environ').read_bytes().decode().split('\0');visible=next((v.split('=',1)[1] for v in env if v.startswith('CUDA_VISIBLE_DEVICES=')),None)
        if hidden and (children or visible!=''):raise ValueError('Waiting/audit controller must remain CPU-hidden and childless')
        return dict(pid=owner['pid'],ticks=fields[19],state=fields[0],children=children,CUDA_VISIBLE_DEVICES=visible)
    phase_path=ROOT/'reports/phase-joint-v3/plan.json';phase=json.loads(phase_path.read_text())
    human_path=ROOT/'reports/phase-human-preference-v4-queue.json';human=json.loads(human_path.read_text())
    code_path=ROOT/'reports/broad-anchored-code-v3-queue.json';code=json.loads(code_path.read_text())
    audit_path=ROOT/'reports/full-held-question-audit-v2/plan.json';audit=json.loads(audit_path.read_text())
    audit_root=audit_path.parent
    future_human_path=audit_root/'future_human.json';future_code_path=audit_root/'future_code.json'
    if (json.loads(future_human_path.read_text())['question_counts_by_benchmark']!={'MMLU-Pro':9,'NQ-open':1}
            or json.loads(future_code_path.read_text())['question_counts_by_benchmark']!={'MMLU-Pro':1}):
        raise ValueError('Exact completed future input-scope match counts required')
    if (audit_root/'failure.json').exists():raise ValueError('Resolve held audit failure before recording an active audit')
    audit_finished=(audit_root/'complete.json').exists()
    at=datetime.now(timezone.utc).isoformat();workers=[identity(p) for p in json.loads((ROOT/'reports/phase-joint-v3/children.json').read_text()) if p['subject'].startswith('train-rank-')]
    if len(workers)!=4:raise ValueError('All four original broad-training workers required')
    journal=[]
    for line in (ROOT/'reports/phase-joint-v3/training-metrics.jsonl').read_bytes().splitlines():
        try:journal.append(json.loads(line))
        except json.JSONDecodeError:break
    if [r['step'] for r in journal]!=list(range(1,len(journal)+1)):raise ValueError('Contiguous actual production journal required')
    runtime=dict(at=at,actual_main_owner=identity(phase['owner']),actual_four_main_workers=workers,
        actual_main_updates=len(journal),planned_main_updates=35284,
        actual_main_targets=int(sum(r['response_EOS_targets'] for r in journal)),planned_main_targets=467581587,
        human_waiting_owner=identity(human['owner'],True),new_code_waiting_owner=identity(code['owner'],True),
        GPU_observation=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used','--format=csv,noheader'],text=True),
        full_new_human_or_code_production_updates=0,new_full_profile_GPU_admission_pending=True,goal_achieved=False)
    OUT.mkdir();atomic_json(OUT/'actual-main-journal-prefix.json',journal);atomic_json(OUT/'runtime.json',runtime)
    source_audit_observation=json.loads((audit_root/('complete.json' if audit_finished else 'progress.json')).read_text())
    atomic_json(OUT/'source-audit-observation.json',dict(status='complete' if audit_finished else 'running',
        owner=None if audit_finished else identity(audit['owner'],True),observation=source_audit_observation))
    requirements=old['requirements']
    requirements['own_architecture']['evidence']+=(' A separate public-reference prefix curriculum is prepared using the existing own '
        'native computation: original complete code prefixes are supplied as training hints at3/2/1/0 quarters, with full original '
        'CE anchors and missing suffix/EOS targets unchanged. No extra neural model, external teacher weights, routing change or student update occurs. '
        'This is a proposed training strategy, not a globally novel architecture claim or measured capability gain.')
    requirements['public_data']['evidence']+=(' The partial-program preparation reuses exactly the existing3,721 public train references '
        '(173 documented/3,548APPS) and adds no downloaded data, generated gold programs or human labels. All14,884 native views '
        'are independently reconstructed from the original complete native packs and unchanged human response bytes.')
    requirements['serious_full_model_training']['evidence']=(requirements['serious_full_model_training']['evidence']+
        f' Latest captured main runtime is {len(journal):,}/35,284 actual updates and {runtime["actual_main_targets"]:,}/467,581,587 targets. '
        'The new partial-program strategy has zero neural generations/updates and is not applied to any live or queued recipe. '
        'Full-curriculum execution admission, real student conditional samples and fresh full-size capacity/training remain required.')
    requirements['broad_independent_benchmarks']['evidence']+=(' The question audit has now completed all current native inputs and '
        'both queued future human/code input pools. Current input membership flags60MMLU-Pro and15NQ question IDs; queued human inputs '
        'flag9MMLU/1NQ, queued code1MMLU (overlap may repeat across scopes). Every one of the75 current matches has been independently '
        'verified against an actual native batch without using the projection or automaton. None of the56 current development QA pulse '
        'stems appears among these current-curriculum matches. Curriculum membership is potential exposure, not a statement about '
        'which questions the partial checkpoint has already consumed. Older possible source pools are still required before finalizing '
        'the fresh cohort. The complete source audit is '+('complete.' if audit_finished else 'still running.'))
    requirements['any_code_correctness_and_efficiency']['evidence']+=(' A partial-program RL curriculum now has32 source-preselected '
        'training episode controls across16 original documented/APPS difficulty-balanced programs. All16 unchanged original references '
        'pass; all32 partial-prefix-then-EOS controls fail to earn full credit, independently reexecuted with retained raw results. '
        'This establishes nontrivial conditional exercises only. There are no student conditional samples, reward-density measurements, '
        'new training updates or new unassisted code-correctness/efficiency gains. Full held code tasks must remain unhinted.')
    if len(requirements)!=10 or any(r['status']!='incomplete_for_requested_final_model' for r in requirements.values()):
        raise ValueError('All ten final requirements must remain incomplete')
    files=[Path(__file__),previous,prep_path,reader_path,matches_path,phase_path,human_path,code_path,audit_path,
        future_human_path,future_code_path,
        OUT/'actual-main-journal-prefix.json',OUT/'runtime.json',OUT/'source-audit-observation.json']
    report=dict(at=at,requirements=requirements,previous_goal_turn_classification='progress',current_goal_turn_classification='progress',
        actual_runtime=runtime,partial_program_preparation=prep,independent_fresh_execution_readback=reader,
        current_question_matches_actual_batch_readback=matches,source_audit_complete_at_review=audit_finished,
        strategy_applied_to_live_or_queued_training=False,student_conditional_generations=0,partial_program_model_updates=0,
        new_unassisted_coding_or_all_domain_gain_measured=False,earlier_held_next_token_loss_gain_preserved=True,
        all_ten_final_requirements_incomplete=True,no_remaining_scope_removed=True,
        source_files_sha256={str(p):file_hash(p) for p in files},candidate_promoted=False,goal_achieved=False)
    atomic_json(OUT/'review.json',report)
    master_path=ROOT/'reports/completion-audit.json';docs=[ROOT/'README.md',ROOT/'RESEARCH.md',master_path]
    protected=set().union(*(set(p['source_files_sha256']) for p in [phase,human,code,audit]))
    if any(str(p) in protected for p in docs):
        print(json.dumps(dict(review=str(OUT/'review.json'),root_documents_source_bound_not_modified=True)));return
    master=json.loads(master_path.read_text())
    with (ROOT/'reports/completion-audit-before-partial-program-curriculum-v1.json').open('xb') as f:f.write(master_path.read_bytes())
    master.update(updated_at=at,requirements={k:v['evidence'] for k,v in requirements.items()},
        latest_full_scope_review=str((OUT/'review.json').relative_to(ROOT)),live_joint_training=runtime,
        previous_goal_turn_classification='progress',current_goal_turn_classification='progress',
        partial_program_curriculum=dict(prepared_native_views=14884,original_train_sources=3721,
            original_reference_controls_freshly_reexecuted=16,prefix_then_EOS_controls_freshly_reexecuted=32,
            strategy_applied_to_live_or_queued_training=False,student_conditional_generations=0,model_updates=0,
            preparation=str(prep_path.relative_to(ROOT)),independent_reader=str(reader_path.relative_to(ROOT))),
        current_question_overlap_actual_batch_readback=str(matches_path.relative_to(ROOT)),
        all_ten_final_requirements_incomplete=True,candidate_promoted=False,goal_achieved=False)
    atomic_json(master_path,master)
    paragraph=(f'Partial-program training curriculum prepared, **{at}**: **14,884 native views** cover all **3,721 existing public '
        'train programs** at3/2/1/0 retained-reference quarters. Every complete original CE anchor and remaining suffix/EOS target '
        'is preserved. A separate reader reconstructed all views without the producer function and freshly reexecuted the '
        '**16 source-preselected full references** plus **32 partial-prefix-then-EOS controls**: all references passed and no '
        'partial-prefix control earned full credit. This prepares nontrivial conditional exercises; it does not demonstrate that '
        'our student has learned them. **Student conditional generations and training updates are zero.** Full execution admission, '
        'real student sampling and fresh full-size capacity/training remain pending. The live broad/human/code pipeline is unchanged; '
        'unassisted held code tests must remain unhinted.\n\n'
        'The complete current and queued human/code native input question scans have finished. The **75 current input matches '
        '(60MMLU-Pro/15NQ)** were separately verified from actual native batches; none matches the current56-question development '
        'pulse. This flags potential curriculum exposure, not exact already-consumed checkpoint ancestry. The older possible source '
        'scan continues before a fresh cohort can be finalized.\n\n'
        f'Current broad training is **{len(journal):,}/35,284 updates** in this snapshot, on all four L40s. The measured held-code '
        'NLL reduction remains **7.45%**; no new unassisted coding, broad-knowledge or image gain is established. '
        '**All ten final requirements remain incomplete.** Evidence: '
        '[review](reports/partial-program-curriculum-v1-review/review.json), '
        '[fresh execution controls](reports/public-reference-prefix-curriculum-v1-independent-reader/complete.json), '
        '[actual native question matches](reports/current-held-question-matches-v1-readback/complete.json).\n\n')
    for path in docs[:2]:
        first,tail=path.read_text().split('\n',1);path.write_text(first+'\n\n'+paragraph+tail.lstrip('\n'))
    print(json.dumps(dict(review=str(OUT/'review.json'),actual_main_updates=len(journal),
        prepared_native_views=14884,actual_fresh_reference_controls=16,actual_fresh_prefix_controls=32,
        student_generations=0,model_updates=0,all_ten_final_requirements_incomplete=True)),flush=True)


if __name__=='__main__':main()
