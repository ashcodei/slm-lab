"""Record admitted CE sources and live execution-grounded candidate preparation."""
from datetime import datetime,timezone
import json
import math
import os
from pathlib import Path
import subprocess

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/screened-CodeContests-execution-v1-review'


def current(owner,waiting=False):
    proc=Path('/proc')/str(owner['pid']);fields=(proc/'stat').read_text().split(') ',1)[1].split()
    if fields[19]!=owner['ticks'] or fields[0] in {'Z','X'}:raise ValueError('Exact live birth identity required')
    children=(proc/'task'/str(owner['pid'])/'children').read_text().strip()
    visible=next((s.split('=',1)[1] for s in (proc/'environ').read_bytes().decode().split('\0') if s.startswith('CUDA_VISIBLE_DEVICES=')),None)
    if waiting and (children or visible!=''):raise ValueError('Original queued GPU successor must remain hidden and childless')
    return dict(pid=owner['pid'],ticks=fields[19],state=fields[0],children=children,CUDA_VISIBLE_DEVICES=visible)


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists():raise ValueError('Fresh CPU-only full-scope evidence update required')
    master_path=ROOT/'reports/completion-audit.json';master=json.loads(master_path.read_text())
    previous_path=ROOT/master['latest_full_scope_review'];previous=json.loads(previous_path.read_text());requirements=previous['requirements']
    paths=dict(native=ROOT/'reports/code-contests-complete-curriculum-v1-readback/complete.json',
        exposure=ROOT/'reports/CodeContests-native-held-text-v2/complete.json',
        view_readback=ROOT/'reports/CodeContests-held-text-view-v1-readback/complete.json',
        CE_manifest=ROOT/'data/code-contests-text-screened-CE-view-v1/manifest.json',
        CE_admission=ROOT/'data/code-contests-text-screened-CE-view-v1/admission.json',
        actual_API=ROOT/'reports/CodeContests-CE-view-v1-actual-API-check.json')
    reports={k:json.loads(p.read_text()) for k,p in paths.items()}
    for name in ['native','exposure','view_readback']:FileBindings(reports[name]['source_files_sha256']).check(full=True)
    if (reports['CE_admission']['admitted_to_CE_training'] is not True or reports['CE_admission']['admitted_to_execution_reward_training'] is not False
            or reports['CE_admission']['manifest_sha256']!=file_hash(paths['CE_manifest'])
            or reports['actual_API']['authoritative_exec_exit_code']!=0 or reports['actual_API']['actual_native_rows_checked']!=58488):
        raise ValueError('Exact screened CE-only admission and actual whole native API check required')
    ce=reports['CE_manifest'];exposure=reports['exposure'];readback=reports['view_readback']
    reference_plan_path=ROOT/'reports/CodeContests-full-Python-references-v1/plan.json'
    trace_plan_path=ROOT/'reports/CodeContests-public-execution-traces-candidate-v1/plan.json'
    reference_plan=json.loads(reference_plan_path.read_text());trace_plan=json.loads(trace_plan_path.read_text())
    reference_progress=json.loads(reference_plan_path.with_name('progress.json').read_text())
    trace_progress=json.loads(trace_plan_path.with_name('progress.json').read_text())
    for path in [reference_plan_path,trace_plan_path]:
        if path.with_name('failure.json').exists():raise ValueError('Original CPU source job has failed')
    main_path=ROOT/'reports/phase-joint-v3/plan.json';main_plan=json.loads(main_path.read_text())
    main_progress=json.loads(main_path.with_name('progress.json').read_text())
    workers=[current(r) for r in json.loads(main_path.with_name('children.json').read_text()) if r['subject'].startswith('train-rank-')]
    if len(workers)!=4:raise ValueError('Every original main worker required')
    names=['phase-human-preference-v4','broad-anchored-code-v3','full-held-QA-pipeline-v1','full-AI2D-pipeline-v1']
    queues={n:json.loads((ROOT/'reports'/(n+'-queue.json')).read_text()) for n in names}
    at=datetime.now(timezone.utc).isoformat()
    runtime=dict(at=at,main_owner=current(main_plan['owner']),original_main_workers=workers,
        main_actual_updates=main_progress['actual_updates'],main_planned_updates=35284,main_phase=main_progress['phase'],
        original_waiting_GPU_successors={n:current(q['owner'],True) for n,q in queues.items()},
        original_reference_CPU_owner=current(reference_plan['owner']),original_public_trace_CPU_owner=current(trace_plan['owner']),
        reference_progress_snapshot=reference_progress,public_trace_candidate_progress_snapshot=trace_progress,
        GPU_observation=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used','--format=csv,noheader'],text=True))
    initial_path=main_path.with_name('held-packed-initial.json');initial=json.loads(initial_path.read_text())
    held_paths=[p for p in main_path.parent.glob('held-packed-*.json') if p.stem.rsplit('-',1)[-1].isdigit()]
    latest_path=max(held_paths,key=lambda p:int(p.stem.rsplit('-',1)[-1]));latest=json.loads(latest_path.read_text())
    for key in ['original_split','selected_original_positions','original_valid_population','evaluated_windows','complete_causal_targets','maximum_original_context','original_manifest_sha256']:
        if latest[key]!=initial[key]:raise ValueError('Unchanged held token input protocol required')
    held=dict(step=int(latest_path.stem.rsplit('-',1)[-1]),initial_NLL=initial['token_mean_NLL'],latest_NLL=latest['token_mean_NLL'],
        relative_reduction=(initial['token_mean_NLL']-latest['token_mean_NLL'])/initial['token_mean_NLL'],perplexity=math.exp(latest['token_mean_NLL']),
        input_protocol_and_report_arithmetic_checked=True,actual_neural_loss_is_producer_observation=True,
        no_fresh_weight_Adam_CUDA_replay_or_generated_quality_claim=True)
    requirements['public_data']['evidence'] += (
        ' All107641selected CodeContests presentations were rebuilt from original publisher code/questions, including40634quarantined '
        'presentations and all67007retained native examples. A stronger actual-input scan covers every96,920,929native context '
        'positions against32953named held stems, with word-boundary literal checks and fixed informative5word-shingle copy '
        'thresholds. It finds600source-question/held-question lexical-copy witnesses and12native occurrences of the generic '
        'one-word MMLU stem Inflation, with no normalized held reference-program matches. Generic occurrences are not treated '
        'as equivalent questions. Conservative component quarantine excludes6946presentations across591additional components. '
        'All600lexical and12literal witnesses were independently rebuilt by direct string/word-set comparisons, without the '
        'producer automaton/inverted search; all67007actual native hashes/roles/view positions and budgets were rederived. '
        'Negative searches are not a second independent exhaustive scan; semantic/all-ancestry independence remains unproved. '
        'A separate CE-only view now admits58488whole publisher-positive examples from8543original training problems, '
        '88,610,948context positions and54,063,335response/EOS targets. Every local API position, native mask and source role '
        'was actually checked, with invalid indices rejected. The raw pack remains unadmitted; execution rewards and new '
        'model training are not admitted or started by this CE source-view artifact. Original held populations are unchanged. '
        'The failed new audit setup caused by U+0085inside one valid original question was preserved; a tested physical '
        'JSONL reader now retains the original question. The existing frozen full-QA pipeline already reads physical '
        'records and was unaffected. No original data or frozen training source was edited.')
    requirements['serious_full_model_training']['evidence'] += (
        f' The original full-weight main run is freshly live at{main_progress["actual_updates"]}/35284updates on allfour '
        f'original L40workers. Its latest fixed-protocol held token observation is step{held["step"]},NLL{held["latest_NLL"]}; '
        'report/input arithmetic is checked without repeating that neural forward or claiming a generated capability score. '
        'All original human-preference, code-RL, text-QA and visual-QA successor owners remain queued/hidden/childless. '
        'The new CE source view and public execution-prediction candidate have not updated or replaced any live model.')
    requirements['any_code_correctness_and_efficiency']['evidence'] += (
        f' A four-CPU-worker source assessment now considers all3959screened train problems with a selected Python3 '
        f'program, one original publisher-order reference per source. At the new snapshot it has '
        f'{reference_progress["original_training_problems_completed"]}completed source dispositions and '
        f'{reference_progress["actual_counts"].get("all_original_finite_cases_pass",0)}full original public/private/generated '
        f'case passes across{reference_progress["actual_counts"].get("actual_original_case_runs",0)}actual fresh-interpreter '
        'case runs. Source programs and test populations remain unmodified; file interfaces, payloads, CPython versions, '
        'strict token-output checker and resource incompatibilities are qualifications, not new wrong-code labels. '
        'Expected outputs remain outside the OS-confined candidate filesystem. Three actual tests check ordinary '
        'program/main/stdin-buffer behavior, all3case kinds, expected-value separation, wrong-last-case full coverage, '
        'incomplete prefixes, token distinctions, infinite execution and whole-population bounds. Execution rewards '
        'still require complete source assessment and independent actual-stdout verdict readback/admission. '
        'An auxiliary shared-language-head program-plus-public-input to actual-stdout task is implemented, with no '
        'added neural parameters or pretrained teacher. Only references passing all original finite cases qualify; '
        'private/generated values never enter its model prompt or target, and full4096contexts are never clipped. '
        'Three formatting/source/context tests pass. Its live candidate writer has produced '
        f'{trace_progress["actual_counts"].get("complete_public_execution_traces",0)}complete examples so far, remains '
        'unadmitted until all3959original references have dispositions and the native/source-stdout readback passes, '
        'and does not constitute a model gain. The completed earlier contrastive trial remains22/48->31/48syntax '
        'but0/48functional/efficiency. No final model or high-end capability claim follows from training-source execution.')
    if len(requirements)!=10 or any(r['status']!='incomplete_for_requested_final_model' for r in requirements.values()):
        raise ValueError('Every original final requirement remains incomplete')
    OUT.mkdir();evidence=[Path(__file__),previous_path,*paths.values(),reference_plan_path,trace_plan_path,main_path,initial_path,latest_path,
        ROOT/'reports/held-text-overlap-v1-tests.log',ROOT/'reports/source-jsonl-records-v1-tests.log',
        ROOT/'reports/code-contests-Python-cases-v1-final-tests.log',ROOT/'reports/code-contests-public-execution-trace-v1-final-tests.log',
        ROOT/'reports/CodeContests-native-held-text-v1-failed-setup.json']
    report=dict(at=at,requirements=requirements,previous_full_scope_review=str(previous_path.relative_to(ROOT)),actual_runtime=runtime,
        admitted_CE_view=ce,complete_native_target_readback=reports['native'],complete_native_exposure_audit=exposure,
        complete_positive_witness_and_view_readback=readback,latest_same_protocol_held_token_loss=held,
        new_reference_and_public_execution_candidates_not_model_assessments=True,
        source_files_sha256={str(p):file_hash(p) for p in evidence},previous_goal_turn_classification='progress',current_goal_turn_classification='progress',
        all_ten_final_requirements_incomplete=True,no_original_scope_removed=True,
        no_new_functional_coding_knowledge_or_visual_model_gain_claim=True,candidate_promoted=False,goal_achieved=False)
    atomic_json(OUT/'review.json',report)
    protected=set(main_plan['source_files_sha256'])
    for q in [*queues.values(),reference_plan,trace_plan]:protected.update(q['source_files_sha256'])
    group=json.loads((ROOT/'reports/AI2D-fixed-exposure-strata-v1/plan.json').read_text());protected.update(group['source_files_sha256'])
    docs=[ROOT/'README.md',ROOT/'RESEARCH.md',master_path]
    if any(str(p) in protected for p in docs):
        print(json.dumps(dict(review=str(OUT/'review.json'),bound_root_documents_preserved=True)),flush=True);return
    (ROOT/'reports/completion-audit-before-screened-CodeContests-execution-v1.json').write_bytes(master_path.read_bytes())
    master.update(updated_at=at,requirements={k:r['evidence'] for k,r in requirements.items()},
        latest_full_scope_review=str((OUT/'review.json').relative_to(ROOT)),latest_actual_main_runtime=runtime,
        latest_admitted_CodeContests_CE_view=str(paths['CE_manifest'].relative_to(ROOT)),
        latest_CodeContests_original_reference_plan=str(reference_plan_path.relative_to(ROOT)),
        latest_public_execution_prediction_candidate_plan=str(trace_plan_path.relative_to(ROOT)),
        current_goal_turn_classification='progress',all_ten_final_requirements_incomplete=True,candidate_promoted=False,goal_achieved=False)
    atomic_json(master_path,master)
    paragraph=(f'Screened code supervision and execution-grounded candidates, **{at}**: all **67,007 retained native targets** '
        'are rebuilt from original source. A full native-input/lexical-copy screen and direct readback of all **600 lexical '
        'copy witnesses and 12 generic literal occurrences** yields a separate **58,488-example CE-only view from 8,543 '
        'training problems**, with 88,610,948 context positions and 54,063,335 output/EOS targets. All original held '
        'assessment cases remain intact. This view uses publisher correctness labels; execution rewards need separate '
        'admission. The original raw pack stays unadmitted.\n\n'
        f'Four CPU workers are checking every original public/private/generated case for **3,959 Python 3 references**; '
        f'the snapshot has **{reference_progress["original_training_problems_completed"]} source dispositions**, '
        f'**{reference_progress["actual_counts"].get("all_original_finite_cases_pass",0)} complete finite-case passes** and '
        f'**{reference_progress["actual_counts"].get("actual_original_case_runs",0):,} actual case runs**. '
        'A separate auxiliary candidate teaches program-and-public-input to actual observed output, with no new '
        'parameters or pretrained teacher and no private/generated values in model text. It currently has '
        f'**{trace_progress["actual_counts"].get("complete_public_execution_traces",0)} complete examples**; finalization, '
        'readback and admission are pending. Neither new dataset has updated a model. The original main run remains '
        f'active at **{main_progress["actual_updates"]:,}/35,284 updates on four L40s**, with all original human/code/text/visual '
        'successors retained. The earlier custom trial improved syntax but remains 0/48 functionally correct. '
        '**All ten final requirements remain incomplete.** '
        '[Review](reports/screened-CodeContests-execution-v1-review/review.json), '
        '[CE source view](data/code-contests-text-screened-CE-view-v1/manifest.json), '
        '[public execution candidate](reports/CodeContests-public-execution-traces-candidate-v1/plan.json).\n\n')
    for path in docs[:2]:
        first,rest=path.read_text().split('\n',1);path.write_text(first+'\n\n'+paragraph+rest.lstrip('\n'))
    print(json.dumps(dict(review=str(OUT/'review.json'),CE_examples=58488,main_updates=main_progress['actual_updates'],
        actual_public_trace_candidates=trace_progress['actual_counts'].get('complete_public_execution_traces',0),goal_achieved=False)),flush=True)


if __name__=='__main__':main()
