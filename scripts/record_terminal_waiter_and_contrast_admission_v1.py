"""Preserve full scope, the concrete lifecycle repair, and measured outcomes."""
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.exact_waiting_process_retirement_v1 import terminal

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/terminal-waiter-and-contrast-admission-v1-review'


def read(path):
    return json.loads(Path(path).read_text())


def birth(identity):
    if terminal(identity):
        return dict(**identity, observed_terminal=True)
    fields = (Path('/proc') / str(identity['pid']) / 'stat').read_text().split(') ', 1)[1].split()
    return dict(pid=identity['pid'], ticks=fields[19], state=fields[0], observed_terminal=False)


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or OUT.exists():
        raise ValueError('A fresh CPU-only full-scope repair review is required')
    master_path = ROOT / 'reports/completion-audit.json'; master = read(master_path)
    previous_path = ROOT / master['latest_full_scope_review']; requirements = deepcopy(read(previous_path)['requirements'])
    handoff_path = ROOT / 'reports/terminal-waiter-pipeline-v1-handoff/complete.json'; handoff = read(handoff_path)
    receipt_path = ROOT / 'reports/legacy-photo-waiter-terminal-v1/complete.json'; receipt = read(receipt_path)
    admitted_path = ROOT / 'reports/CodeContests-execution-contrast-v1-readback-admission/complete.json'; admitted = read(admitted_path)
    main_path = ROOT / 'reports/completed-phase-joint-v3-CPU-readback-v1/complete.json'; completed = read(main_path)
    native_path = ROOT / 'reports/completed-main-native-code-CPU-v2/complete.json'; native = read(native_path)
    metrics_path = ROOT / 'reports/completed-main-metrics-v1/complete.json'; metrics = read(metrics_path)
    if (not handoff['explicit_terminal_waiter_handoff_complete'] or handoff['actual_new_focused_controls_passed'] != 27
            or not all(terminal(row['owner']) for row in handoff['old_pipeline_owners'].values())
            or not terminal(receipt['original_waiter']) or handoff['old_processes_signalled'] != 0
            or admitted['actual_counts']['complete_candidate_pairs'] != 1620
            or not admitted['every_original_source_and_candidate_pair_and_actual_raw_counterexample_rebuilt']
            or not admitted['every_admitted_whole_sequence_and_target_EOS_count_cold_read_and_batched_exact']
            or not admitted['dataset_admitted_to_machine_preference_training']
            or completed['actual_original_updates'] != 35284):
        raise ValueError('The actual terminal lifecycle handoff, full main and separate complete preference admission must finish')
    for name, count in [('terminal-waiter-gpu-lease-v2-tests.log',3), ('terminal-waiter-pipeline-drivers-v1-tests.log',24)]:
        if f'{count} passed' not in (ROOT/'reports'/name).read_text():
            raise ValueError('All actual revised ownership and pipeline controls must pass')
    if any(frame['actual_native_verdict_differences'] for frames in native['actual_frame_outcomes'].values() for frame in frames.values()):
        raise ValueError('Every native program reexecution must reproduce its original syntax/functional verdict')
    current = ROOT / 'reports/code-first-continuation-v3'; plan = read(current/'plan.json')
    if (plan['planned_steps'] != 7558 or not plan['complete_public_state_source_epoch_once']
            or plan['actual_counts']['total_unique'] != 984399690):
        raise ValueError('The same whole state-inclusive code-first recipe must remain intact')
    protected = dict(plan['source_files_sha256']); owners = {}
    for name, row in handoff['new_waiting_pipeline'].items():
        q = read(row['queue']); actual = birth(q['owner'])
        if actual['observed_terminal']:
            raise ValueError('Every newly applied pipeline owner must remain live in this snapshot')
        owners[name] = actual; protected.update(q['source_files_sha256'])
    children = read(current/'children.json'); live_children = [dict(birth(p), subject=p['subject']) for p in children if not terminal(p)]
    progress_path = current/'progress.json'; progress = read(progress_path) if progress_path.exists() else None
    capacity_path = current/'capacity.json'; capacity = read(capacity_path) if capacity_path.exists() else None
    if (current/'failure.json').exists() or (ROOT/'reports/code-first-continuation-v3-parent-failure.json').exists():
        raise ValueError('Preserve and resolve any actual new source-stage failure before reporting progress')
    if not live_children:
        raise ValueError('The repaired run must have actual live workers, not only a queue or plan')
    at = datetime.now(timezone.utc).isoformat()
    runtime = dict(at=at, applied_pipeline_owners=owners, actual_code_first_workers=live_children,
        code_first_capacity_complete=capacity is not None and capacity.get('passed') is True,
        actual_code_first_production_updates=0 if progress is None else progress['actual_updates'],
        actual_GPU_observation=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used','--format=csv,noheader'],text=True),
        actual_GPU_process_observation=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True))
    scores = native['actual_endpoint_totals']
    changes = {
        'own_architecture': ' The next full-model continuation is now the version3copy of the same state-inclusive recipe, with no new neural architecture/weights/teacher or parameter arithmetic. Its GPU ownership is repaired by actual exclusive kernel flock ownership and an immutable receipt for the terminal legacy process, replacing the impossible requirement for that old birth to stay asleep. No fabricated waiter is created. All new parameter/cold/recovery gates remain mandatory.',
        'public_data': ' All3959original source dispositions,285980retained positive-reference case outputs and2256complete native first-negative pairs now independently reconstruct. The separately admitted1620machine preference pairs retain1713639chosen/1687251rejectedcontext positions and807045chosen/780657rejectedresponse/EOS targets. Both are unmodified original public train programs under the same full prompt. Actual negative observations comprise46151fresh case runs; each admitted negative has a completed output mismatch repeated byte-exactly in another fresh interpreter. All32953held stems scan both actual native inputs, with0component exclusions. This does not certify semantic or all-history independence. The unchanged candidate stays unadmitted; the separate pack is admitted only to machineDPO/chosen-anchorCE and is not yet inserted into the running production lineage.',
        'under_1b_parameters': ' The repaired continuation still trains the complete984399690-value own main student, with all317language/112vision/16word tensors and the same full445leaf Adam/recovery/cold checks. Human/code actor branches remain planned at999706442values and final dense export at984399690. Source preference data and the new ownership module add no neural parameters.',
        'text_and_images': ' Complete initial/final main development cohorts remain identical:20OCRdeliveries/16images,15objectdeliveries/14photos and32captionedphotos. OCR exact match stays0/20although character error rate improves0.95417to0.88333. Proper-image object exact results go4/15to3/15; image-to-caption retrieval remains1/32atboth. These are small custom development checks, not proof of broad visual ability. All visual/code/text/word replay and full3088AI2Dtimes3conditions remain in the repaired lineage.',
        'human_feedback': ' The1620new pairs are explicit machine execution preferences and never fresh human labels. Their separate API admits only machineDPO and chosen-reference anchorCE; rejected programs are forbidden as positiveCEtargets. Existing full85786human pairs per chosenCE/DPO arm,4642held human pairs and3721execution-training sources per code arm remain unchanged in human-v7/code-v6. Their factory math/own-reference semantics remain the tested version2implementations. No new machine-preference model training has run.',
        'serious_full_model_training': ' The original main and all native assessments finish normally at35284updates; all four training ranks and original registered assessment children are terminal. The restored old photo-data waiter then exits with its own preserved error, Final original photo-pair configuration differs. The old code-first-v2controller aborts before GPU ownership, after writing a complete CPU initializer; its seven dependent controllers also end without production updates. Every old file, initializer, error and queue is preserved. Eight tested replacement controllers now retain the same7558source updates, all58488CODE/2616output/1310state examples and12replay families. Their actual live four-GPU capacity or production workers are captured in this review. This is an explicit new attempt after authoritative failure, not an age/timeout restart or new model-quality success.',
        'high_end_llm_capability': f' Completed main native initial/final reexecutions show syntax{scores["initial"]["syntax"]}/48to{scores["trained"]["syntax"]}/48andfunctional{scores["initial"]["functional"]}/48to{scores["trained"]["functional"]}/48. Secondary ordinaryPythonresults are1/24to0/24. Native verdicts reproduce every original output. The new full-size source objective and preference data are interventions to investigate, not demonstrated high-end capability or GPTparity.',
        'broad_independent_benchmarks': ' Current fixed small development checks now have a standalone PNG/SVG/CSV artifact with complete before/after counts. Generated knowledge goes0/56to2/56andanswer-likelihood3/56to5/56, both below the roughly11.5percent chance baseline for these questions; math exact goes1/16to0/16. The same selected question/image identities and requested budgets are checked. All16961QAquestions,3088AI2Dtimes3conditions and600MBPP/1816cases remain at eight fixed endpoints through the new paths. These full final independent outcomes remain incomplete.',
        'gpu_runtime': ' The original legacy waiter birth is authoritatively terminal; its exact preserved failure/status and original main restoration are recorded. The new owner requires the exact live parent to hold the actual exclusive kernel flock, rejects a live/reused original birth appropriately and cleans only registered exact children via pidfds. Actual CPU tests preserve an unrelated process. All27new lifecycle and revised-driver controls pass; no old process is signalled. Eight new controllers are concretely launched and handed off, and actual four-rank capacity work has begun. New source/held programs, previous model files and all data-preference artifacts remain intact.',
        'any_code_correctness_and_efficiency': ' All96retained original initial/final native programs are freshly reexecuted with the correct Python/JavaScript dispatcher; all48Pythonprograms also use the separate ordinary-stdlib protocol. A first interim reader accidentally used a C++backend on these languages; that native result is explicitly qualified invalid and preserved, then replaced by version2with24canonical native reference controls and exact original verdict reproduction. No wrong intermediate syntax/functional aggregate is used. Original generation action ids/EOS were not retained by this main producer, so they are not reconstructed or certified. Full future MBPP/QA/image controllers retain their stronger raw-action protocol. Both main endpoints have0/48empirical efficiency passes; arbitrary-code and asymptotic claims remain unsupported.',
    }
    for key, note in changes.items(): requirements[key]['evidence'] += note
    if len(requirements) != 10 or any(r['status'] != 'incomplete_for_requested_final_model' for r in requirements.values()):
        raise ValueError('All ten original final requirements remain explicitly incomplete')
    OUT.mkdir()
    atomic_json(OUT/'runtime.json', runtime)
    evidence = [Path(__file__), previous_path, handoff_path, receipt_path, admitted_path, main_path,
        native_path, metrics_path, OUT/'runtime.json', ROOT/'reports/completed-main-native-code-CPU-v1-qualified-usage.json',
        ROOT/'reports/terminal-waiter-gpu-lease-v2-tests.log',ROOT/'reports/terminal-waiter-pipeline-drivers-v1-tests.log',
        ROOT/'src/slm_lab/code_first_endpoints_v3.py']
    atomic_json(OUT/'review.json',dict(at=at, requirements=requirements, previous_full_scope_review=str(previous_path.relative_to(ROOT)),
        actual_runtime=runtime, source_files_sha256={str(p):file_hash(p) for p in evidence},
        previous_goal_turn_classification='progress',current_goal_turn_classification='progress',
        all_ten_final_requirements_incomplete=True,no_original_scope_removed=True,candidate_promoted=False,goal_achieved=False))
    docs=[ROOT/'README.md',ROOT/'RESEARCH.md',master_path]
    if any(str(p) in protected for p in docs):raise ValueError('Review complete; an actual fixed-source root document must remain unchanged')
    (OUT/'previous-completion-audit.json').write_bytes(master_path.read_bytes())
    master.update(updated_at=at,requirements={k:v['evidence'] for k,v in requirements.items()},
        latest_full_scope_review=str((OUT/'review.json').relative_to(ROOT)),latest_actual_main_runtime=runtime,
        latest_code_first_handoff=str(handoff_path.relative_to(ROOT)),latest_code_first_pipeline=str((OUT/'runtime.json').relative_to(ROOT)),
        latest_queued_new_code_continuation='reports/code-first-continuation-v3-queue.json',
        latest_full_original_MBPP_queue='reports/full-original-MBPP-pipeline-v4-queue.json',
        latest_complete_main_native_reexecution=str(native_path.relative_to(ROOT)),
        latest_completed_main_metrics='reports/completed-main-metrics-v1/complete.json',
        latest_public_execution_contrast_preparation='reports/CodeContests-execution-contrast-candidate-v1/complete.json',
        latest_public_execution_contrast_admission=str(admitted_path.relative_to(ROOT)),
        latest_terminal_legacy_waiter_receipt=str(receipt_path.relative_to(ROOT)),
        current_goal_turn_classification='progress',all_ten_final_requirements_incomplete=True,candidate_promoted=False,goal_achieved=False)
    atomic_json(master_path,master)
    paragraph=(f'Completed main, repaired ownership and admitted machine preferences, **{at}**: '
        'the original **35,284-update** run and native assessments are complete. Fresh program executions '
        'show **1/48 → 0/48 functional coding**, **40/48 → 29/48 syntax**, and secondary ordinary Python '
        '**1/24 → 0/24**. Small development checks show knowledge **0/56 → 2/56**, math **1/16 → 0/16**, '
        'OCR exact **0/20 → 0/20** and objects **4/15 → 3/15**. These are limited development cohorts, '
        'and no frontier capability is established.\n\n'
        'After restoration, the old photo waiter ended with its preserved configuration error. '
        'The failed successor and every prior artifact are retained. **27 new checks pass** for a '
        'terminal-waiter ownership path with actual kernel flock verification; all eight replacement '
        'controllers are launched, and the new four-GPU capacity or production run is live. '
        'The same **7,558-update** code/state/replay curriculum and subsequent full human DPO, execution RL '
        'and eight-endpoint assessments remain. A separate complete readback admits **1,620 original '
        'code preference pairs**, backed by **46,151 fresh negative-case executions**, for future machine '
        'DPO/chosen-code anchoring; they have not yet been used in new model training. '
        '**All ten final requirements remain incomplete.** '
        '[Completed metrics](reports/completed-main-metrics-v1/completed-main-metrics.png), '
        '[handoff](reports/terminal-waiter-pipeline-v1-handoff/complete.json), '
        '[full review](reports/terminal-waiter-and-contrast-admission-v1-review/review.json). '
        'Earlier entries are historical snapshots.\n\n')
    for path in docs[:2]:
        first,rest=path.read_text().split('\n',1);path.write_text(first+'\n\n'+paragraph+rest.lstrip('\n'))
    print(json.dumps(dict(review=str(OUT/'review.json'),admitted_machine_pairs=1620,
        code_first_capacity_complete=runtime['code_first_capacity_complete'],
        code_first_production_updates=runtime['actual_code_first_production_updates'],
        all_ten_final_requirements_incomplete=True,goal_achieved=False)),flush=True)


if __name__=='__main__':main()
