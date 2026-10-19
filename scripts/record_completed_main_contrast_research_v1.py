"""Record a completed main epoch and actual execution-preference research."""
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
OUT = ROOT / 'reports/completed-main-execution-contrast-v1-review'
QUEUES = ('code-first-continuation-v2', 'phase-human-preference-v6', 'broad-anchored-code-v5',
          'code-first-final-standalone-CPU-v2', 'full-held-QA-pipeline-v3', 'full-AI2D-pipeline-v3',
          'full-original-MBPP-pipeline-v3', 'AI2D-fixed-exposure-strata-v3')


def read(path):
    return json.loads(Path(path).read_text())


def current(identity):
    p = Path('/proc') / str(identity['pid'])
    if terminal(identity):
        return dict(**identity, observed_terminal=True)
    fields = (p / 'stat').read_text().split(') ', 1)[1].split()
    environment = (p / 'environ').read_bytes().decode().split('\0')
    visible = next((v.split('=', 1)[1] for v in environment if v.startswith('CUDA_VISIBLE_DEVICES=')), None)
    return dict(pid=identity['pid'], ticks=fields[19], state=fields[0], observed_terminal=False,
                CUDA_VISIBLE_DEVICES=visible, command=(p / 'cmdline').read_bytes().decode().replace('\0', ' ').strip())


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or OUT.exists():
        raise ValueError('A fresh CPU-only full-scope review is required')
    master_path = ROOT / 'reports/completion-audit.json'; master = read(master_path)
    previous_path = ROOT / master['latest_full_scope_review']; previous = read(previous_path)
    requirements = deepcopy(previous['requirements'])
    completed_path = ROOT / 'reports/completed-phase-joint-v3-CPU-readback-v1/complete.json'
    completed = read(completed_path)
    if (completed['actual_original_updates'] != 35284 or completed['actual_families'] != 28
            or completed['actual_changed_selected_tensors'] != 445 or completed['actual_unused_fixed_tensors'] != 16
            or not completed['all_native_parameters_exactly_equal_private_full_model']
            or not completed['every_actual_logged_source_position_and_complete_denominator_reconciled']
            or not completed['every_active_Adam_clock_matches_actual_family_updates']):
        raise ValueError('Actual full completed epoch and every final state partition must reconcile')
    early_path = ROOT / 'reports/CodeContests-contrast-early-independent-readback-v1.json'; early = read(early_path)
    if (not early['complete'] or len(early['original_observation_positions']) != 32
            or not early['every_observed_whole_program_native_EOS_mask_and_raw_counterexample_rebuilt']):
        raise ValueError('Actual separate integration readback of32retained observations is required')
    tests = {'code-contests-contrast-candidates-v1-tests.log': 3, 'code-contests-contrast-readback-v1-tests.log': 2}
    for name, count in tests.items():
        if f'{count} passed' not in (ROOT / 'reports' / name).read_text():
            raise ValueError('Every new actual execution/source/native-mask focused control must pass')
    source = ROOT / 'reports/CodeContests-execution-contrast-candidate-v1'
    reader = ROOT / 'reports/CodeContests-execution-contrast-v1-readback-admission'
    plans = {str(folder): read(folder / 'plan.json') for folder in (source, reader)}
    jobs = {}
    protected = dict(completed['source_files_sha256'])
    for name, plan in plans.items():
        if (Path(name) / 'failure.json').exists():
            raise ValueError('Preserve and resolve any actual source-job failure before reporting progress')
        process = current(plan['owner'])
        if process['observed_terminal'] and not (Path(name) / 'complete.json').exists():
            raise ValueError('An exact source job ended without complete evidence')
        jobs[name] = dict(process=process, complete=(Path(name) / 'complete.json').exists())
        protected.update(plan['source_files_sha256'])
    main_folder = ROOT / 'reports/phase-joint-v3'; main_plan = read(main_folder / 'plan.json')
    main_children = read(main_folder / 'children.json')
    original_ranks = [p for p in main_children if p['subject'].startswith('train-rank-')]
    if len(original_ranks) != 4 or not all(terminal(p) for p in original_ranks):
        raise ValueError('All four original training ranks must now be terminal')
    queues = {}
    for name in QUEUES:
        path = ROOT / 'reports' / (name + '-queue.json'); queue = read(path); process = current(queue['owner'])
        if process['observed_terminal'] or process['CUDA_VISIBLE_DEVICES'] != '':
            raise ValueError('The applied eight CPU controllers must remain live and CUDA-hidden')
        queues[name] = dict(process=process, queue_sha256=file_hash(path))
        protected.update(queue['source_files_sha256'])
    latest = read(main_folder / 'held-packed-35284.json')
    foundation = read(main_folder / 'held-packed-12853.json')
    if any(latest[k] != foundation[k] for k in ['selected_original_positions', 'complete_causal_targets', 'original_manifest_sha256']):
        raise ValueError('The final retention comparison must keep exactly the same original cohort')
    at = datetime.now(timezone.utc).isoformat()
    source_progress = read(source / ('complete.json' if (source / 'complete.json').exists() else 'progress.json'))
    source_complete = (source / 'complete.json').exists()
    admission_complete = (reader / 'complete.json').exists()
    CE_progress_path = ROOT / 'reports/code-first-continuation-v2/progress.json'
    CE_updates = read(CE_progress_path)['actual_updates'] if CE_progress_path.exists() else 0
    runtime = dict(at=at, original_main=current(main_plan['owner']), main_training_updates=35284,
        original_training_ranks_terminal=True, original_main_children=[dict(current(p), subject=p['subject']) for p in main_children],
        actual_GPU_observation=subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used', '--format=csv,noheader'], text=True),
        actual_GPU_processes=subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader'], text=True),
        main_parent_native_assessments_complete=(main_folder / 'complete.json').exists(),
        new_code_first_production_updates=CE_updates, applied_controllers=queues, new_CPU_source_jobs=jobs)
    totals = {k: sum(row[k] for row in completed['actual_source_budget'].values())
              for k in ['examples', 'complete_context_tokens', 'response_EOS_targets']}
    updates = {
        'own_architecture': ' The current full984Mmain now completes its declared source epoch. A static independent CPU reader checks all461unique full tensors, all445active Adam states and the complete own foundation parameter digest without constructing a model. The new machine-preference method uses the existing shared vocabulary/decoder with0new parameters; it contrasts whole original positive and incorrect public programs only after actual output observations. No globally novel architecture or capability gain is established.',
        'public_data': f' A new CPU source job examines all3959previously declared original Python3reference sources. It fixes the first original publisher-order Python3incorrect program before observing behavior, preserves both complete4096-bounded native programs andEOS masks, and excludes whole source components on any32953held-stem signal. Current source snapshot: {json.dumps(source_progress.get("actual_counts", {}), separators=(",", ":"))}. A separate reader waits for that exact owner and independently rebuilds every original record, both BPE sequences, every retained output verdict and all source dispositions before separate machine-DPO/chosen-anchor-CE admission. The original candidate stays unadmitted; no such pair is added to existing production queues in this turn.',
        'under_1b_parameters': ' The completed main has exactly984399690unique full student values across461unique tensors and one stored tied vocabulary. Every final native parameter equals the full private snapshot. All445selected tensors change and all16unused critic/auxiliary tensors stay exact versus the starting native weights. These are actual stored values, not a parameter estimate; final post-trained export remains pending.',
        'text_and_images': ' The completed epoch includes all28declared source families, with actual language/vision/word active update counts22960/12324/5725. Ordinary photos, diagram reasoning, image reading and real words are retained. Every family source position and whole target/context denominator reconciles with the immutable plan, journal and final private records. Native initial/final broad assessment is ongoing or newly completed in the runtime snapshot; no visual quality gain is inferred from state integrity.',
        'human_feedback': ' The new accepted/incorrect program pairs are machine execution preferences, not human judgments. The queued85786pair human chosenCE/DPO comparison and4642held human pairs remain unchanged. The separate new admission permits only machineDPO and chosen-program anchorCE; rejected programs must never become positiveCE targets. Timeouts, syntax/version failures and nonrepeatable outputs do not become algorithmic failure labels. Actual use in a new model-training stage remains unstarted.',
        'serious_full_model_training': f' The original main now finishes all35284updates:12853foundation plus22431instruction/vision updates. The static reader reconciles {totals["examples"]}selected family presentations, {totals["complete_context_tokens"]}complete context positions and{totals["response_EOS_targets"]}response/causal/EOS targets across28families. All selected family records occur once; the four declared exercise repeats and cross-family overlap mean this is not all unique original data once. All four training ranks are terminal and the same parent continues native assessment/restoration. The reader validates stored clocks/moments and CPU RNG parsing, but does not replay35284steps, recertify unarchived per-step tensors or instantiate aCUDA generator.',
        'high_end_llm_capability': f' Final fixed packed-code NLL is{latest["token_mean_NLL"]}versus{foundation["token_mean_NLL"]}atfoundation12853on exactly128windows/524160targets. Completing a source epoch is not evidence of frontier capability; retention is worse after the later phase. The older matched12853/14336native comparison remains0/48functional and efficiency passes. Current final-model native results must be read on their own original budgets before claiming any new gain.',
        'broad_independent_benchmarks': ' All original full QA16961, AI2D3088times3conditions andMBPP600/1816case comparisons at eight fixed endpoints remain intact. The independent32observation early contrast integration check rebuilds real original programs, masks and raw counterexamples but is not a held benchmark or whole3959source admission. No original benchmark is repurposed for the new machine preference.',
        'gpu_runtime': ' The original four main training ranks finish normally; their own parent now owns native initial/final assessment workers. All eight applied CUDA-hidden controllers remain live. The new negative observation job uses at most4CPUworkers; its separate reader waits on the exact process birth. Neither stops or restarts original training, replaces its frozen code, allocates extra GPUs or changes existing source admissions.',
        'any_code_correctness_and_efficiency': ' The new feedback path targets the sparse-reward problem by pairing a complete original reference that passed every finite original case with an original incorrect program having a completed wrong token output repeated byte-exactly in a second fresh interpreter. Five new controls pass, including actual standard-library programs, public-pass/private-mismatch coverage, no fallback after overflow, source-role and raw-output corruption rejection. Thirty-two real retained observations also independently reconstruct. Full original judge, custom constructive-checker equivalence, arbitrary-code correctness and asymptotic efficiency are not claimed; new production machine-preference updates remain0.',
    }
    for key, note in updates.items():
        requirements[key]['evidence'] += note
    if len(requirements) != 10 or any(row['status'] != 'incomplete_for_requested_final_model' for row in requirements.values()):
        raise ValueError('All ten original final requirements must remain incomplete')
    OUT.mkdir()
    atomic_json(OUT / 'candidate-progress-snapshot.json', source_progress)
    evidence = [Path(__file__), previous_path, completed_path, early_path, main_folder / 'training-complete.json',
                main_folder / 'held-packed-35284.json', OUT / 'candidate-progress-snapshot.json',
                *[Path(name) / 'plan.json' for name in plans], *[ROOT / 'reports' / name for name in tests]]
    atomic_json(OUT / 'review.json', dict(
        at=at, requirements=requirements, previous_full_scope_review=str(previous_path.relative_to(ROOT)),
        actual_runtime=runtime, actual_completed_main_source_totals=totals,
        current_execution_contrast_preparation_complete=source_complete,
        current_execution_contrast_admission_complete=admission_complete,
        new_contrast_pairs_not_inserted_into_existing_production_lineage=True,
        source_files_sha256={str(path): file_hash(path) for path in evidence},
        previous_goal_turn_classification='progress', current_goal_turn_classification='progress',
        no_original_scope_removed=True, all_ten_final_requirements_incomplete=True,
        candidate_promoted=False, goal_achieved=False,
    ))
    docs = [ROOT / 'README.md', ROOT / 'RESEARCH.md', master_path]
    if any(str(p) in protected for p in docs):
        raise ValueError('The review is complete; an actual fixed-source root document must not be changed')
    (OUT / 'previous-completion-audit.json').write_bytes(master_path.read_bytes())
    master.update(updated_at=at, requirements={k: v['evidence'] for k, v in requirements.items()},
        latest_full_scope_review=str((OUT / 'review.json').relative_to(ROOT)), latest_actual_main_runtime=runtime,
        latest_completed_main_full_state_readback=str(completed_path.relative_to(ROOT)),
        latest_public_execution_contrast_preparation='reports/CodeContests-execution-contrast-candidate-v1/plan.json',
        latest_public_execution_contrast_admission='reports/CodeContests-execution-contrast-v1-readback-admission/plan.json',
        latest_public_execution_contrast_early_readback=str(early_path.relative_to(ROOT)),
        previous_goal_turn_classification='progress', current_goal_turn_classification='progress',
        all_ten_final_requirements_incomplete=True, candidate_promoted=False, goal_achieved=False)
    atomic_json(master_path, master)
    paragraph = (
        f'Completed main epoch and execution-preference research, **{at}**: all **35,284 updates** '
        'have finished. The independent CPU audit reconciles all 28 source families, **984,399,690 '
        'student values**, all 445 optimizer states and exact native/private model equality. '
        'All 445 selected tensors changed; 16 unused tensors stayed exact. Native assessments and '
        'the existing code-first → human DPO → execution RL → full assessment pipeline remain intact.\n\n'
        'A new CPU job fixes the first original Python 3 negative per source and observes actual '
        'output counterexamples against already verified references. **Five new controls pass**, '
        'and a separate reader reconstructs 32 real observations. The complete 3,959-source job and '
        'separate machine-preference admission are still pending in this snapshot; no new pairs '
        'are inserted into the existing production lineage. Final fixed code-token NLL is **0.86855**; '
        '**no new generated-code or broad quality gain is claimed, and all ten final requirements '
        'remain incomplete.** [Completed main audit](reports/completed-phase-joint-v3-CPU-readback-v1/complete.json), '
        '[full review](reports/completed-main-execution-contrast-v1-review/review.json). '
        'Earlier entries are historical snapshots.\n\n'
    )
    for path in docs[:2]:
        first, rest = path.read_text().split('\n', 1)
        path.write_text(first + '\n\n' + paragraph + rest.lstrip('\n'))
    print(json.dumps(dict(review=str(OUT / 'review.json'), main_training_complete=True,
        current_code_first_updates=CE_updates, new_focused_tests_passed=5,
        new_execution_contrast_admission_complete=admission_complete,
        all_ten_final_requirements_incomplete=True, goal_achieved=False)), flush=True)


if __name__ == '__main__':
    main()
