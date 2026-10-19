"""Record whole-source capacity planning and actual visual loop/export controls."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from record_visual_stage_and_first_repair_arm_v1 import read, sha, atomic
from slm_lab.exact_waiting_process_retirement_v1 import terminal

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/visual-complete-capacity-loop-and-native-export-v1-review'
PREVIOUS = ROOT / 'reports/visual-source-stage-and-first-GPU-repair-arm-v1-review/review.json'


def main():
    if OUT.exists():
        raise ValueError('Preserve all previous completed and failed research artifacts')
    logs = [('joint-visual-capacity-cases-v1-tests.log', '9 passed'),
        ('joint-visual-experiment-v1-tests.log', '6 passed'), ('joint-visual-native-export-v1-tests.log', '6 passed')]
    for name, expected in logs:
        if expected not in (ROOT / 'reports' / name).read_text():
            raise ValueError('Every source-selection, actual-neural-loop and native-export control must pass')
    cases = ROOT / 'reports/joint-visual-actual-capacity-cases-CPU-v1'
    full = read(cases / 'complete.json')
    plan = read(cases / 'actual-complete-source-capacity-plan.json')
    if (not full['complete'] or not full['passed'] or full['actual_original_epoch_source_views'] != 302627
            or full['actual_original_epoch_entries'] != 4733 or full['actual_selected_capacity_entries'] != 10
            or full['actual_selected_capacity_source_presentations'] != 510
            or full['training_admitted'] or full['actual_full_size_GPU_capacity_passed']
            or sha(cases / 'actual-complete-source-capacity-plan.json') != full['actual_plan_sha256']):
        raise ValueError('The actual full-source plan must preserve its true unadmitted source-only scope')
    folders = {p.parent.resolve() for p in (ROOT / 'reports/joint-visual-experiment-v1-pytest').rglob('0.json')
        if read(p).get('exact_next_cross_objective_weights_Adam_metrics_private_RNG') is True}
    if len(folders) != 1:
        raise ValueError('Exactly one complete actual four-worker experiment proof set is required')
    proofs = [next(iter(folders)) / (str(r) + '.json') for r in range(4)]
    for rank, path in enumerate(proofs):
        value = read(path)
        if (not value['complete'] or value['rank'] != rank or value['actual_own_small_parameters'] != 70338
                or value['actual_capacity_logical_updates'] != 2 or value['actual_capacity_physical_optimizer_calls'] != 3
                or not value['complete_actual_three_source_loop_and_all_rank_private_metrics']
                or not value['all_four_actual_private_RNGs_distinct']
                or not value['one_rank_preflight_failure_coordinated_before_any_forward']
                or not value['actual_initial_baseline_and_complete_final_cursor_committed']):
            raise ValueError('Every actual own-small rank must prove the complete source/recovery loop')
    if len({read(p)['actual_final_shared_checkpoint_sha256'] for p in proofs}) != 1:
        raise ValueError('All actual ranks must retain one identical common durable state')
    cold_files = {p.resolve() for p in (ROOT / 'reports/joint-visual-native-export-v1-pytest').rglob('model.pt')}
    if len(cold_files) != 2 or any(not p.with_name('tokenizer.json').exists() for p in cold_files):
        raise ValueError('Both actual FP32andBF16own-small complete native packages/tokenizers must exist')
    at = datetime.now(timezone.utc).isoformat()
    progress = read(ROOT / 'reports/full-function-repair-comparison-GPU-v2/progress.json')
    snapshots = dict(at=at, original_GPU_production_progress=progress,
        original_GPU_parent_live=not terminal(dict(pid=101066, ticks='54137054')),
        actual_GPU_rows=subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total',
            '--format=csv,noheader'], text=True),
        original_pixel_producer_progress=read(ROOT / 'reports/full-multimodal-pixel-validation-CPU-v1/progress.json'),
        original_public_code_readback_progress=read(ROOT / 'reports/full-public-code-shard-readback-CPU-v1/progress.json'),
        full_visual_source_chain=[dict(owner=b, live=not terminal(b)) for b in [dict(pid=51248, ticks='53113593'),
            dict(pid=106808, ticks='54228332'), dict(pid=108345, ticks='54257301'), dict(pid=109189, ticks='54271186')]],
        full_visual_data_admission_exists=(ROOT / 'reports/full-multimodal-corrected-source-index-CPU-v1/source-training-admission.json').exists(),
        no_existing_worker_changed_or_competing_GPU_job_context_or_lease=True)
    requirements = deepcopy(read(PREVIOUS)['requirements'])
    requirements['text_and_images']['evidence'] = (
        'A full-source selector reconstructs all4733schedule entries/all302627original view occurrences and91585whole word identities '
        'before selecting10complete capacity batches/510presentations spanning all9families,2952native tokens plus196image prefix, '
        '64global caption negatives, the actual50byte longest word and original singleton/3empty-rank tail. Only capacity cursor steps '
        'are renumbered. Nine selection controls pass; no pixels or GPUcapacity were inferred from this source-only work. '
        'The complete durable experiment loop passes6actual own-small-neural tests:4CPUreplicas preserve exact cross-objective '
        'next-source weights/Adam/metrics/private RNG, reject bad source/missing moment/nonfinite saved values and valid peer RNG '
        'before mutation, commit the original selected baseline and complete final cursor, retain all private rank metrics, and '
        'coordinate one-rank ownership/source preflight errors before any forward. Six native-export controls pass: complete native '
        'base/tokenizer cold-loads preserve every parameter and exact128token text/448pixel image/64x448word outputs inFP32andBF16; '
        'learned actor deltas, changed policy norms/critic and nonfinite factors cannot be silently discarded. These use the explicit '
        '70338parameter research fixture, not the984Mmodel. Full actual source-bridge integration, full-size four-GPUcapacity/cold4096text '
        'and an owned complete production/export/evaluation controller remain pending; no new visual training or quality gain is claimed.')
    requirements['serious_full_model_training']['evidence'] = (
        'The first actual four-L40coding arm remains independently verified complete:306updates/4851presentations/2287110response-EOS '
        'targets/all561Adam leaves at clock306, exact shared/private-state readback and actual native4096text/448photo/word cold proof. '
        f'The existing unchanged GPUstudy now reports {progress.get("profile")} {progress.get("actual_complete_updates")}/306. '
        'All3arms and all complete baseline/final assessments remain required. The new visual experiments are actual small CPUmechanics, '
        'and the complete source planning is data-only. No new visual GPUjob, new learned values or new accuracy result was introduced. '
        'The original17family public epoch, full85786human pairs, RLstages and all9final endpoints remain in scope.')
    for requirement in requirements.values():
        requirement['previous_evidence'] = str(PREVIOUS)
    if len(requirements) != 10 or any(r['status'] != 'incomplete_for_requested_final_model' for r in requirements.values()):
        raise ValueError('All ten requested final capabilities remain unachieved')
    names = ['src/slm_lab/joint_visual_capacity_cases_v1.py', 'src/slm_lab/joint_visual_experiment_v1.py',
        'src/slm_lab/joint_visual_native_export_v1.py', 'tests/test_joint_visual_capacity_cases_v1.py',
        'tests/test_joint_visual_experiment_v1.py', 'tests/test_joint_visual_native_export_v1.py',
        'scripts/prepare_joint_visual_capacity_cases_CPU_v1.py', 'scripts/run_joint_visual_experiment_tests_v1.py',
        'src/slm_lab/joint_visual_training_stage_v1.py', 'src/slm_lab/joint_visual_private_checkpoint_v2.py',
        'src/slm_lab/joint_visual_durable_state_v2.py', 'src/slm_lab/own_dense_standalone_v1.py',
        'reports/first-GPU-repair-arm-complete-readback-CPU-v1/complete.json',
        'scripts/record_visual_stage_and_first_repair_arm_v1.py']
    files = {str(p): sha(p) for p in [Path(__file__), PREVIOUS, cases / 'complete.json',
        cases / 'actual-complete-source-capacity-plan.json', *proofs,
        *[ROOT / 'reports' / n for n, _ in logs], *[ROOT / n for n in names],
        *cold_files, *[p.with_name('tokenizer.json') for p in cold_files]]}
    OUT.mkdir()
    atomic(OUT / 'runtime-snapshot.json', snapshots)
    classification = 'progress: all302627source occurrences yield actual10batch/510view worst-shape capacity plan;9selection6actual experiment-loop6native-cold-export checks pass; the original GPUrepair arm continues without intervention'
    atomic(OUT / 'complete.json', dict(complete=True, at=at, source_files_sha256=files,
        actual_new_controls=21, source_selection_controls=9, actual_own_small_experiment_loop_controls=6,
        actual_own_small_native_export_controls=6, actual_all_source_views_reconstructed=302627,
        actual_all_source_entries_reconstructed=4733, actual_capacity_entries=10, actual_capacity_source_presentations=510,
        actual_four_replica_cross_objective_recovery_and_complete_source_loop_proved=True,
        actual_FP32_BF16_native_packages_and_cold_outputs_proved=True,
        actual_mechanics_research_parameters=70338, original_GPU_production_progress=progress,
        original31strict_recovery_and16stage_controls_remain_completed_evidence=True,
        full_source_bridge_actual_GPUcapacity_owned_controller_and_full_assessment_pending=True,
        new_visual_production_updates=0, new_accuracy_result=False,
        all_original_human_RL_and_nine_final_endpoints_preserved=True,
        all_ten_final_requirements_incomplete=True, goal_achieved=False))
    files[str(OUT / 'complete.json')] = sha(OUT / 'complete.json')
    files[str(OUT / 'runtime-snapshot.json')] = sha(OUT / 'runtime-snapshot.json')
    atomic(OUT / 'review.json', dict(at=at, requirements=requirements, previous_full_scope_review=str(PREVIOUS),
        previous_goal_turn_classification='progress:16stage controls and independent first306update GPUarm readback complete',
        current_goal_turn_classification=classification, source_files_sha256=files, goal_achieved=False))
    metrics = (
        '**21new controls pass:**9whole-source capacity-selection checks,6actual own-small experiment-loop checks, '
        'and6native-export/cold-load checks. The previous31strict checkpoint and16source-stage controls remain separate evidence.\n\n'
        'All **302627original source views,4733steps and91585whole-word identities** are reconstructed before selecting '
        '**10complete original batches/510presentations** across all9families. They include2952native tokens plus196image tokens, '
        '64global matching examples, the actual longest50byte word and the original singleton/3empty-rank tail. '
        'Selected targets and occurrences stay intact; only local capacity step numbers change. **This is not GPUcapacity or data admission.**\n\n'
        'Four actual small CPUreplicas now run the complete experiment loop: committed initial baseline, exact cross-objective '
        'next-source weights/Adam/metrics/private-RNG replay, valid-peer-RNG and corrupted-state rejection, full final cursor '
        'and private rank metrics, and coordinated preflight failures before any forward. Two own-small complete native packages '
        'cold-load every saved value and preserve exact128token text/448pixel image/64x448word outputs inFP32andBF16. '
        'Export rejects any learned actor delta, changed policy normalization/critic or invalid factor rather than dropping it. '
        '**These use a70338parameter research fixture. Actual984Mfour-GPUcapacity,4096token cold parity and the owned controller remain pending.**\n\n'
        f'The existing coding study reports **{progress.get("profile")} {progress.get("actual_complete_updates")}/306**. '
        'Its first arm is complete with independent full source/Adam/state readback. The remaining arms and all matched final '
        'assessments are required. **No new accuracy gain is established.** The last matched code result remains1/48functional, '
        '0/48efficient and0/24ordinary Python. All17public families,85786human pairs, RLstages and9final endpoints remain. '
        '**All ten requested final requirements remain incomplete.**\n')
    (OUT / 'metrics.md').write_text(metrics)
    audit_path = ROOT / 'reports/completion-audit.json'
    audit = read(audit_path)
    audit.update(at=at, updated_at=at, requirements=requirements, goal_achieved=False,
        latest_full_scope_review=str(OUT / 'review.json'), latest_actual_visual_complete_capacity_loop_and_native_export=str(OUT / 'complete.json'),
        current_goal_turn_classification=classification)
    atomic(audit_path, audit)
    entry = (f'Complete visual source planning, experiment loop and native export, **{at}**: '
        '**21new controls pass**. All302627views/4733steps/91585whole-word identities yield10intact original capacity '
        'batches/510presentations covering all9families and actual worst shapes. Four actual own-small CPUreplicas prove '
        'cross-objective recovery and the complete durable source loop; two native packages preserve exact text/image/word '
        'cold outputs. **Actual984Mvisual GPUcapacity, full source/bridge integration and owned controller remain pending.** '
        f'The original GPUcoding study reports {progress.get("profile")} {progress.get("actual_complete_updates")}/306. '
        '**No new accuracy result is established; all ten final requirements remain incomplete.** '
        '[Evidence](reports/visual-complete-capacity-loop-and-native-export-v1-review/metrics.md), '
        '[full review](reports/visual-complete-capacity-loop-and-native-export-v1-review/review.json). Earlier entries are historical snapshots.\n\n')
    for name in ['README.md', 'RESEARCH.md']:
        path = ROOT / name
        path.write_text(entry + path.read_text())
    print(json.dumps(dict(review=str(OUT / 'review.json'), actual_controls=21,
        actual_source_views=302627, new_accuracy_result=False, goal_achieved=False)))


if __name__ == '__main__':
    main()
