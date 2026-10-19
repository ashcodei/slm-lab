"""Record actual connected visual controls and untouched complete predecessors."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from record_visual_stage_and_first_repair_arm_v1 import read, sha, atomic
from slm_lab.exact_waiting_process_retirement_v1 import terminal
from slm_lab.joint_visual_runtime_contract_v1 import ready, PIPELINE_QUEUE, FINALIZER_QUEUE, SOURCE

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/visual-connected-controller-and-aligned-continuation-v1-review'
PREVIOUS = ROOT / 'reports/visual-complete-capacity-loop-and-native-export-v1-review/review.json'


def main():
    if OUT.exists():
        raise ValueError('Preserve all completed and failed original artifacts')
    checks = [('joint-visual-cold-overlay-and-runtime-contract-v1-tests.log', '20 passed'),
        ('joint-visual-controller-v1-tests.log', '14 passed')]
    for name, expected in checks:
        if expected not in (ROOT / 'reports' / name).read_text():
            raise ValueError('All20cold/scope and14connected-controller controls must pass')
    folders = {p.parent.resolve() for p in (ROOT / 'reports/joint-visual-controller-v1-pytest').rglob('0.json')
        if read(p).get('format') == 'own-whole-nine-family-visual-runtime-capacity-v1'}
    if len(folders) != 1:
        raise ValueError('One complete actual four-worker own-small controller proof set is required')
    proofs = [next(iter(folders)) / (str(r) + '.json') for r in range(4)]
    for rank, path in enumerate(proofs):
        p = read(path)
        if (p['rank'] != rank or not p['complete'] or p['mode'] != 'mechanics'
                or p['actual_actor_parameters'] != 70338 or p['actual_native_parameters'] != 68098
                or p['actual_complete_capacity_entries'] != 10 or p['actual_capacity_source_presentations'] != 13
                or p['actual_selected_Adam_leaves'] != 31 or p['actual_physical_optimizer_calls'] != 11
                or p['actual_final_sparse_group_clocks'] != dict(vision=10, pointing=1, word=2)
                or p['full_size_GPU_capacity_passed'] is not False
                or not p['recovery']['exact_next_source_weights_Adam_metrics_and_private_RNG']
                or not p['recovery']['valid_foreign_rank_RNG_rejected_before_mutation']
                or not p['independent_native_cold']['actual_text_photo_logits_and_word_readout_bit_exact']):
            raise ValueError('Every actual CPUworker must prove the full small nine-family capacity flow without claiming GPUcapacity')
    if len({read(p)['final_actual_shared_checkpoint_sha256'] for p in proofs}) != 1:
        raise ValueError('All four actual replicas must verify the same complete common selected checkpoint')
    predecessors = dict(pipeline_owner=read(ROOT / PIPELINE_QUEUE)['owner'],
        finalizer_owner=read(ROOT / FINALIZER_QUEUE)['owner'], source_owner=read(ROOT / SOURCE / 'plan.json')['owner'])
    live = {k: not terminal(v) for k, v in predecessors.items()}
    admitted = ready(ROOT, **predecessors)
    if not all(live.values()) or admitted:
        raise ValueError('Reassess the actual predecessor state; this review expects the three original owners still live and unadmitted')
    at = datetime.now(timezone.utc).isoformat()
    progress = read(ROOT / 'reports/full-function-repair-comparison-GPU-v2/progress.json')
    snapshot = dict(at=at, exact_canonical_and_source_predecessors=predecessors, actual_predecessor_births_live=live,
        actual_runtime_gate_ready=False, planned_initializer='fixed completed machine-DPO aligned native student, after original full pipeline',
        current_original_GPU_progress=progress,
        original_GPU_parent_live=not terminal(dict(pid=101066, ticks='54137054')),
        actual_GPU_rows=subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total',
            '--format=csv,noheader'], text=True),
        original_pixel_producer_progress=read(ROOT / 'reports/full-multimodal-pixel-validation-CPU-v1/progress.json'),
        original_CPU_training_live=not terminal(dict(pid=18545, ticks='50128787')),
        original_interim_CPU_assessment_live=not terminal(dict(pid=25240, ticks='51635952')),
        no_neural_worker_changed_and_no_competing_GPU_context_lease_or_job=True,
        no_live_progress_journal_registry_or_mutable_human_queue_bound=True)
    requirements = deepcopy(read(PREVIOUS)['requirements'])
    requirements['text_and_images']['evidence'] = (
        'The connected capacity controller passes14checks, including actual70338parameter research/68098native CPUmodels inFP32/BF16 '
        'and4actual CPUreplicas running all9source-family objective paths over10complete own-small entries/13presentations/11physical '
        'optimizer calls. All31small selected Adam leaves end at exact sparse clocks vision10/pointing1/word2. Shared committed values, '
        'exact cross-objective weights/Adam/metrics/private-RNG replay, valid-peer-RNG rejection, genuine empty tails and independent '
        'native cold text/image/word outputs are proved. Six separate cold-overlay checks and14scope/dependency checks also pass. '
        'Capacity cold uses a separately loaded original native plus exactly verified selected values, retaining all fixed heads/language; '
        'it does not create a redundant full-model capacity export or substitute an overlay for final standalone inference. '
        'Full-size training admission still requires4matching real999706442research/984399690native GPUcapacity reports, all130Adam '
        'leaves, all10actual510source cases,4096text/448photo/64x448FP32word cold and source bindings. CPUmechanics cannot pass this gate. '
        'The real303k source/pixel chain, actual984Mcapacity and owned GPUdispatch/final export/full evaluation remain pending. '
        'No new visual GPUtraining or new model-quality result is claimed; all previous failed word/prefix gates remain unwaived.')
    requirements['human_feedback']['evidence'] = (
        'The planned visual continuation now starts from the fixed complete aligned machine-DPO standalone, after all canonical '
        'human/code/machine/native stages and original9endpoint scopes finish. It preserves that student’s learned language while '
        'updating only existing vision/pointing/word values. Its tested new dependency gate checks the exact live pipeline/finalizer/source '
        'births, all16961QA/3088AI2Dtimes3/600MBPP/4642human results, all3429held words with proper/blank/wrong conditions, all28actual '
        'native assessment receipts and complete visual data budgets. Actual original predecessor owners are currently live, so the '
        'gate returnsFalse and grants no GPUlease. All85786human pairs and original RLstages remain; no human/RLquality milestone is inferred.')
    requirements['serious_full_model_training']['evidence'] = (
        'The original first four-L40coding arm remains complete with independent306update/4851presentation/2287110target/561Adam '
        f'readback. Existing production snapshot: {progress.get("profile")} {progress.get("actual_complete_updates")}/306. '
        'All3matched arms and full baseline/final assessments remain required. New connected-controller and cold work uses explicit '
        'small CPUmodels or synthetic scope-control reports; it cannot count as actual984Mtraining or4GPUcapacity. '
        'The complete4733step/302627view future visual epoch and every original17family/human/RL/final-endpoint scope remain intact.')
    for r in requirements.values():
        r['previous_evidence'] = str(PREVIOUS)
    if len(requirements) != 10 or any(r['status'] != 'incomplete_for_requested_final_model' for r in requirements.values()):
        raise ValueError('All ten requested final requirements are still unachieved')
    names = ['src/slm_lab/joint_visual_cold_overlay_v1.py', 'src/slm_lab/joint_visual_runtime_contract_v1.py',
        'src/slm_lab/joint_visual_controller_v1.py', 'tests/test_joint_visual_cold_overlay_v1.py',
        'tests/test_joint_visual_runtime_contract_v1.py', 'tests/test_joint_visual_controller_v1.py',
        'scripts/run_joint_visual_controller_tests_v1.py', 'src/slm_lab/joint_visual_experiment_v1.py',
        'src/slm_lab/joint_visual_training_stage_v1.py', 'src/slm_lab/joint_visual_native_export_v1.py',
        'src/slm_lab/joint_visual_private_checkpoint_v2.py', 'src/slm_lab/joint_visual_durable_state_v2.py',
        'reports/joint-visual-actual-capacity-cases-CPU-v1/complete.json',
        'reports/first-GPU-repair-arm-complete-readback-CPU-v1/complete.json',
        'scripts/record_visual_stage_and_first_repair_arm_v1.py']
    files = {str(p): sha(p) for p in [Path(__file__), PREVIOUS, *proofs,
        *[ROOT / 'reports' / n for n, _ in checks], *[ROOT / n for n in names]]}
    OUT.mkdir()
    atomic(OUT / 'runtime-snapshot.json', snapshot)
    classification = 'progress:34new cold/dependency/connected-capacity checks pass;4actual small CPUreplicas run all9family objectives and independent native cold; the actual original aligned/source predecessors are verified live and no GPUwork is admitted'
    atomic(OUT / 'complete.json', dict(complete=True, at=at, source_files_sha256=files, actual_new_checks=34,
        actual_connected_controller_checks=14, actual_cold_overlay_checks=6, dependency_scope_controls=14,
        actual_four_own_small_research_replicas=True, actual_research_parameters=70338, actual_native_parameters=68098,
        actual_complete_mechanics_capacity_entries=10, actual_mechanics_source_presentations=13,
        actual_mechanics_physical_optimizer_calls=11, actual_mechanics_Adam_leaves=31,
        exact_cross_objective_private_recovery_and_independent_native_cold=True,
        small_mechanics_cannot_unlock_actual_full_size_training=True,
        actual_original_predecessors=predecessors, actual_original_predecessors_verified_live=True,
        actual_new_runtime_gate_ready=False, no_new_GPU_context_or_lease_or_job=True,
        planned_initializer_is_complete_aligned_native_after_all_original_scopes=True,
        current_original_GPU_progress=progress, full_size_GPUcapacity_owned_dispatch_final_export_and_assessment_pending=True,
        new_visual_production_updates=0, new_accuracy_result=False,
        all_original_human_RL_and_nine_final_endpoints_preserved=True,
        all_ten_final_requirements_incomplete=True, goal_achieved=False))
    files[str(OUT / 'complete.json')] = sha(OUT / 'complete.json')
    files[str(OUT / 'runtime-snapshot.json')] = sha(OUT / 'runtime-snapshot.json')
    atomic(OUT / 'review.json', dict(at=at, requirements=requirements, previous_full_scope_review=str(PREVIOUS),
        previous_goal_turn_classification='progress:21checks/full302627source capacity plan/experiment loop/native export complete',
        current_goal_turn_classification=classification, source_files_sha256=files, goal_achieved=False))
    metrics = (
        '**34new checks pass:**14connected-controller checks,6independent-native cold checks and14dependency/scope controls. '
        'Four actual CPUreplicas run all9visual objective paths over10own-small entries/13presentations, verify all31small Adam '
        'leaves at sparse clocks vision10/pointing1/word2, recover the exact next cross-objective update/private RNG, reject a valid '
        'foreign-rank RNG and preserve exact independently loaded native text/image/word outputs. '
        '**Research/native counts are70338/68098; these are not actual984Mfour-GPUcapacity results.**\n\n'
        'Capacity cold can reload the original complete native and apply only verified current vision/pointing/word values, '
        'preserving every other value. This avoids a redundant full-model capacity file. **Final inference still requires its own '
        'complete standalone export and cold proof.** The training gate requires all4real full-size GPUreports,130Adam leaves, '
        'all10original capacity cases/510presentations and4096text/448photo/64x448FP32word cold parity.\n\n'
        'The intended visual initializer is the **completed aligned native student after the existing full pipeline**, preserving '
        'its language/human-feedback learning. The original pipeline/finalizer/full-source owners are verified live now; the new '
        'gate therefore returnsFalse and acquires no GPUlease. Full source admission, actual GPUcapacity, owned GPUdispatch, '
        'final export and complete candidate evaluation remain pending.\n\n'
        f'The existing four-L40coding study reports **{progress.get("profile")} {progress.get("actual_complete_updates")}/306**. '
        'The first arm remains complete with independent source/Adam/state readback. All3arms and matched assessments are required. '
        '**No new accuracy gain is established.** The last matched code result remains1/48functional,0/48efficient and0/24ordinary '
        'Python. All17public families,85786human pairs, full RLstages and9final endpoints remain. '
        '**All ten requested final requirements remain incomplete.**\n')
    (OUT / 'metrics.md').write_text(metrics)
    audit_path = ROOT / 'reports/completion-audit.json'
    audit = read(audit_path)
    audit.update(at=at, updated_at=at, requirements=requirements, goal_achieved=False,
        latest_full_scope_review=str(OUT / 'review.json'), latest_actual_visual_connected_controller=str(OUT / 'complete.json'),
        current_goal_turn_classification=classification)
    atomic(audit_path, audit)
    entry = (f'Connected visual controller and aligned continuation gates, **{at}**: **34new checks pass**. '
        'Four actual small CPUreplicas run all9visual objective paths, complete private recovery and independent native cold. '
        'Full training still requires4real984MGPUcapacity reports and all original510capacity presentations. The planned initializer '
        'is the completed aligned native after the intact canonical pipeline; actual original predecessors are verified live, '
        'so the new gate remains closed. **Actual visual GPUcapacity, owned dispatch/export/evaluation and quality remain pending.** '
        f'The existing GPUcoding study reports {progress.get("profile")} {progress.get("actual_complete_updates")}/306. '
        '**No new accuracy result is established; all ten final requirements remain incomplete.** '
        '[Evidence](reports/visual-connected-controller-and-aligned-continuation-v1-review/metrics.md), '
        '[full review](reports/visual-connected-controller-and-aligned-continuation-v1-review/review.json). Earlier entries are historical snapshots.\n\n')
    for name in ['README.md', 'RESEARCH.md']:
        path = ROOT / name
        path.write_text(entry + path.read_text())
    print(json.dumps(dict(review=str(OUT / 'review.json'), actual_new_checks=34,
        actual_runtime_ready=False, new_accuracy_result=False, goal_achieved=False)))


if __name__ == '__main__':
    main()
