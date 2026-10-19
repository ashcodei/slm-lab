"""Record actual stage controls, closed GPU training and all unfinished goals."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/visual-source-stage-and-first-GPU-repair-arm-v1-review'
PREVIOUS = ROOT / 'reports/joint-visual-strict-snapshot-recovery-v2-review/review.json'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic(path, value):
    path = Path(path)
    pending = path.with_suffix(path.suffix + '.pending')
    with pending.open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(path)


def main():
    from slm_lab.exact_waiting_process_retirement_v1 import terminal
    if OUT.exists():
        raise ValueError('Preserve every previous completed or failed review')
    tests = ROOT / 'reports/joint-visual-training-stage-v1-tests.log'
    if '16 passed' not in tests.read_text():
        raise ValueError('All16actual own-small-neural stage controls must pass')
    folders = {p.parent.resolve() for p in (ROOT / 'reports/joint-visual-training-stage-v1-pytest').rglob('0.json')
        if read(p).get('exact_next_source_Adam_weights_metrics_and_private_RNG') is True}
    if len(folders) != 1:
        raise ValueError('Exactly one actual complete four-replica stage proof set is required')
    proofs = [next(iter(folders)) / (str(r) + '.json') for r in range(4)]
    for rank, path in enumerate(proofs):
        value = read(path)
        if (value['rank'] != rank or not value['complete'] or value['actual_own_small_parameters'] != 70338
                or not value['one_rank_wrong_clock_pointer_decode_and_source_errors_rejected_everywhere_before_forward']
                or not value['singleton_word_with_three_real_empty_ranks']):
            raise ValueError('Every actual rank must preserve source sequencing and coordinated rejection')
    if len({read(p)['actual_shared_checkpoint_sha256'] for p in proofs}) != 1:
        raise ValueError('All four actual workers must read the same common durable payload')
    reader_path = ROOT / 'reports/first-GPU-repair-arm-complete-readback-CPU-v1/complete.json'
    reader = read(reader_path)
    if (not reader['complete'] or not reader['passed'] or reader['actual_optimizer_updates'] != 306
            or reader['actual_complete_presentations'] != 4851 or reader['actual_original_response_EOS_targets'] != 2287110
            or reader['actual_ordinary_Adam_leaves_at_exact_clock306'] != 561
            or reader['new_accuracy_result'] is not False or reader['newly_executed_neural_forwards'] != 0):
        raise ValueError('The complete original actual GPU production arm requires independent full readback')
    gpu = ROOT / 'reports/full-function-repair-comparison-GPU-v2'
    at = datetime.now(timezone.utc).isoformat()
    progress = read(gpu / 'progress.json')
    source_owners = [dict(pid=51248, ticks='53113593'), dict(pid=106808, ticks='54228332'),
        dict(pid=108345, ticks='54257301'), dict(pid=109189, ticks='54271186')]
    snapshot = dict(at=at, current_original_GPU_progress=progress,
        original_GPU_parent_live=not terminal(dict(pid=101066, ticks='54137054')),
        actual_GPU_rows=subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total',
            '--format=csv,noheader'], text=True),
        original_CPU_training_live=not terminal(dict(pid=18545, ticks='50128787')),
        original_interim_assessment_live=not terminal(dict(pid=25240, ticks='51635952')),
        full_visual_source_chain=[dict(owner=b, live=not terminal(b)) for b in source_owners],
        full_source_training_admission_exists=(ROOT / 'reports/full-multimodal-corrected-source-index-CPU-v1/source-training-admission.json').exists(),
        no_existing_neural_worker_signalled_reprioritized_restarted_or_changed=True,
        no_competing_GPU_job_context_or_lease=True)
    prior = read(PREVIOUS)
    requirements = deepcopy(prior['requirements'])
    requirements['text_and_images']['evidence'] = (
        'The new internal source-aware visual stage passes16own-small-neural controls. It binds the complete schedule, '
        'initializer/fixed/source hashes, exact objective and grouped Adam settings; validates actual sparse optimizer clocks '
        'against the completed source prefix before every update; coordinates source/decode errors before nested collectives; '
        'and uses strict-v2shared durable recovery. Four actual70338parameter CPU replicas preserve exact next-source weights, '
        'Adam, metrics and distinct private RNGs with a genuine one-source/three-empty-rank word tail. '
        'Wrong clocks, settings, mode, schedule and source/restore faults reject before forward/mutation. '
        'The original corrected-source delivery bridge adds global full-caption/whole-word metadata and strips only adapter bookkeeping. '
        'Its complete actual source integration, full-size four-GPU capacity, owned production controller, native export and quality '
        'remain pending. No new visual neural training or runtime admission is inferred. All previous source/pixel/word failures remain recorded and unwaived.')
    requirements['serious_full_model_training']['evidence'] = (
        'The first actual four-L40plain-repeat coding arm has completed all306updates/4851presentations/2287110original positive '
        'response/EOS targets. A separate CUDA-hidden source/state reader reconstructs every source fingerprint, all1224rank deliveries '
        'and actual BF16forward records, verifies the closed shared journal/manifest, all561finite Adam leaves at clock306 and all4distinct '
        'private RNGs, and retains the actual4096text/448photo/real-word cold-export proof. No new neural forward or accuracy result was created. '
        'All3original matched arms and complete baseline/final evaluations remain required; no candidate is promoted. '
        f'The original live progress snapshot is {progress.get("profile")} {progress.get("actual_complete_updates")}/306. '
        'The complete4447update public epoch and all17families remain; full visual/human/RL/final-capability milestones are unfinished.')
    requirements['human_feedback']['evidence'] = (
        'All85786original human pairs and full held populations remain in the recovered canonical human/preference scopes. '
        'The ten versioned successor queues are already restored; no live queue/controller or initializer was edited by this work. '
        'This coding CEexperiment is not human feedback, on-policy RL or evidence of a completed full human-alignment milestone.')
    for value in requirements.values():
        value['previous_evidence'] = str(PREVIOUS)
    if len(requirements) != 10 or any(r['status'] != 'incomplete_for_requested_final_model' for r in requirements.values()):
        raise ValueError('No requested final requirement may be marked complete from these controls')
    direct = ['src/slm_lab/joint_visual_source_delivery_v1.py', 'src/slm_lab/joint_visual_training_stage_v1.py',
        'tests/test_joint_visual_training_stage_v1.py', 'scripts/run_joint_visual_training_stage_tests_v1.py',
        'scripts/read_first_GPU_repair_arm_CPU_v1.py', 'src/slm_lab/joint_visual_objective_update_v1.py',
        'src/slm_lab/joint_visual_private_checkpoint_v2.py', 'src/slm_lab/joint_visual_durable_state_v2.py',
        'src/slm_lab/corrected_multimodal_batches_v1.py',
        'reports/full-function-repair-comparison-GPU-v2/train-plain_repeat/complete.json',
        'reports/full-function-repair-comparison-GPU-v2/capacity-complete.json',
        'reports/completed-public-epoch-and-visual-integration-v1-review/matched-metrics.md']
    files = {str(p): sha(p) for p in [Path(__file__), PREVIOUS, tests, reader_path, *proofs, *[ROOT / n for n in direct]]}
    OUT.mkdir()
    atomic(OUT / 'runtime-snapshot.json', snapshot)
    classification = 'progress: source-aware visual sequencing passes16actual-neural controls; the first complete four-L40coding arm has full independent source/Adam/state readback; all matched accuracy results and final milestones remain pending'
    atomic(OUT / 'complete.json', dict(complete=True, at=at, source_files_sha256=files,
        actual_visual_stage_controls=16, actual_four_own_small_replicas=True, actual_mechanics_parameters=70338,
        exact_next_source_Adam_weights_metrics_and_private_RNG=True,
        sparse_Adam_clocks_checked_against_actual_complete_prefix_before_each_update=True,
        source_decode_and_restore_failures_coordinated_before_kernels_or_mutation=True,
        first_actual_GPU_arm_complete=True, first_actual_GPU_arm_full_readback_passed=True,
        actual_GPU_arm_updates=306, actual_GPU_arm_complete_presentations=4851,
        actual_GPU_arm_positive_response_EOS_targets=2287110, actual_GPU_arm_Adam_leaves_at_clock306=561,
        current_original_GPU_progress=progress, new_full_size_visual_updates=0,
        full_visual_source_bridge_integration_GPU_capacity_owned_controller_and_quality_pending=True,
        last_matched_native_code_functional=1, last_matched_native_code_population=48,
        last_matched_native_code_efficiency=0, last_matched_ordinary_Python_functional=0,
        no_new_accuracy_result=True, all_original_human_RL_and_nine_final_endpoints_preserved=True,
        all_ten_final_requirements_incomplete=True, goal_achieved=False))
    files[str(OUT / 'complete.json')] = sha(OUT / 'complete.json')
    files[str(OUT / 'runtime-snapshot.json')] = sha(OUT / 'runtime-snapshot.json')
    atomic(OUT / 'review.json', dict(at=at, requirements=requirements, previous_full_scope_review=str(PREVIOUS),
        current_goal_turn_classification=classification, source_files_sha256=files, goal_achieved=False))
    metrics = (
        'The first four-L40coding arm has completed **306updates,4851complete source presentations and2287110response/EOStargets**. '
        'Independent CPUreadback rebuilds every source fingerprint and all1224rank delivery records, verifies all561ordinary Adam '
        'leaves at exactly306, checks the actual shared checkpoint/whole journal and four distinct private RNGs, and retains the '
        'original4096text/448photo/real-word cold-load proof. **The remaining two arms and all matched final assessments are still required.**\n\n'
        'The new source-aware visual sequencing stage passes **16actual own-small-neural tests**. Four actual CPUreplicas exercise '
        'native imageCE, full caption matching and whole-wordCTC/nativeCE; exact next-source durable recovery; sparse Adam clock '
        'verification before every update; source/decode failures; wrong-clock/settings/schedule/mode rejection; coordinated '
        'restore errors; and a genuine singleton word with three empty ranks. These controls use a70338parameter own graph. '
        'The original31strict checkpoint checks remain separate completed evidence.\n\n'
        '**No new accuracy result is established.** The last matched result remains **1/48functional native deliveries, '
        '0/48efficient,0/24ordinary Python**. Full actual visual source/bridge integration, four-GPUvisual capacity, an owned '
        'training/export controller and subsequent held-out quality remain pending. No GPU job was added or changed. '
        'All85786human pairs,17public families, full RL/human stages and nine final endpoints remain. '
        '**All ten requested final requirements remain incomplete.**\n')
    (OUT / 'metrics.md').write_text(metrics)
    audit_path = ROOT / 'reports/completion-audit.json'
    audit = read(audit_path)
    audit.update(at=at, updated_at=at, requirements=requirements, goal_achieved=False,
        latest_full_scope_review=str(OUT / 'review.json'), latest_actual_visual_source_stage_and_first_GPU_repair_arm=str(OUT / 'complete.json'),
        current_goal_turn_classification=classification)
    atomic(audit_path, audit)
    entry = (f'Visual stage controls and first completed coding arm, **{at}**: the first four-L40coding arm completes '
        '**306updates/4851presentations/2287110targets**, with independent full source/Adam/shared-state readback. '
        '**16new own-small-neural stage tests pass**, including four-worker exact next-source recovery, actual sparse Adam '
        'clock validation, coordinated source/decode/restore errors and empty word-tail ranks. Full visual data/bridge integration, '
        'actual GPUcapacity and owned training/export remain pending. The remaining coding arms and complete matched assessments '
        'are required. **No new accuracy gain is established; all ten final requirements remain incomplete.** '
        '[Evidence](reports/visual-source-stage-and-first-GPU-repair-arm-v1-review/metrics.md), '
        '[full review](reports/visual-source-stage-and-first-GPU-repair-arm-v1-review/review.json). Earlier entries are historical snapshots.\n\n')
    for name in ['README.md', 'RESEARCH.md']:
        path = ROOT / name
        path.write_text(entry + path.read_text())
    print(json.dumps(dict(review=str(OUT / 'review.json'), actual_neural_controls=16,
        completed_GPU_training_updates=306, new_accuracy_result=False, goal_achieved=False)))


if __name__ == '__main__':
    main()
