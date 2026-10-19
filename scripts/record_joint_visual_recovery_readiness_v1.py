"""Review selected visual recovery controls alongside unchanged full goal scope."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

from slm_lab.exact_waiting_process_retirement_v1 import terminal

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/joint-visual-private-durable-recovery-v1-review'
PREVIOUS = ROOT / 'reports/visual-modality-clock-and-full-source-chain-v1-review/review.json'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic(path, value):
    pending = path.with_suffix(path.suffix + '.pending')
    with pending.open('x') as stream:
        stream.write(json.dumps(value, indent=2) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(path)


def proof_files(folder):
    candidates = {p.parent.resolve() for p in folder.rglob('0.json')
        if read(p).get('actual_small_own_parameters') == 70338}
    if len(candidates) != 1:
        raise ValueError('One complete actual four-own-replica proof folder required')
    root = next(iter(candidates))
    return [root / (str(rank) + '.json') for rank in range(4)]


def main():
    if OUT.exists():
        raise ValueError('Preserve every original review')
    at = datetime.now(timezone.utc).isoformat()
    private_log = ROOT / 'reports/joint-visual-private-checkpoint-v1-file-launch-tests.log'
    durable_log = ROOT / 'reports/joint-visual-durable-state-v1-tests.log'
    if '13 passed' not in private_log.read_text() or '7 passed' not in durable_log.read_text():
        raise ValueError('All actual selected private and filesystem recovery controls must pass')
    private_paths = proof_files(ROOT / 'reports/joint-visual-private-checkpoint-v1-file-launch-pytest')
    durable_paths = proof_files(ROOT / 'reports/joint-visual-durable-state-v1-pytest')
    private, durable = [read(p) for p in private_paths], [read(p) for p in durable_paths]
    for rank in range(4):
        if (not private[rank]['complete'] or private[rank]['rank'] != rank
                or private[rank]['actual_saved_selected_values'] != 46362
                or not private[rank]['exact_next_cross_family_Adam_weights_and_all_CPU_RNG']
                or not private[rank]['two_one_rank_corruptions_rejected_everywhere_before_mutation']
                or not durable[rank]['complete'] or durable[rank]['rank'] != rank
                or durable[rank]['common_selected_tensor_values_stored_once'] != 46362
                or not durable[rank]['exact_next_source_model_Adam_and_all_CPU_RNG']
                or not durable[rank]['root_only_IO_failure_rejected_everywhere_before_any_mutation']
                or not durable[rank]['old_complete_generation_preserved_on_publication_failure']):
            raise ValueError('Every actual small-own-neural private/durable control must have its full claimed scope')
    if (len({p['actual_private_RNG_sha256'] for p in private}) != 4
            or len({p['committed_checkpoint_sha256'] for p in durable}) != 1):
        raise ValueError('Four actual distinct private RNG boundaries and one shared tensor file required')
    failed = ROOT / 'reports/joint-visual-private-checkpoint-v1-stdin-launch-failure/failure-record.json'
    if any(sha(path) != digest for path, digest in read(failed)['source_files_sha256'].items()):
        raise ValueError('The failed stdin launch and unchanged original source/tests must remain preserved')
    gpu = ROOT / 'reports/full-function-repair-comparison-GPU-v2'
    capacity = read(gpu / 'capacity-complete.json')
    if (not capacity['complete'] or not capacity['all_actual_recovery_and_cold_gates_passed']
            or capacity['all_profiles'] != ['plain_repeat', 'semantic_repair', 'weighted_semantic_repair']):
        raise ValueError('All original full actual fourGPUrepair capacity profiles must have passed')
    progress = read(gpu / 'progress.json')
    snapshot = dict(at=at, actual_GPU_training_progress=progress,
        actual_GPU_capacity_complete=capacity,
        actual_GPU_rows=subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total',
            '--format=csv,noheader'], text=True), original_neural_parent=dict(pid=101066, ticks='54137054'),
        original_neural_parent_live=not terminal(dict(pid=101066, ticks='54137054')),
        original_CPU_training_live=not terminal(dict(pid=18545, ticks='50128787')),
        original_interim_assessment_live=not terminal(dict(pid=25240, ticks='51635952')),
        source_chain=[dict(owner=birth, actual_live=not terminal(birth)) for birth in
            [dict(pid=51248, ticks='53113593'), dict(pid=106808, ticks='54228332'),
             dict(pid=108345, ticks='54257301'), dict(pid=109189, ticks='54271186')]],
        visual_source_training_admitted=(ROOT / 'reports/full-multimodal-corrected-source-index-CPU-v1/source-training-admission.json').exists(),
        disk_free_bytes=__import__('shutil').disk_usage(ROOT).free,
        no_existing_neural_worker_signalled_reprioritized_or_restarted=True,
        no_mutable_training_journal_registry_or_human_queue_source_bound=True)
    prior = read(PREVIOUS)
    requirements = deepcopy(prior['requirements'])
    requirements['text_and_images']['evidence'] = (
        'The new selected-only visual private recovery format passes13actual own-small-neural controls. '
        'It stores only existing vision/pointing/word weights and sparse group Adam states, binds the complete source prefix and '
        'each external private rank RNG boundary, and validates all peers before mutation. '
        'A filesystem wrapper passes7morecontrols: one common selected tensor/moment payload plus all private rank RNG/metrics, '
        'two durable generations, exact actual next-source recovery, old mmap survival, root receipt/pointer publication faults, '
        'corrupt bytes and one-rank read failures. Four70338parameter CPU replicas exercise these controls;46362existing values are selected. '
        'The original stdin launcher failure is preserved; a guarded importable-file runner uses identical source/tests/tolerances. '
        'This is not actual984M/GPUvisual capacity or a learned quality gain. Full source checks, training stage/controller and '
        'actual full-size GPU visual capacity/recovery remain pending. Old word/prefix numerical failures remain unwaived.')
    requirements['serious_full_model_training']['evidence'] = (
        'All three actual fourL40coding-repair capacity/recovery/cold-output profiles passed for the complete999706442parameter actor '
        'and all561selected Adam leaves. Production training now runs the original306updates per each of the three matched arms, '
        f'with current actual progress in the immutable snapshot: {progress.get("profile")}/{progress.get("actual_complete_updates")}. '
        'Every original source and all baseline/final code/source/broad assessments remain required. '
        'The new visual recovery tests use explicit70338parameter CPU graphs and add no production visual updates; '
        'the full4733step/302627view data and runtime admission are not inferred from them.')
    requirements['gpu_runtime']['evidence'] = (
        'The exact existing GPU owner101066has completed all three actual full-size fourL40capacity gates and runs production repair training. '
        'Existing CPU training, interim assessment and full source readers continue. '
        'New tiny recovery tests are CUDA-hidden, nice19, one thread and idle-scheduled. '
        'They create no competing GPU context/lease and never signal, reprioritize or restart existing neural workers. '
        'Current utilization and source handles are captured in the runtime snapshot; no accuracy claim follows from capacity success.')
    for value in requirements.values():
        value['previous_evidence'] = str(PREVIOUS)
    if len(requirements) != 10 or any(v['status'] != 'incomplete_for_requested_final_model' for v in requirements.values()):
        raise ValueError('All original ten final requirements remain incomplete')
    names = ['src/slm_lab/joint_visual_private_checkpoint_v1.py', 'tests/test_joint_visual_private_checkpoint_v1.py',
        'scripts/run_joint_visual_private_checkpoint_tests_v1.py', 'src/slm_lab/joint_visual_durable_state_v1.py',
        'tests/test_joint_visual_durable_state_v1.py', 'scripts/run_joint_visual_durable_state_tests_v1.py',
        'reports/full-function-repair-comparison-GPU-v2/plan.json',
        'reports/full-function-repair-comparison-GPU-v2/capacity-complete.json',
        'reports/visual-modality-clock-and-full-source-chain-v1-review/complete.json']
    paths = [Path(__file__), PREVIOUS, private_log, durable_log, failed, *private_paths, *durable_paths,
        *[ROOT / name for name in names]]
    files = {str(p): sha(p) for p in paths}
    OUT.mkdir()
    atomic(OUT / 'runtime-snapshot.json', snapshot)
    classification = 'progress: visual selected private13and durable7own-neural checks pass, including four actual CPU replicas and filesystem faults; all three actual fourL40repair capacity profiles passed and full306update/arm production training advances'
    atomic(OUT / 'complete.json', dict(complete=True, at=at, source_files_sha256=files,
        actual_new_private_recovery_checks=13, actual_new_durable_recovery_checks=7,
        actual_own_small_parameters=70338, actual_existing_selected_values=46362,
        every_four_rank_private_and_durable_receipt_verified=True,
        original_stdin_bootstrap_failure_preserved_without_changing_source_or_tests=True,
        full_model_actual_visual_GPU_capacity_and_complete_visual_training_controller_pending=True,
        current_GPU_repair_progress=progress, all_three_full_size_GPU_repair_capacity_profiles_passed=True,
        new_visual_production_updates=0, new_model_quality_gain_not_established=True,
        all_original_human_RL_and_nine_final_endpoint_scopes_preserved=True, goal_achieved=False))
    files[str(OUT / 'complete.json')] = sha(OUT / 'complete.json')
    files[str(OUT / 'runtime-snapshot.json')] = sha(OUT / 'runtime-snapshot.json')
    atomic(OUT / 'review.json', dict(at=at, requirements=requirements, previous_full_scope_review=str(PREVIOUS),
        previous_goal_turn_classification='progress: measured all445modality Adam clocks and launched the full visual source check chain',
        current_goal_turn_classification=classification, source_files_sha256=files, goal_achieved=False))
    summary = (
        'The visual recovery primitives now pass **13 private-state and seven durable-filesystem controls**. '
        'Actual four-worker CPU tests recover the next native/photo/word update, weights, Adam and every private RNG exactly, '
        'including a real one-source tail with three ranks performing no forward. '
        'Wrong sparse clocks, sources, future schedules, committed cursors and valid foreign RNG states are rejected before mutation.\n\n'
        'The durable representation stores existing selected weights and Adam moments once, alongside all private rank RNG/metrics. '
        'Two generations and an external receipt/LATEST pointer preserve the preceding complete state when receipt or pointer publication fails. '
        'Tests cover actual root-only disk faults, changed committed bytes, one-rank read failures, old mmap survival and exact subsequent neural updates. '
        'The fixture has **70,338 own parameters and 46,362 selected values**. These are mechanics checks; '
        '**actual full-size visual GPU capacity and the complete training/controller integration remain pending.**\n\n'
        'The initial four-worker test launch failed before entering its worker because multiprocessing could not import stdin. '
        'Its source, tests and log are preserved. An importable guarded runner passes every original test with no tolerance or source change.\n\n'
        'All three existing coding-repair profiles have passed actual four-L40 capacity/recovery/cold-output gates. '
        f'Production training snapshot: **{progress.get("profile")} {progress.get("actual_complete_updates")}/306**. '
        'Every original source budget and all matched endpoints remain. The larger visual source chain is still separate; no visual training is started by these tests.\n\n'
        'Latest verified coding remains **1/48 functional,0/48 efficiency and0/24 ordinary Python**. '
        'No new free-generation improvement is established. All ten requested final-model requirements remain incomplete.\n')
    (OUT / 'metrics.md').write_text(summary)
    audit = read(ROOT / 'reports/completion-audit.json')
    audit.update(at=at, updated_at=at, requirements=requirements, latest_full_scope_review=str(OUT / 'review.json'),
        current_goal_turn_classification=classification, latest_actual_joint_visual_private_durable_recovery=str(OUT / 'complete.json'),
        goal_achieved=False)
    atomic(ROOT / 'reports/completion-audit.json', audit)
    paragraph = (f'Visual private/durable recovery readiness, **{at}**: **13private-state and7filesystem checks pass**, '
        'including four actual own-small CPU replicas, exact next-source weights/Adam/private-RNG recovery and coordinated root I/O failures. '
        'Existing selected tensors/moments are stored once; frozen language/actor/critic values are not copied. '
        '**This is not full-size visual GPU admission or learned quality.** All three existing four-L40 coding-repair capacity profiles passed; '
        f'production snapshot: **{progress.get("profile")} {progress.get("actual_complete_updates")}/306**. '
        'The full visual source chain and training/controller integration remain pending. '
        '**No new accuracy gain is claimed; all ten final requirements remain incomplete.** '
        '[Controls and limits](reports/joint-visual-private-durable-recovery-v1-review/metrics.md), '
        '[full scope](reports/joint-visual-private-durable-recovery-v1-review/review.json). Earlier entries are historical snapshots.\n\n')
    for name in ['README.md', 'RESEARCH.md']:
        path = ROOT / name
        value = path.read_text()
        if value.startswith('# '):
            first, rest = value.split('\n', 1)
            path.write_text(first + '\n\n' + paragraph + rest.lstrip('\n'))
        else:
            path.write_text(paragraph + value)
    if any(sha(p) != digest for p, digest in files.items()):
        raise ValueError('Every frozen new and historical evidence byte must remain exact')
    print(json.dumps(dict(review=str(OUT / 'review.json'), new_own_neural_checks=20,
        actual_GPU_training=progress.get('actual_complete_updates'), new_visual_updates=0, goal_achieved=False)))


if __name__ == '__main__':
    main()
