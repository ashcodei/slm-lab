"""Record strict snapshot correction and all original unachieved model requirements."""
import ast
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

from slm_lab.exact_waiting_process_retirement_v1 import terminal

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/joint-visual-strict-snapshot-recovery-v2-review'
PREVIOUS = ROOT / 'reports/joint-visual-private-durable-recovery-v1-review/review.json'


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


def proof_paths(marker):
    parents = {p.parent.resolve() for p in (ROOT / 'reports/joint-visual-recovery-v2-pytest').rglob('0.json')
        if read(p).get(marker) is True}
    if len(parents) != 1:
        raise ValueError('Exactly one complete actual four-worker proof set required for ' + marker)
    return [next(iter(parents)) / (str(rank) + '.json') for rank in range(4)]


def main():
    if OUT.exists():
        raise ValueError('Preserve every previous review and failed source version')
    tests = ROOT / 'reports/joint-visual-recovery-v2-tests.log'
    if '31 passed' not in tests.read_text():
        raise ValueError('All original20unchanged and11new strict snapshot controls must pass')
    gap_path = ROOT / 'reports/joint-visual-v1-invalid-snapshot-gap/complete.json'
    gap = read(gap_path)
    if (not gap['gap_confirmed'] or not gap['legacy_v1_published_nonfinite_selected_weight']
            or not gap['legacy_restore_rejected_before_neural_mutation']
            or any(sha(p) != h for p, h in gap['source_files_sha256'].items())):
        raise ValueError('The actual deliberately invalid small-fixture finding and immutable v1 source must remain explicit')
    for stem in ['joint_visual_private_checkpoint', 'joint_visual_durable_state']:
        def bodies(version):
            tree = ast.parse((ROOT / 'tests' / ('test_' + stem + '_v' + str(version) + '.py')).read_text())
            return [ast.dump(n) for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
        if bodies(1) != bodies(2):
            raise ValueError('No original test function, tolerance or assertion may change in the corrected version')
    private_paths = proof_paths('exact_next_cross_family_Adam_weights_and_all_CPU_RNG')
    durable_paths = proof_paths('exact_next_source_model_Adam_and_all_CPU_RNG')
    guard_paths = proof_paths('one_rank_nonfinite_live_weight_rejected_by_every_actual_peer')
    for paths in [private_paths, durable_paths, guard_paths]:
        for rank, path in enumerate(paths):
            value = read(path)
            if not value['complete'] or value['rank'] != rank:
                raise ValueError('Every actual rank must provide its complete claimed own-neural proof')
    if (len({read(p)['actual_private_RNG_sha256'] for p in private_paths}) != 4
            or len({read(p)['committed_checkpoint_sha256'] for p in durable_paths}) != 1):
        raise ValueError('Four private RNGs and one actual common durable tensor file required')
    at = datetime.now(timezone.utc).isoformat()
    gpu = ROOT / 'reports/full-function-repair-comparison-GPU-v2'
    progress = read(gpu / 'progress.json')
    snapshot = dict(at=at, actual_existing_GPU_training_progress=progress,
        actual_existing_GPU_parent_live=not terminal(dict(pid=101066, ticks='54137054')),
        actual_GPU_rows=subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total',
            '--format=csv,noheader'], text=True),
        original_CPU_training_live=not terminal(dict(pid=18545, ticks='50128787')),
        original_interim_assessment_live=not terminal(dict(pid=25240, ticks='51635952')),
        actual_visual_source_chain=[dict(owner=b, actual_live=not terminal(b)) for b in
            [dict(pid=51248, ticks='53113593'), dict(pid=106808, ticks='54228332'),
             dict(pid=108345, ticks='54257301'), dict(pid=109189, ticks='54271186')]],
        visual_source_training_admission_exists=(ROOT / 'reports/full-multimodal-corrected-source-index-CPU-v1/source-training-admission.json').exists(),
        no_existing_neural_worker_signalled_reprioritized_or_restarted=True,
        no_mutable_original_training_journal_child_registry_or_human_queue_bound=True)
    prior = read(PREVIOUS)
    requirements = deepcopy(prior['requirements'])
    requirements['text_and_images']['evidence'] = (
        'Strict visual snapshot/recovery v2 passes31own-small-neural tests: all original13private and7durable test functions/thresholds '
        'remain identical, plus11checks for invalid live weights/settings, valid recovery of damaged selected values, test fidelity '
        'and one-rank publication rejection. A deliberately invalid70338parameter CPU fixture proved v1could publish NaN weights '
        'that its reload then rejected; the source, bad fixture and finding are preserved. V2adds coordinated prepublication '
        'finite-weight and ordinary-Adam validation, with no wire-format/source-target or numerical-gradient change. '
        'Four actual CPU replicas preserve exact next-source model/Adam/private-RNG recovery and one shared tensor payload. '
        'These are not full-size GPUvisual capacity, an admitted training controller, or learned capability. '
        'Full4733step/302627view source checks and actual full-size visual runtime remain pending; old word/prefix failures stay unwaived.')
    requirements['serious_full_model_training']['evidence'] = (
        'The complete original4447update public epoch and all three actual four-L40repair capacity gates remain complete. '
        f'Existing full-scope coding-repair production snapshot: {progress.get("profile")} '
        f'{progress.get("actual_complete_updates")}/306; all three306update arms and all matched evaluations remain required. '
        'New recovery controls use explicit70338parameter CPU fixtures, with46362existing values selected; no new visual production '
        'updates or new full-size competing model job was started. Complete source admission, actual GPUvisual capacity and the full visual controller remain pending.')
    for value in requirements.values():
        value['previous_evidence'] = str(PREVIOUS)
    if len(requirements) != 10 or any(v['status'] != 'incomplete_for_requested_final_model' for v in requirements.values()):
        raise ValueError('No required final capability has been demonstrated by these mechanics controls')
    names = ['src/slm_lab/joint_visual_private_checkpoint_v2.py', 'src/slm_lab/joint_visual_durable_state_v2.py',
        'tests/test_joint_visual_private_checkpoint_v2.py', 'tests/test_joint_visual_durable_state_v2.py',
        'tests/test_joint_visual_snapshot_guards_v2.py', 'scripts/run_joint_visual_recovery_tests_v2.py',
        'reports/joint-visual-recovery-v2-exact-source-map.json',
        'reports/full-function-repair-comparison-GPU-v2/capacity-complete.json',
        'reports/joint-visual-private-checkpoint-v1-stdin-launch-failure/failure-record.json']
    files = {str(p): sha(p) for p in [Path(__file__), PREVIOUS, tests, gap_path,
        *private_paths, *durable_paths, *guard_paths, *[ROOT / name for name in names]]}
    OUT.mkdir()
    atomic(OUT / 'runtime-snapshot.json', snapshot)
    classification = 'progress:31strict visual recovery checks pass with all original20test bodies unchanged; an actual NaN-publication gap is preserved and corrected in v2; existing full-scope fourL40coding training advances without intervention'
    atomic(OUT / 'complete.json', dict(complete=True, at=at, source_files_sha256=files,
        actual_total_strict_recovery_checks=31, original20test_bodies_and_tolerances_identical=True,
        added_checks=11, actual_four_worker_private_and_durable_and_publication_proofs_rechecked=True,
        actual_v1_nonfinite_publication_gap_verified=True, strict_v2_snapshot_prepublication_checks_passed=True,
        bad_current_selected_values_can_be_recovered_from_a_valid_committed_file=True,
        actual_own_small_parameters=70338, actual_existing_selected_values=46362,
        full_size_visual_GPU_capacity_complete_source_admission_and_training_controller_pending=True,
        existing_GPU_production_progress=progress, new_visual_production_updates=0,
        no_new_model_quality_gain_established=True, all_original_human_RL_and_nine_final_endpoints_preserved=True,
        goal_achieved=False))
    files[str(OUT / 'complete.json')] = sha(OUT / 'complete.json')
    files[str(OUT / 'runtime-snapshot.json')] = sha(OUT / 'runtime-snapshot.json')
    atomic(OUT / 'review.json', dict(at=at, requirements=requirements, previous_full_scope_review=str(PREVIOUS),
        previous_goal_turn_classification='progress: complete modality-clocks/source-chain work and actual repair capacity advances',
        current_goal_turn_classification=classification, source_files_sha256=files, goal_achieved=False))
    text = (
        'Strict visual recovery v2 passes **31 checks**, including all original **13 private-state and seven durable-file tests unchanged**. '
        'Four actual own-small CPU replicas retain exact next-source weights, Adam, private RNGs and source cursors. '
        'One common tensor payload and external per-rank boundaries preserve recovery without copying the frozen language/actor/critic weights.\n\n'
        'An additional fault probe exposed a v1gap: an invalid live selected NaN weight could be published, although reload correctly rejected it before model mutation. '
        'The deliberately damaged70338parameter fixture, original source and bad checkpoint remain preserved. '
        '**V2 checks finite selected weights and valid finite Adam settings before publication**, coordinates one-rank failures, '
        'and still permits damaged current selected values to be repaired from a valid independently committed checkpoint. '
        'All original source/test functions, tolerances and source-target definitions remain unchanged.\n\n'
        'These are **CPU mechanics tests on a70338parameter own graph, with46362selected values**. '
        'They do not prove actual984Mvisual GPU capacity, complete data admission or learned quality. '
        'The full source chain, actual visual GPU capacity/recovery and complete training/controller integration remain pending.\n\n'
        f'Existing full-size four-L40coding-repair training snapshot: **{progress.get("profile")} {progress.get("actual_complete_updates")}/306**. '
        'All three original arms and all complete baseline/final source/code/broad assessments remain. '
        'No existing neural worker was signalled, reprioritized or restarted. '
        'Latest verified coding remains **1/48functional,0/48efficiency,0/24ordinaryPython**. '
        '**No new accuracy gain is claimed; all ten final-model requirements remain incomplete.**\n')
    (OUT / 'metrics.md').write_text(text)
    audit = read(ROOT / 'reports/completion-audit.json')
    audit.update(at=at, updated_at=at, requirements=requirements, latest_full_scope_review=str(OUT / 'review.json'),
        current_goal_turn_classification=classification, latest_actual_strict_visual_snapshot_recovery=str(OUT / 'complete.json'),
        goal_achieved=False)
    atomic(ROOT / 'reports/completion-audit.json', audit)
    paragraph = (f'Strict visual snapshot/recovery, **{at}**: **31checks pass**, retaining all original20test bodies/tolerances. '
        'A deliberately damaged own-small fixture exposed a v1NaN-publication gap; source and artifact are preserved. '
        'V2rejects invalid live selected values/settings before publication and retains exact four-replica private/durable recovery. '
        '**This is not full-size visual GPU admission or learned accuracy.** Existing four-L40coding-repair training snapshot: '
        f'**{progress.get("profile")} {progress.get("actual_complete_updates")}/306**; all three full arms and every final endpoint remain. '
        '**All ten requested final-model requirements remain incomplete.** '
        '[Corrected boundary and limits](reports/joint-visual-strict-snapshot-recovery-v2-review/metrics.md), '
        '[full scope](reports/joint-visual-strict-snapshot-recovery-v2-review/review.json). Earlier entries are historical snapshots.\n\n')
    for name in ['README.md', 'RESEARCH.md']:
        path = ROOT / name
        value = path.read_text()
        if value.startswith('# '):
            first, rest = value.split('\n', 1)
            path.write_text(first + '\n\n' + paragraph + rest.lstrip('\n'))
        else:
            path.write_text(paragraph + value)
    if any(sha(p) != digest for p, digest in files.items()):
        raise ValueError('All original and corrected frozen evidence bytes must remain exact')
    print(json.dumps(dict(review=str(OUT / 'review.json'), strict_CPU_checks=31,
        original_tests_unchanged=20, actual_existing_GPU_updates=progress.get('actual_complete_updates'), goal_achieved=False)))


if __name__ == '__main__':
    main()
