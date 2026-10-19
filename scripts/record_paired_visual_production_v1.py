"""Retain and independently reconcile new visual lifecycle/capacity evidence."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess

import torch

from slm_lab.code_residual_collective_checkpoint_v2 import fingerprint
from slm_lab.code_residual_lifecycle_v4 import checkpoint_layout
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.finalized_visual_replay_v1 import FinalizedVisualReplay
from slm_lab.paired_visual_lifecycle_v1 import EXPORT_FORMAT
from slm_lab.paired_visual_production_v1 import RUNTIME_FILES, PACKAGE, bind_visual_baselines

ROOT = Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(1)
    source = Path('/tmp/pytest-of-jovyan/pytest-111/test_actual_four_worker_visual0')
    output = ROOT / 'reports/paired-visual-production-v1-CPU'
    report = ROOT / 'reports/paired-visual-production-v1-readiness.json'
    if output.exists() or report.exists():
        raise ValueError('Preserve existing retained artifacts; inspect before recording again')
    log = ROOT / 'reports/paired-visual-production-v1-final-tests.log'
    if '5 passed' not in log.read_text() or 'FAILED' in log.read_text():
        raise ValueError('Final five focused checks required')
    ranks = [json.loads((source / f'rank-{i}.json').read_text()) for i in range(4)]
    for i, rank in enumerate(ranks):
        if (rank['rank'] != i or rank['model_counts']['total'] != 8304 or not rank['exact_explicit_resume_matches_reference']
                or rank['disposable_capacity']['actual_full_kernel_optimizer_updates'] != 1
                or len(rank['disposable_capacity']['probes']) != 16 or len(rank['disposable_capacity']['all_eight_visual_families_exercised']) != 8):
            raise ValueError('Actual four-worker lifecycle and all-family disposable probe evidence required')
    reference = torch.load(source / 'reference.pt', weights_only=True, map_location='cpu')
    complete = torch.load(source / 'lifecycle/resume.pt', weights_only=True, map_location='cpu')
    previous = torch.load(source / 'lifecycle/previous.pt', weights_only=True, map_location='cpu')
    inference = torch.load(source / 'lifecycle/inference.pt', weights_only=True, map_location='cpu')
    if (fingerprint(reference) != fingerprint(complete) or complete['steps'] != 16 or previous['steps'] != 15
            or complete['state_sha256'] != fingerprint({k: v for k, v in complete.items() if k != 'state_sha256'})
            or inference['format'] != EXPORT_FORMAT or fingerprint(inference['model']) != fingerprint(complete['model'])
            or any(k in inference for k in ['optimizer', 'worker_rng_states'])
            or inference['complete_training_checkpoint_sha256'] != file_hash(source / 'lifecycle/resume.pt')):
        raise ValueError('Complete independent reference/recovery/native-export readback differs')
    for moment in complete['optimizer']['state'].values():
        if float(moment['step']) != 16 or not all(bool(torch.isfinite(moment[k]).all()) for k in ['exp_avg', 'exp_avg_sq']):
            raise ValueError('Full finite vision Adam inventory required')
    ledger = checkpoint_layout(source / 'lifecycle', complete['identity']['sources']['run_plan_sha256'])
    if ledger['latest']['steps'] != 16 or ledger['previous']['steps'] != 15:
        raise ValueError('Actual bounded latest/previous checkpoint generations required')
    old_path = ROOT / 'reports/paired-visual-training-v1-readiness.json'
    old = json.loads(old_path.read_text())
    for path, digest in old['files_sha256'].items():
        if file_hash(path) != digest:
            raise ValueError('Prior paired kernel/source/admission evidence changed')
    native_reports = {n: ROOT / 'reports/extended-curriculum-v1' / (n + '.json') for n in ['vision', 'ocr', 'objects']}
    native = {n: json.loads(p.read_text()) for n, p in native_reports.items()}
    baseline_file = ROOT / 'reports/paired-visual-production-v1-existing-joint-baseline-interface.json'
    if baseline_file.exists():
        raise ValueError('Preserve existing actual-baseline interface receipt')
    atomic_json(baseline_file, dict(format='own-paired-visual-baseline-binding-v1', baseline_completed=True,
        baseline_checkpoint_sha256=native['vision']['checkpoint_sha256'], reports={n: str(p) for n, p in native_reports.items()},
        files_sha256={str(p): file_hash(p) for p in native_reports.values()},
        scope='Actual existing completed-joint baseline interface only, not a final selected-policy baseline or production plan.'))
    bound = bind_visual_baselines(dict(visual_evaluation_binding=str(baseline_file), visual_evaluation_binding_sha256=file_hash(baseline_file),
        base_checkpoint_sha256=native['vision']['checkpoint_sha256'], tokenizer_sha256=native['vision']['tokenizer_sha256']))
    for path, digest in bound.items():
        if file_hash(path) != digest:
            raise ValueError('Actual retained native baseline/runtime/data bytes differ')
    try:
        FinalizedVisualReplay(ROOT, None, None)
        source_gate = dict(final_source_opened=True)
    except ValueError as error:
        if 'Finalized complete30-family' not in str(error): raise
        source_gate = dict(final_source_opened=False, rejected_unfinalized_source=True, reason=str(error))
    def process(pid):
        p = Path('/proc') / str(pid); fields = (p / 'stat').read_text().split(') ', 1)[1].split()
        if fields[0] == 'Z': raise ValueError('Expected original actor is terminal')
        return dict(pid=pid, start_ticks=fields[19], parent_pid=int(fields[1]), argv=(p / 'cmdline').read_bytes().replace(b'\0', b' ').decode())
    actors = {role: process(pid) for role, pid in [('controller', 2754743), ('collector', 130149),
        ('paired_preparer', 2755925), ('full_source_preparer', 2756113), ('human_feedback_waiter', 130151)]}
    rows = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_gpu_memory', '--format=csv,noheader,nounits'],
        capture_output=True, text=True, check=True).stdout.splitlines()
    workers = [dict(**process(int(row.split(',')[0])), GPU_record=row) for row in rows]
    if len(workers) != 4 or any(process(w['parent_pid'])['parent_pid'] != 2754743 for w in workers):
        raise ValueError('Reinspect actual original four-GPU controller ownership')
    metrics = [json.loads(line) for line in (ROOT / 'reports/curriculum-disk-cleanup-resumption-v1/joint.log').read_text().splitlines()
        if line.startswith('{"step"')]
    shutil.copytree(source, output, ignore=shutil.ignore_patterns('rendezvous', '*.lock'))
    files = [PACKAGE / (n + '.py') for n in RUNTIME_FILES]
    files += [ROOT / 'tests/test_paired_visual_production_v1.py', Path(__file__).resolve(), old_path, log,
        ROOT / 'reports/paired-visual-production-v1-focused-tests.log', ROOT / 'reports/paired-visual-production-v1-release-tests.log', baseline_file]
    files += [p for p in output.rglob('*') if p.is_file()]
    bindings = {str(p): file_hash(p) for p in files}; bindings.update(bound)
    result = dict(format='own-paired-visual-production-readiness-v1', observed_at=datetime.now(timezone.utc).isoformat(),
        previous_goal_turn_classification=old['current_turn_classification'],
        current_turn_classification=dict(classification='progress', reason='Integrated actual vision kernel with bounded lifecycle/explicit recovery/native export, disposable full-layout capacity probes and final-source/base/baseline/lease-bound four-L40 controller. Five focused checks pass.'),
        implementation=dict(production_controller_implemented=True, capacity_mode_implemented=True, no_actual_ready_plan_created=True,
            final_original_all30_source_and_base_required=True, full_native_code_knowledge_and_visual_baselines_required=True,
            actual_full706M_capacity_proof_required_for_production=True, full_three_checkpoint_slots_and_native_export_funded_before_launch=True,
            shared_GPU_lease_and_original_actor_checks=True, automatic_candidate_promotion=False),
        mechanics=dict(new_focused_checks=5, actual_four_CPU_workers=ranks, model_parameters=8304,
            complete_reference_updates=16, first_stop_step=7, failed_save_step8_preserves_durable_step7=True,
            resumed_final_step16_equals_complete_independent_reference=True, native_export_weights_match=True,
            real_existing_joint_baseline_interface=dict(checkpoint_sha256=native['vision']['checkpoint_sha256'],
                caption_examples=len(native['vision']['samples']), OCR_examples=len(native['ocr']['results']), object_examples=len(native['objects']['results']),
                final_own_policy_baseline=False, new_generation_or_quality_score=False),
            initial_release_failure='Test restored deliberately corrupted export on rank0 before other ranks completed byte checks; added barrier before restoring. Failed log retained; final five checks pass.'),
        actual_final_source_gate=source_gate, actual_original_actors=actors, actual_original_GPU_workers=workers,
        actual_original_latest_metric=metrics[-1], actual_new_full706M_visual_updates=0, actual_full706M_capacity_run=False,
        capability_status='No new full-model capability result; existing0/4 paired diagnostic and weak code/domain/OCR results remain. No high-end performance demonstrated.',
        remaining_requirements=['Finish live original paired/all30 sources, exact final own-policy/source binding and genuine-human feedback continuation.',
            'Bind final-policy complete native baselines, create actual full-input plan and run real706M/four-L40 capacity before visual production.',
            'Complete actual visual and857M execution training; run unchanged native broad/coding/image assessments and demonstrate strong all-round capability.'],
        candidate_promoted=False, new_capability_gain_measured=False, goal_achieved=False, files_sha256=bindings)
    atomic_json(report, result)
    print(json.dumps(dict(report=str(report), new_checks=5, fixture_parameters=8304,
        original_stage_step=metrics[-1]['step'], real_final_source_opened=source_gate['final_source_opened'],
        actual_new_full706M_visual_updates=0, goal_achieved=False), indent=2))


if __name__ == '__main__': main()
