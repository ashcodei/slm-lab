"""Matched language-control and joint visual-QA learning from our own snapshot."""
from datetime import datetime, timezone
import json
from pathlib import Path
import os
import shutil
import subprocess
import sys

import torch
from slm_lab.clevr_evaluation import compare_reports
from slm_lab.completed_state_cleanup import retire
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.evaluation_provenance import capture

ROOT = Path(__file__).resolve().parents[1]
STEPS = 1250


def main():
    torch.set_num_threads(1)
    reports = ROOT / 'reports/clevr-learning-probe-v1'
    checkpoints = ROOT / 'checkpoints/clevr-learning-probe-v1'
    if reports.exists() or checkpoints.exists(): raise ValueError('Existing probe requires inspected resume; never overwrite/restart')
    reports.mkdir(); checkpoints.mkdir()
    baseline_path = ROOT / 'reports/clevr-foundation-baseline-v1/visual-reasoning.json'
    baseline = json.loads(baseline_path.read_text())
    evaluator_files = [ROOT / path for path in baseline['evaluation_provenance']['files_sha256']]
    if baseline['evaluation_provenance'] != capture(evaluator_files): raise ValueError('Current evaluator differs from completed baseline')
    files = [*evaluator_files, Path(__file__), ROOT / 'src/slm_lab/completed_state_cleanup.py', ROOT / 'scripts/audit_clevr_inputs.py']
    loaded = capture(files)
    base = ROOT / 'checkpoints/clevr-foundation-baseline-v1/model.pt'
    if file_hash(base) != baseline['checkpoint_sha256']: raise ValueError('Current own foundation base differs from assessment')
    data = ROOT / 'data/clevr-reasoning-v1'; audit_path = ROOT / 'reports/clevr-original-input-audit-v1.json'
    audit = json.loads(audit_path.read_text()); manifest = json.loads((data / 'manifest.json').read_text())
    if (audit.get('passed') is not True or audit['manifest_sha256'] != file_hash(data / 'manifest.json')
            or audit['audit_source_sha256'] != file_hash(ROOT / 'scripts/audit_clevr_inputs.py')
            or audit['splits'] != manifest['splits'] or audit['files_sha256'] != manifest['files_sha256']):
        raise ValueError('Current visual data differs from original-source input audit')
    for relative, digest in audit['source_files_sha256'].items():
        if file_hash(ROOT / relative) != digest: raise ValueError('Audited model/data source changed')
    for relative, digest in manifest['files_sha256'].items():
        if file_hash(data / relative) != digest: raise ValueError('Finalized visual questions/scenes changed')
    plan = {'created_at': datetime.now(timezone.utc).isoformat(), 'base_checkpoint_sha256': baseline['checkpoint_sha256'],
            'baseline_assessment_sha256': file_hash(baseline_path), 'input_audit_sha256': file_hash(audit_path),
            'manifest_sha256': audit['manifest_sha256'], 'training_data_sha256': manifest['files_sha256']['train.jsonl'],
            'validation_data_sha256': manifest['files_sha256']['valid.jsonl'], 'available_training_questions': 40000,
            'steps_per_branch': STEPS, 'record_exposures_per_branch': STEPS * 8, 'complete_train_pass_fraction': .25,
            'batch_size': 2, 'gradient_accumulation': 4, 'world_size': 1, 'seed': 42, 'learning_rate': 2e-5,
            'image_size': 448, 'device': 'cuda:2', 'branches': {'language-control': 'language', 'joint': 'all'},
            'external_model_teacher_used': False, 'goal_achieved': False,
            'scope': 'Matched exploratory10000-example native-QA budgets, not full curriculum completion or policy promotion.',
            'source_provenance': loaded}
    atomic_json(reports / 'plan.json', plan)

    def status(message, **values):
        atomic_json(reports / 'status.json', {'status': message, 'updated_at': datetime.now(timezone.utc).isoformat(),
                                             'goal_achieved': False, **values})

    def run(arguments):
        if capture(files) != loaded: raise ValueError('Loaded learning/evaluation implementation changed')
        subprocess.run(arguments, check=True, cwd=ROOT,
                       env={**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'TOKENIZERS_PARALLELISM': 'false'})

    try:
        assessments = {}; stages = {}
        for name, trainable in plan['branches'].items():
            if shutil.disk_usage(ROOT).free < 30 * 1024**3: raise ValueError('Insufficient disk for bounded branch completion; inspect before starting')
            directory = checkpoints / name
            status('training matched original visual-question branch', branch=name, trainable=trainable, steps=STEPS)
            run([sys.executable, '-u', '-m', 'slm_lab.training', 'sft', '--checkpoint', str(base),
                 '--tokenizer', str(base.parent / 'tokenizer.json'), '--data', str(data / 'train.jsonl'), '--output', str(directory),
                 '--trainable', trainable, '--steps', str(STEPS), '--batch-size', '2', '--gradient-accumulation', '4',
                 '--learning-rate', '2e-5', '--seed', '42', '--image-size', '448', '--device', 'cuda:2', '--threads', '1',
                 '--gradient-checkpointing', '--checkpoint-every', '0', '--log-every', '25'])
            training = json.loads((directory / 'training.json').read_text())
            if (training['steps'] != STEPS or training['planned_steps'] != STEPS or training['stage'] != 'sft'
                    or training['world_size'] != 1 or training['trainable'] != trainable or training['image_size'] != 448
                    or training['data_sha256'] != plan['training_data_sha256'] or training['truncated_sequences'] != 0
                    or [item['step'] for item in training['metrics']] != list(range(1, STEPS + 1))):
                raise ValueError('Matched branch training did not complete its declared original-question budget')
            if name == 'language-control':
                original = torch.load(base, map_location='cpu', weights_only=True, mmap=True)
                trained = torch.load(directory / 'model.pt', map_location='cpu', weights_only=True, mmap=True)
                vision = [key for key in original['model'] if key.startswith('vision.')]
                if not vision or any(not torch.equal(original['model'][key], trained['model'][key]) for key in vision):
                    raise ValueError('Frozen vision control changed its visual tensors')
                atomic_json(reports / 'control-vision-preservation.json', {'exact_frozen_vision_tensors': len(vision), 'goal_achieved': False})
                del original, trained
            # Retire only this completed branch's exact redundant optimizer,
            # retaining every model tensor/report before its assessment.
            retire(directory, reports / (name + '-completed-state-cleanup.json'))
            stages[name] = {'training_report_sha256': file_hash(directory / 'training.json'),
                            'model_sha256': file_hash(directory / 'model.pt'),
                            'actual_target_exposures': sum(item['global_target_tokens'] for item in training['metrics']),
                            'record_exposures': STEPS * 8}
            status('assessing matched original visual-question branch', branch=name)
            path = reports / (name + '-visual-reasoning.json')
            run([sys.executable, '-u', '-m', 'slm_lab.clevr_evaluation', '--checkpoint', str(directory / 'model.pt'),
                 '--data', str(data / 'valid.jsonl'), '--output', str(path), '--device', 'cuda:2'])
            assessments[name] = json.loads(path.read_text())
            compare_reports(baseline, assessments[name])
        control = assessments['language-control']; joint = assessments['joint']
        retention = compare_reports(control, joint)
        first = control['overall']; second = joint['overall']
        comparison = {'base': baseline['overall'], 'language_control': first, 'joint': second, 'stages': stages,
                      'joint_vs_control_accuracy_change': second['correct_image_exact_accuracy'] - first['correct_image_exact_accuracy'],
                      'joint_correct_minus_blank_match': second['correct_image_exact_accuracy'] - second['blank_original_target_match'],
                      'joint_correct_minus_wrong_match': second['correct_image_exact_accuracy'] - second['wrong_image_original_target_match'],
                      'matched_control_retention_guard': retention, 'policy_promoted': False, 'goal_achieved': False,
                      'scope': 'Matched reused synthetic-question development probe; native image/format signal, not real-image, broad-code or high-end capability.',
                      'limitations': ['Only10000 of40000 shuffled training questions per branch; no complete CLEVR/CoGenT claim.',
                                      'No automatic policy selection; all broader production capability/retention assessments remain required.',
                                      'Source baseline is interim language pretraining, before later instruction/vision curriculum.']}
        atomic_json(reports / 'comparison.json', comparison)
        status('matched original visual reasoning learning probe completed; broader capability unverified')
    except Exception as error:
        status('matched visual learning probe failed; inspect without restarting live work', error=str(error))
        raise


if __name__ == '__main__': main()
