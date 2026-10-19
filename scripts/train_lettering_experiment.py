"""Test an exact-label lettering supplement before extended language training."""
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json, jsonl
from slm_lab.visual_selection import compare
from train_multimodal_pilot import complete_stage

ROOT = Path(__file__).resolve().parents[1]
COMPLETED = 'visual lettering experiment completed; selection recorded'


def main():
    os.chdir(ROOT)
    reports = ROOT / 'reports/lettering-experiment-v1'; reports.mkdir(exist_ok=True)
    data = ROOT / 'data/rendered-text-v1'
    prior = ROOT / 'reports/human-feedback-pilot-v1/status.json'

    def status(state, **fields):
        atomic_json(reports / 'status.json', {'status': state,
                    'updated_at': datetime.now(timezone.utc).isoformat(), **fields})

    while True:
        pending = []
        if not (data / 'manifest.json').exists():
            pending.append('completed lettering data preparation')
        if not prior.exists() or json.loads(prior.read_text()).get('status') != 'human-feedback pilot training/evaluation completed; broader goal unverified':
            pending.append('completed current human-feedback training/evaluation')
        if not pending:
            break
        status('waiting to test lettering curriculum', pending=pending); time.sleep(30)

    manifest = json.loads((data / 'manifest.json').read_text())
    for name, digest in manifest['files_sha256'].items():
        if file_hash(data / name) != digest:
            raise ValueError('Rendered records or font notice changed')
    base = ROOT / 'checkpoints/human-feedback-pilot-v1/dpo/model.pt'
    old = ROOT / 'data/alignment-curriculum-v1'
    old_manifest = json.loads((old / 'manifest.json').read_text())
    for name in ['images.train.jsonl', 'instruction.train.jsonl']:
        if file_hash(old / name) != old_manifest['files_sha256'][name]:
            raise ValueError('Original retention curriculum changed')
    curriculum = ROOT / 'data/lettering-experiment-v1'; curriculum.mkdir(exist_ok=True)
    # Resolve synthetic image paths without mutating the original audited files.
    with (curriculum / 'lettering.train.jsonl').open('w') as target:
        for row in jsonl(data / 'train.jsonl'):
            if file_hash(data / row['image']) != row['stored_image_sha256']:
                raise ValueError('Rendered image changed: ' + row['id'])
            row['image'] = str((data / row['image']).resolve())
            row['font_notice'] = str((data / row['font_notice']).resolve())
            target.write(json.dumps(row, ensure_ascii=True) + '\n')
    with (curriculum / 'joint.train.jsonl').open('w') as target:
        for source in [curriculum / 'lettering.train.jsonl', old / 'images.train.jsonl', old / 'instruction.train.jsonl']:
            with source.open() as handle:
                for line in handle:
                    target.write(line)
    specs = [
        {'name': 'lettering-warmup', 'stage': 'sft', 'file': 'lettering.train.jsonl',
         'trainable': 'vision', 'passes': 2, 'image_size': 224, 'batch': 4, 'accumulation': 2, 'lr': 1e-4},
        {'name': 'joint-reading', 'stage': 'sft', 'file': 'joint.train.jsonl',
         'trainable': 'all', 'passes': 1, 'image_size': 448, 'batch': 2, 'accumulation': 4, 'lr': 1e-5},
    ]
    for spec in specs:
        source = curriculum / spec['file']; spec['records'] = sum(1 for _ in jsonl(source))
        spec['steps'] = math.ceil(spec['records'] * spec['passes'] / (4 * spec['batch'] * spec['accumulation']))
        spec['data_sha256'] = file_hash(source)
    plan = {'base_checkpoint': str(base), 'base_checkpoint_sha256': file_hash(base),
            'rendered_manifest_sha256': file_hash(data / 'manifest.json'), 'stages': specs,
            'goal_achieved': False,
            'purpose': 'Test whether exact-label lettering improves real OCR while preserving measured photos/code/knowledge.',
            'selection': 'Use the candidate only if matched validation checks pass; otherwise retain the original DPO policy.'}
    plan_path = reports / 'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError('Lettering experiment differs from its existing plan')
    atomic_json(plan_path, plan)
    env = {**os.environ, 'OMP_NUM_THREADS': '1', 'TOKENIZERS_PARALLELISM': 'false'}
    checkpoint = base
    try:
        previous = None
        for spec in specs:
            directory = ROOT / 'checkpoints/lettering-experiment-v1' / spec['name']
            if not complete_stage(directory, spec):
                if shutil.disk_usage(ROOT).free < 25 * 1024**3:
                    raise ValueError('Insufficient free disk for lettering checkpoints')
                snapshots = sorted(directory.glob('step-*.pt'))
                resume = bool(snapshots)
                source_checkpoint = snapshots[-1] if resume else checkpoint
                if directory.exists() and any(directory.iterdir()) and not resume:
                    raise ValueError('Incomplete lettering stage has no resumable snapshot; inspect it')
                command = [str(ROOT / '.venv/bin/torchrun'), '--standalone', '--nproc_per_node=4', '-m', 'slm_lab.training', 'sft',
                           '--checkpoint', str(source_checkpoint), '--tokenizer', str(base.parent / 'tokenizer.json'),
                           '--data', str(curriculum / spec['file']), '--output', str(directory),
                           '--trainable', spec['trainable'], '--steps', str(spec['steps']),
                           '--batch-size', str(spec['batch']), '--gradient-accumulation', str(spec['accumulation']),
                           '--learning-rate', str(spec['lr']), '--image-size', str(spec['image_size']),
                           '--device', 'cuda', '--threads', '1', '--gradient-checkpointing',
                           '--checkpoint-every', '500', '--keep-checkpoints', '1', '--log-every', '10']
                if resume:
                    command.append('--resume')
                status('training lettering experiment', stage=spec['name'], planned_steps=spec['steps'], resume=resume)
                subprocess.run(command, check=True, cwd=ROOT, env=env)
                if not complete_stage(directory, spec):
                    raise ValueError('Lettering stage did not complete')
            checkpoint = directory / 'model.pt'
            if previous:
                for path in [previous / 'training-state.pt', *previous.glob('step-*.pt')]:
                    path.unlink(missing_ok=True)
            previous = directory
        for path in previous.glob('step-*.pt'):
            path.unlink(missing_ok=True)
        modules = [
            ('slm_lab.ocr_evaluation', 'ocr.json', ['--data', str(old / 'images.valid.jsonl')]),
            ('slm_lab.vision_evaluation', 'vision.json', ['--data', str(old / 'contrastive.valid.jsonl')]),
            ('slm_lab.knowledge_evaluation', 'knowledge.json', ['--data', str(ROOT / 'data/knowledge-eval-v1/questions.jsonl')]),
            ('slm_lab.code_quality', 'code-quality.json', []),
        ]
        for module, name, extra in modules:
            status('evaluating lettering experiment', task=module)
            subprocess.run([sys.executable, '-u', '-m', module, '--checkpoint', str(checkpoint),
                            '--output', str(reports / name), '--device', 'cuda', *extra], check=True, cwd=ROOT, env=env)
        status('evaluating held-out rendered lettering')
        subprocess.run([sys.executable, '-u', '-m', 'slm_lab.ocr_evaluation', '--checkpoint', str(checkpoint),
                        '--data', str(data / 'valid.jsonl'), '--output', str(reports / 'rendered-ocr.json'),
                        '--device', 'cuda'], check=True, cwd=ROOT, env=env)

        def read_reports(root):
            return {key: json.loads((root / name).read_text()) for key, name in [
                ('ocr', 'ocr.json'), ('vision', 'vision.json'), ('knowledge', 'knowledge.json'),
                ('code', 'code-quality.json'), ('code_instruction', 'code-quality-instruction.json')]}

        baseline = read_reports(ROOT / 'reports/human-feedback-pilot-v1/dpo')
        candidate = read_reports(reports)
        if any(value['checkpoint_sha256'] != plan['base_checkpoint_sha256'] for value in baseline.values()):
            raise ValueError('Baseline reports do not belong to the original DPO policy')
        candidate_sha = file_hash(checkpoint)
        if any(value['checkpoint_sha256'] != candidate_sha for value in candidate.values()):
            raise ValueError('Candidate reports do not belong to the lettering checkpoint')
        selection = compare(baseline, candidate)
        chosen = checkpoint if selection['promote_candidate'] else base
        atomic_json(reports / 'selection.json', {**selection, 'selected_checkpoint': str(chosen),
                    'selected_checkpoint_sha256': file_hash(chosen), 'candidate_checkpoint': str(checkpoint),
                    'baseline_checkpoint': str(base)})
        status(COMPLETED, selected_checkpoint=str(chosen), promoted=selection['promote_candidate'])
    except Exception as error:
        status('requires inspection; lettering experiment stopped', error=str(error)); raise


if __name__ == '__main__':
    main()
