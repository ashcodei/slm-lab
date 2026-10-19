"""Resume the retained own curriculum using its exact frozen stage plan."""
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import torch

from .corpus import file_hash
from .diverse_images import atomic_json
from .evaluation_provenance import capture
from .completed_state_cleanup import retire

ROOT = Path(__file__).resolve().parents[2]


def resume_matches(metadata, spec, token_hash):
    if (metadata['stage'] != spec['stage'] or metadata['planned_steps'] != spec['steps']
            or metadata['data_sha256'] != spec['data_sha256'] or metadata['trainable'] != spec['trainable']
            or metadata['tokenizer_sha256'] != token_hash or metadata['world_size'] != 4
            or not 0 < metadata['steps'] < spec['steps'] or 'optimizer' not in metadata
            or 'torch_rng_state' not in metadata or 'cuda_rng_state' not in metadata):
        raise ValueError('Resume requires exact original stage/data/tokenizer/world and saved optimizer/RNG')
    options = metadata['training_options']
    required = {'batch_size': spec['batch'], 'gradient_accumulation': spec['accumulation'],
        'learning_rate': spec['lr'], 'image_size': spec['image_size'], 'gradient_checkpointing': True,
        'seed': 42, 'beta': .1, 'kl_coef': .05, 'ppo_epochs': 2, 'max_new_tokens': 16,
        'contrastive_temperature': .07}
    if options != required: raise ValueError('Original training options differ; do not silently change a resume')
    if metadata['truncated_sequences'] != 0 and spec['stage'] == 'pretrain':
        raise ValueError('Original packed pretraining snapshot reports clipping')
    return metadata['steps']


def command(spec, checkpoint, directory, resume=False):
    result = [str(ROOT / '.venv/bin/torchrun'), '--standalone', '--nproc_per_node=4', '-m', 'slm_lab.training',
        spec['stage'], '--checkpoint', str(checkpoint), '--tokenizer', str(Path(checkpoint).parent / 'tokenizer.json'),
        '--data', spec['data'], '--output', str(directory), '--trainable', spec['trainable'],
        '--steps', str(spec['steps']), '--batch-size', str(spec['batch']), '--gradient-accumulation', str(spec['accumulation']),
        '--learning-rate', str(spec['lr']), '--image-size', str(spec['image_size']), '--device', 'cuda', '--threads', '1',
        '--gradient-checkpointing', '--checkpoint-every', '1000', '--keep-checkpoints', '1', '--log-every', '10']
    if resume: result.append('--resume')
    return result


def preserve_interrupted_output(directory, checkpoint):
    """The core runner resumes only into a fresh output directory."""
    directory, checkpoint = Path(directory), Path(checkpoint)
    if checkpoint.parent != directory or not checkpoint.is_file():
        raise ValueError('Expected the verified source snapshot inside this interrupted stage')
    index = 1
    while directory.with_name(directory.name + '-interrupted-' + str(index)).exists(): index += 1
    retained = directory.with_name(directory.name + '-interrupted-' + str(index))
    directory.rename(retained)
    return retained / checkpoint.name


def main():
    from train_multimodal_pilot import complete_stage
    torch.set_num_threads(1)
    reports = ROOT / 'reports/extended-curriculum-v1'
    plan_path = reports / 'plan.json'; plan_sha = file_hash(plan_path); plan = json.loads(plan_path.read_text())
    files = [ROOT / name for name in capture()['files_sha256']] + [Path(__file__),
        ROOT / 'scripts/train_extended_curriculum.py', ROOT / 'scripts/train_multimodal_pilot.py',
        ROOT / 'src/slm_lab/completed_state_cleanup.py']
    runtime = capture(files)
    def status(state, **fields):
        atomic_json(reports / 'status.json', {'status': state, 'updated_at': datetime.now(timezone.utc).isoformat(),
            'goal_achieved': False, **fields})
    def guard():
        if file_hash(plan_path) != plan_sha or capture(files) != runtime:
            raise ValueError('Frozen plan or loaded recovery/training implementation changed')
    env = {**os.environ, 'OMP_NUM_THREADS': '1', 'TOKENIZERS_PARALLELISM': 'false'}
    checkpoint = Path(plan['base_checkpoint']); previous = checkpoint.parent
    try:
        guard()
        for spec in plan['stages']:
            data = Path(spec['data']); manifest = data / 'corpus.json' if data.is_dir() else data
            if file_hash(manifest) != spec['data_sha256']: raise ValueError('Frozen stage data changed: ' + spec['name'])
            if data.is_dir():
                corpus = json.loads(manifest.read_text())
                for split in corpus['splits'].values():
                    for shard in split['shards']:
                        if file_hash(data / shard['path']) != shard['sha256']: raise ValueError('Actual packed shard changed')
            directory = ROOT / 'checkpoints/extended-curriculum-v1' / spec['name']
            if not complete_stage(directory, spec):
                snapshots = sorted(directory.glob('step-*.pt')); resume = bool(snapshots)
                if resume:
                    start = snapshots[-1]
                    metadata = torch.load(start, map_location='cpu', weights_only=True, mmap=True)
                    completed = resume_matches(metadata, spec, file_hash(start.parent / 'tokenizer.json'))
                    if len(metadata['metrics']) != completed or metadata['metrics'][-1]['step'] != completed:
                        raise ValueError('Snapshot update metrics incomplete')
                    if spec['name'] == 'language-pretraining' and any(row['global_target_tokens'] != 262080 for row in metadata['metrics']):
                        raise ValueError('Original global pretraining target count changed')
                    source_digest = file_hash(start)
                    start = preserve_interrupted_output(directory, start)
                    receipt = {'verified_at': datetime.now(timezone.utc).isoformat(), 'stage': spec['name'],
                        'checkpoint': str(start), 'checkpoint_sha256': source_digest, 'resume_steps': completed,
                        'planned_steps': spec['steps'], 'frozen_plan_sha256': plan_sha, 'original_options': metadata['training_options'],
                        'resume_command': command(spec, start, directory, True), 'source_runtime': runtime, 'goal_achieved': False}
                    atomic_json(reports / ('restart-resume-' + spec['name'] + '-v2.json'), receipt)
                    del metadata; gc.collect()
                else:
                    start = checkpoint; completed = 0
                    if directory.exists() and any(directory.iterdir()):
                        raise ValueError('Incomplete stage has no verified resumable snapshot')
                if shutil.disk_usage(ROOT).free < 25 * 1024**3:
                    raise ValueError('Insufficient persistent space for atomic curriculum checkpoints')
                status('training extended public-data curriculum', stage=spec['name'], planned_steps=spec['steps'],
                       resume=resume, resumed_from_steps=completed)
                guard(); subprocess.run(command(spec, start, directory, resume), cwd=ROOT, check=True, env=env); guard()
                if not complete_stage(directory, spec): raise ValueError('Recovered stage did not complete its frozen budget')
            # Completed-stage consumers need exact retained weights, not Adam
            # state. Verify them through the existing retirement helper first.
            if (directory / 'training-state.pt').exists():
                retire(directory, reports / ('restart-retirement-' + spec['name'] + '.json'))
            for snapshot in directory.glob('step-*.pt'): snapshot.unlink()
            checkpoint = directory / 'model.pt'
            if previous:
                for path in [previous / 'training-state.pt', *previous.glob('step-*.pt')]: path.unlink(missing_ok=True)
            previous = directory
        for path in previous.glob('step-*.pt'): path.unlink(missing_ok=True)
        old = ROOT / 'data/alignment-curriculum-v1'; math_data = ROOT / 'data/math-reasoning-v2'
        public_code = ROOT / 'data/public-code-assessment-v1'; objects = ROOT / 'data/object-qa-v1'
        assessments = [
            ('slm_lab.code_quality', 'code-quality.json', []),
            ('slm_lab.knowledge_evaluation', 'knowledge.json', ['--data', str(ROOT / 'data/knowledge-eval-v1/questions.jsonl')]),
            ('slm_lab.math_evaluation', 'math.json', ['--data', str(math_data / 'test.jsonl')]),
            ('slm_lab.vision_evaluation', 'vision.json', ['--data', str(old / 'contrastive.valid.jsonl')]),
            ('slm_lab.ocr_evaluation', 'ocr.json', ['--data', str(old / 'images.valid.jsonl')]),
            ('slm_lab.object_evaluation', 'objects.json', ['--data', str(objects / 'valid.jsonl')]),
            ('slm_lab.public_code_benchmark', 'public-code.json', ['--data', str(public_code / 'questions.jsonl')]),
            ('slm_lab.cpp_evaluation', 'cpp-code.json', []),
        ]
        for module, name, extra in assessments:
            guard(); status('evaluating completed extended curriculum', task=module)
            subprocess.run([sys.executable, '-u', '-m', module, '--checkpoint', str(checkpoint), '--device', 'cuda',
                '--output', str(reports / name), *extra], check=True, cwd=ROOT, env=env)
        atomic_json(reports / 'summary.json', {'checkpoint': str(checkpoint), 'checkpoint_sha256': file_hash(checkpoint),
            'goal_achieved': False, 'high_end_llm_parity': 'Not established',
            'pending': ['Fresh human-feedback training and policy comparisons on this expanded model',
                        'Broader languages/repository-level tasks', 'Matched high-end-model capability comparisons']})
        status('extended language and multimodal training/evaluation completed; broader capability unverified', checkpoint=str(checkpoint))
    except Exception as error:
        status('requires inspection; recovered extended pipeline stopped', error=str(error)); raise
