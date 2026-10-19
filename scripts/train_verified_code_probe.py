"""Measure a bounded verified-program SFT intervention on an own foundation snapshot."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

import torch
from tokenizers import Tokenizer

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json, jsonl
from slm_lab.model import ModelConfig, TinyMultimodalLM
from slm_lab.training import Encoder
from slm_lab.completed_state_cleanup import retire
from train_multimodal_pilot import complete_stage

ROOT = Path(__file__).resolve().parents[1]


def model_digest(values):
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--physical-gpu', type=int, default=2)
    parser.add_argument('--passes', type=int, default=2)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--reports', default='reports/verified-code-foundation-probe-v1')
    parser.add_argument('--checkpoints', default='checkpoints/verified-code-foundation-probe-v1')
    args = parser.parse_args()
    if args.passes < 1 or not 0 <= args.physical_gpu <= 3:
        parser.error('Positive passes and an existing local GPU index are required')
    torch.set_num_threads(1)
    reports = ROOT / args.reports
    checkpoints = ROOT / args.checkpoints
    reports.mkdir(exist_ok=True)

    def status(message, **fields):
        atomic_json(reports / 'status.json', {'status': message,
                    'updated_at': datetime.now(timezone.utc).isoformat(), 'goal_achieved': False, **fields})

    try:
        source = Path(args.checkpoint).resolve()
        data = ROOT / 'data/verified-code-curriculum-v1'
        manifest = json.loads((data / 'manifest.json').read_text())
        data_digest = file_hash(data / 'train.jsonl')
        if data_digest != manifest['train_sha256']:
            raise ValueError('Verified-program training data changed')
        rows = list(jsonl(data / 'train.jsonl'))
        if len(rows) != manifest['train_examples'] or any(row.get('image') for row in rows):
            raise ValueError('Expected the frozen text-only verified-program curriculum')
        base = checkpoints / 'base/model.pt'
        plan_path = reports / 'plan.json'
        implementations = {name: file_hash(ROOT / name) for name in
                           ['scripts/train_verified_code_probe.py', 'src/slm_lab/model.py',
                            'src/slm_lab/training.py', 'src/slm_lab/objectives.py',
                            'src/slm_lab/completed_state_cleanup.py']}
        if plan_path.exists():
            if not args.resume:
                raise ValueError('Existing experiment requires explicit inspected resume')
            plan = json.loads(plan_path.read_text())
            if (plan['passes'] != args.passes or plan['physical_gpu'] != args.physical_gpu
                    or plan['source_checkpoint'] != str(source)
                    or plan['training_data_sha256'] != data_digest
                    or plan['implementations'] != implementations
                    or file_hash(base) != plan['base_checkpoint_sha256']):
                raise ValueError('Recorded coding probe differs from this invocation')
        else:
            if checkpoints.exists() and any(checkpoints.iterdir()):
                raise ValueError('Unrecorded checkpoint preparation requires inspection')
            if shutil.disk_usage(ROOT).free < 35 * 1024**3:
                raise ValueError('Need 35 GiB free before creating this separate probe')
            status('preparing weights-only own foundation snapshot')
            source_digest = file_hash(source)
            state = torch.load(source, map_location='cpu', weights_only=True, mmap=True)
            if state['stage'] != 'pretrain':
                raise ValueError('This intervention requires an own foundation pretraining snapshot')
            config = ModelConfig(**state['config'])
            tokenizer_path = source.parent / 'tokenizer.json'
            if file_hash(tokenizer_path) != state['tokenizer_sha256']:
                raise ValueError('Source tokenizer changed')
            tokenizer = Tokenizer.from_file(str(tokenizer_path))
            encoder = Encoder(tokenizer, config, data, 448)
            target_tokens = 0
            maximum_tokens = 0
            for row in rows:
                ids, labels = encoder.sequence(row, 'response')
                maximum_tokens = max(maximum_tokens, len(ids))
                target_tokens += sum(value != -100 for value in labels[1:])
            if encoder.truncated:
                raise ValueError('The proposed verified-code probe would truncate examples')
            with torch.device('meta'):
                parameter_count = TinyMultimodalLM(config).parameter_counts()['total']
            if parameter_count >= 1_000_000_000:
                raise ValueError('Model exceeds user parameter limit')
            before = model_digest(state['model'])
            base.parent.mkdir(parents=True)
            # Keep metadata and learned weights; old Adam/RNG/reference states
            # are unnecessary for a new intervention and remain in the active source.
            shared_source = 'optimizer' not in state and 'reference_model' not in state
            if shared_source:
                # A completed model.pt already excludes large optimizer/reference
                # histories. Sharing its immutable bytes avoids another 2.8 GiB copy.
                os.link(source, base)
            else:
                payload = {key: value for key, value in state.items()
                           if key not in {'optimizer', 'reference_model', 'torch_rng_state', 'cuda_rng_state'}}
                payload['source_checkpoint_sha256'] = source_digest
                temporary = base.with_suffix('.tmp')
                torch.save(payload, temporary)
                temporary.replace(base)
            shutil.copy2(tokenizer_path, base.parent / 'tokenizer.json')
            saved = torch.load(base, map_location='cpu', weights_only=True, mmap=True)
            if model_digest(saved['model']) != before or saved['config'] != state['config']:
                raise ValueError('Weights-only source differs from the foundation snapshot')
            if file_hash(source) != source_digest:
                raise ValueError('Foundation source changed during snapshot conversion')
            plan = {
                'source_checkpoint': str(source), 'source_checkpoint_sha256': source_digest,
                'source_completed_steps': state['steps'], 'base_checkpoint': str(base),
                'base_shares_exact_source_bytes': shared_source,
                'base_checkpoint_sha256': file_hash(base), 'model_state_sha256': before,
                'parameters': parameter_count, 'tokenizer_sha256': state['tokenizer_sha256'],
                'training_manifest_sha256': file_hash(data / 'manifest.json'),
                'training_data_sha256': data_digest, 'training_rows': len(rows),
                'unique_source_programs': manifest['unique_programs'], 'passes': args.passes,
                'variant_row_exposures': args.passes * len(rows),
                'targets_per_pass': target_tokens, 'maximum_sequence_tokens': maximum_tokens,
                'truncated_training_examples': 0, 'steps': math.ceil(args.passes * len(rows) / 8),
                'batch_size': 2, 'gradient_accumulation': 4, 'learning_rate': 1e-5,
                'physical_gpu': args.physical_gpu, 'implementations': implementations,
                'scope': 'Separate own-language supervised code experiment; no automatic production promotion.',
                'goal_achieved': False,
                'limitations': manifest['limitations'] + [
                    'Name variants and repeated passes are exposures, not distinct algorithms.',
                    'Python-focused intervention can change other languages and general behavior.',
                    'Local code correctness/stress checks do not establish ANY-code or high-end general capability.',
                ],
            }
            atomic_json(plan_path, plan)
            del saved, state
            if not shared_source: del payload

        environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(args.physical_gpu),
                       'OMP_NUM_THREADS': '1', 'TOKENIZERS_PARALLELISM': 'false'}

        def assess(checkpoint, label):
            for mode in ['completion', 'instruction']:
                path = reports / (label + '-' + mode + '.json')
                if path.exists():
                    report = json.loads(path.read_text())
                    if report['checkpoint_sha256'] != file_hash(checkpoint) or report['prompt_mode'] != mode:
                        raise ValueError('Existing coding report differs from the frozen checkpoint/protocol')
                    continue
                status('evaluating bounded coding intervention', branch=label, prompt_mode=mode)
                subprocess.run([sys.executable, '-u', '-m', 'slm_lab.code_quality', '--checkpoint', str(checkpoint),
                                '--output', str(path), '--device', 'cuda', '--prompt-mode', mode,
                                '--max-new-tokens', '512'], check=True, cwd=ROOT, env=environment)

        assess(base, 'baseline')
        candidate = checkpoints / 'candidate'
        spec = {'stage': 'sft', 'trainable': 'language', 'image_size': 448,
                'steps': plan['steps'], 'data_sha256': data_digest}
        if not complete_stage(candidate, spec):
            if shutil.disk_usage(ROOT).free < 25 * 1024**3:
                raise ValueError('Insufficient free disk for the separate coding probe')
            snapshots = sorted(candidate.glob('step-*.pt'))
            if candidate.exists() and any(candidate.iterdir()) and not (args.resume and snapshots):
                raise ValueError('Incomplete coding probe needs inspected resumable state')
            resume = bool(args.resume and snapshots)
            status('training verified-program intervention', steps=plan['steps'], physical_gpu=args.physical_gpu)
            command = [sys.executable, '-u', '-m', 'slm_lab.training', 'sft', '--checkpoint',
                       str(snapshots[-1] if resume else base), '--tokenizer', str(base.parent / 'tokenizer.json'),
                       '--data', str(data / 'train.jsonl'), '--output', str(candidate), '--device', 'cuda',
                       '--trainable', 'language', '--steps', str(plan['steps']), '--batch-size', '2',
                       '--gradient-accumulation', '4', '--learning-rate', '1e-5', '--image-size', '448',
                       '--threads', '1', '--gradient-checkpointing', '--checkpoint-every', '100',
                       '--keep-checkpoints', '1', '--log-every', '10']
            if resume:
                command.append('--resume')
            subprocess.run(command, check=True, cwd=ROOT, env=environment)
            if not complete_stage(candidate, spec):
                raise ValueError('Verified-code intervention did not complete')
        for path in candidate.glob('step-*.pt'):
            path.unlink()
        initial = torch.load(base, map_location='cpu', weights_only=True, mmap=True)
        trained = torch.load(candidate / 'model.pt', map_location='cpu', weights_only=True, mmap=True)
        vision_names = [name for name in initial['model'] if name.startswith('vision.')]
        if (not vision_names or initial['config'] != trained['config']
                or initial['tokenizer_sha256'] != trained['tokenizer_sha256']
                or any(not torch.equal(initial['model'][name], trained['model'][name]) for name in vision_names)):
            raise ValueError('Language-only code intervention changed the architecture/tokenizer/vision weights')
        atomic_json(reports / 'vision-preservation.json', {'exact_frozen_vision_tensors': len(vision_names),
                    'base_checkpoint_sha256': file_hash(base), 'candidate_checkpoint_sha256': file_hash(candidate / 'model.pt'),
                    'goal_achieved': False})
        del initial, trained
        assess(candidate / 'model.pt', 'candidate')
        comparison = {}
        for mode in ['completion', 'instruction']:
            before = json.loads((reports / ('baseline-' + mode + '.json')).read_text())
            after = json.loads((reports / ('candidate-' + mode + '.json')).read_text())
            for field in ['evaluation_provenance', 'tokenizer_sha256', 'generation_budget',
                          'decoding', 'prompt_mode', 'suite_sha256', 'task_count', 'inference_precision']:
                if before[field] != after[field]:
                    raise ValueError('Before/after coding evaluation protocol differs: ' + field)
            keys = lambda report: [(row['language'], row['task'], row['specification']) for row in report['results']]
            if keys(before) != keys(after):
                raise ValueError('Before/after coding cases differ')
            comparison[mode] = {name: {'baseline': before[name], 'candidate': after[name]}
                                for name in ['syntax_pass_rate', 'functional_pass_at_1',
                                             'efficiency_stress_pass_rate', 'by_language']}
        if {name: file_hash(ROOT / name) for name in implementations} != implementations:
            raise ValueError('Coding probe implementation changed during the run')
        atomic_json(reports / 'comparison.json', {'plan': str(plan_path), 'results': comparison,
                    'production_promoted': False, 'goal_achieved': False,
                    'limitations': plan['limitations']})
        if (candidate / 'training-state.pt').exists():
            retire(candidate, reports / 'completed-state-cleanup.json')
        status('verified-code intervention and local before/after checks completed; broader capability unverified')
    except Exception as error:
        status('requires inspection; verified-code probe stopped', error=str(error))
        raise


if __name__ == '__main__':
    main()
