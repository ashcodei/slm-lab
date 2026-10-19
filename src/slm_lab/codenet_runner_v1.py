"""Explicit-plan complete-code training; never launched by importing this module."""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
from tokenizers import Tokenizer
import torch
from torch import distributed as dist

from .codenet_checkpoint_v1 import FileBindings, restore_rng, save_collective, validate_resume
from .codenet_replay_v1 import FullReplayEncoder, ReplayCorpus
from .codenet_replay_schedule_v1 import coverage_replay_schedule
from .codenet_stage_v1 import EpochOrder, Objective, exposure_budget, freeze_for_code_stage, objective_gradients
from .codenet_token_pack_v1 import CompleteCodePack, packed_encoder
from .corpus import file_hash
from .diverse_images import atomic_json
from .model import ModelConfig, TinyMultimodalLM
from .training import TrainingOutputs, load_checkpoint


REQUIRED_GUARDS = {'held_code_functional', 'held_code_efficiency', 'public_mbpp',
    'six_language_module_code', 'general_knowledge', 'math', 'ocr', 'paired_photo',
    'own_context_reading', 'human_fact_questions'}
INTEGER_OPTIONS = {'seed', 'passes', 'batch_size', 'gradient_accumulation', 'replay_batch_size',
    'replay_accumulation', 'image_size', 'checkpoint_every', 'log_every'}
FLOAT_OPTIONS = {'learning_rate', 'weight_decay', 'gradient_clip', 'replay_coefficient'}
RUNTIME_FILES = {'codenet_runner_v1.py', 'codenet_checkpoint_v1.py', 'codenet_replay_v1.py',
    'codenet_replay_schedule_v1.py',
    'codenet_stage_v1.py', 'codenet_token_pack_v1.py', 'training.py', 'model.py', 'corpus.py', 'diverse_images.py'}


class VerifiedReplayEncoder(FullReplayEncoder):
    def pixels(self, row):
        tensor = super().pixels(row)
        if tensor is None:
            return None
        expected = row.get('model_pixel_sha256_' + str(self.image_size))
        rgb = np.rint((tensor.permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
        if not expected or hashlib.sha256(rgb.tobytes()).hexdigest() != expected:
            raise ValueError('Actual replay encoder pixels differ from the original golden hash')
        return tensor


class FixedReductionModel(torch.nn.Module):
    """Accumulate both objectives locally, then average gradients in fixed tensor order.

    The shared accumulation kernel already scales by world/global target count.
    Applying one worker average to the final sum retains those exact objective
    normalizers without depending on DDP's rebuilt bucket layout after resume.
    """
    def __init__(self, raw):
        super().__init__()
        self.outputs = TrainingOutputs(raw, 'sft')

    def forward(self, **batch):
        return self.outputs(**batch)

    def no_sync(self):
        return nullcontext()


def average_gradients(model, device, world):
    if world == 1:
        return
    parameters = [p for p in model.parameters() if p.requires_grad]
    missing = torch.tensor(sum(p.grad is None for p in parameters), device=device, dtype=torch.long)
    dist.all_reduce(missing, op=dist.ReduceOp.MAX)
    if missing.item():
        raise ValueError('Every worker must have gradients for every trainable language tensor')
    for parameter in parameters:
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        parameter.grad.div_(world)


def validate_plan(plan, world):
    if plan.get('format') != 'slm-complete-code-run-plan-v1' or plan.get('world_size') != world or world != 4:
        raise ValueError('Explicit four-worker complete-code run plan required')
    options = plan['training_options']
    if set(options) != INTEGER_OPTIONS | FLOAT_OPTIONS:
        raise ValueError('Exact complete-code training options required')
    if any(type(options[k]) is not int or options[k] < (0 if k == 'seed' else 1) for k in INTEGER_OPTIONS):
        raise ValueError('Valid explicit integer training options required')
    if any(type(options[k]) not in {int, float} or not math.isfinite(options[k]) or options[k] <= 0 for k in FLOAT_OPTIONS):
        raise ValueError('Finite positive objective and optimization coefficients required')
    if options['image_size'] != 448:
        raise ValueError('Bound448px replay interface required')
    if plan.get('training_ready') is not True or plan.get('automatic_candidate_promotion') is not False:
        raise ValueError('Complete readiness evidence and separate capability-based promotion required')
    if plan.get('gradient_reduction') != 'fixed_parameter_order_worker_average_v1':
        raise ValueError('Fixed resumable gradient-reduction contract required')
    if plan.get('replay_selection') != 'full_pass_floor_then_uniform_extra_family_updates_v1':
        raise ValueError('Explicit complete-source replay coverage contract required')
    for key in ['base_checkpoint', 'tokenizer', 'code_pack', 'replay', 'replay_audit', 'evaluation_binding', 'output']:
        if not Path(plan[key]).is_absolute():
            raise ValueError('Absolute immutable stage paths required')
    for key in ['base_checkpoint', 'tokenizer', 'code_manifest', 'code_admission', 'replay_manifest',
                'replay_audit', 'evaluation_binding']:
        digest = plan[key + '_sha256']
        if not isinstance(digest, str) or len(digest) != 64 or any(x not in '0123456789abcdef' for x in digest):
            raise ValueError('Exact source artifact digests required')
    package = Path(__file__).resolve().parent
    sources = plan['source_files_sha256']
    if not {str(package / name) for name in RUNTIME_FILES} <= sources.keys():
        raise ValueError('Complete neural runner, objective, reader and model source binding required')
    storage = plan['storage']
    if (set(storage) != {'maximum_checkpoint_bytes', 'maximum_final_model_bytes', 'minimum_free_bytes'}
        or any(type(v) is not int or v < 1 for v in storage.values())
        or storage['maximum_checkpoint_bytes'] < 8_300_000_000
        or storage['maximum_final_model_bytes'] < 3_100_000_000
        or storage['minimum_free_bytes'] < 26 * 1024**3):
        raise ValueError('Full own-model/Adam atomic replacement and persistent-writer storage budgets required')
    if not isinstance(plan.get('preceding_processes'), list) or not plan['preceding_processes']:
        raise ValueError('Explicit earlier curriculum worker identities required')
    return options


def bind_evaluations(plan):
    path = Path(plan['evaluation_binding'])
    if file_hash(path) != plan['evaluation_binding_sha256']:
        raise ValueError('Bound code, efficiency and broad-retention evaluation plan required')
    evaluations = json.loads(path.read_text())
    if (evaluations.get('format') != 'slm-complete-code-evaluation-binding-v1'
        or evaluations.get('baseline_checkpoint_sha256') != plan['base_checkpoint_sha256']
        or evaluations.get('baseline_completed') is not True
        or evaluations.get('promote_from_training_loss') is not False
        or set(evaluations.get('guards', {})) != REQUIRED_GUARDS):
        raise ValueError('Completed exact-checkpoint baseline and all capability guards required')
    files = {str(path): plan['evaluation_binding_sha256']}
    for name, guard in evaluations['guards'].items():
        if guard.get('baseline_complete') is not True or not guard.get('files_sha256'):
            raise ValueError('Missing baseline data/report binding: ' + name)
        if not Path(guard['baseline_report']).is_absolute() or guard['baseline_report'] not in guard['files_sha256']:
            raise ValueError('Actual baseline report required: ' + name)
        for filename, digest in guard['files_sha256'].items():
            if not Path(filename).is_absolute() or (filename in files and files[filename] != digest):
                raise ValueError('Inconsistent evaluation input/report binding')
            files[filename] = digest
    return files


def training_update(model, optimizer, code_pack, code_encoder, replay, replay_encoder,
                    step, options, device, amp, rank, world):
    """The tested sampler/objectives are the same path used by the actual driver."""
    if world > 1 and not isinstance(model, FixedReductionModel):
        raise ValueError('Distributed complete-code runner requires its explicit fixed reduction wrapper')
    code_order = EpochOrder(len(code_pack), options['seed'])
    code_groups = [[{'packed_code_index': i} for i in group] for group in code_order.rank_groups(
        step, rank, world, options['batch_size'], options['gradient_accumulation'])]
    total_steps = exposure_budget(len(code_pack), options['passes'], world,
                                 options['batch_size'], options['gradient_accumulation'])['optimizer_steps']
    populations = tuple((name,replay.size(name)) for name in replay.families)
    schedule = coverage_replay_schedule(populations,total_steps,
                                       world*options['replay_batch_size']*options['replay_accumulation'])
    name, occurrence = schedule.at(step)
    replay_order = EpochOrder(replay.size(name), options['seed'] + 1000 + replay.families.index(name))
    replay_groups = [[replay.row(name, i) for i in group] for group in replay_order.rank_groups(
        occurrence, rank, world, options['replay_batch_size'], options['replay_accumulation'])]
    if not 0 <= step < total_steps:
        raise ValueError('Optimizer step outside the declared complete exposure budget')
    learning_rate = options['learning_rate'] * min(1., (step + 1) / max(1, total_steps // 10))
    learning_rate *= .1 + .9 * .5 * (1 + math.cos(math.pi * step / total_steps))
    for group in optimizer.param_groups:
        group['lr'] = learning_rate
    optimizer.zero_grad(set_to_none=True)
    metrics = objective_gradients(model, [Objective('complete_code', code_encoder, code_groups, 'response', 1.),
        Objective(name, replay_encoder, replay_groups, 'response', options['replay_coefficient'])],
        device, amp, world=world)
    average_gradients(model, device, world)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), options['gradient_clip'], error_if_nonfinite=True)
    optimizer.step()
    return {**metrics, 'step': step + 1, 'replay_family': name, 'replay_occurrence': occurrence,
            'learning_rate': learning_rate, 'gradient_norm_before_clipping': float(norm)}


def train(plan_path, resume=False):
    plan_path = Path(plan_path).resolve()
    plan, plan_digest = json.loads(plan_path.read_text()), file_hash(plan_path)
    rank, world, local = (int(os.environ.get(k, v)) for k, v in [('RANK', '0'), ('WORLD_SIZE', '1'), ('LOCAL_RANK', '0')])
    options = validate_plan(plan, world)
    for actor in plan['preceding_processes']:
        proc = Path('/proc') / str(actor['pid'])
        try:
            fields = (proc / 'stat').read_text().split(') ', 1)[1].split()
        except FileNotFoundError:
            continue
        if fields[19] == str(actor['start_ticks']) and fields[0] != 'Z':
            raise ValueError('Earlier curriculum actor is still live; no overlapping GPU run')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 4:
        raise ValueError('Exactly the reserved four GPUs must be visible')
    device = torch.device('cuda', local)
    torch.cuda.set_device(device)
    if not torch.cuda.is_bf16_supported():
        raise ValueError('Bound BF16-capable CUDA environment required')
    torch.set_num_threads(1)
    dist.init_process_group('nccl')
    try:
        output = Path(plan['output'])
        output.mkdir(parents=True, exist_ok=True)
        storage = plan['storage']
        if shutil.disk_usage(output).free < (storage['minimum_free_bytes'] +
            2 * storage['maximum_checkpoint_bytes'] + storage['maximum_final_model_bytes']):
            raise ValueError('Insufficient whole-stage checkpoint/atomic replacement/final-model storage budget')
        if not resume and any(output.iterdir()):
            raise ValueError('Use a new output directory or explicit same-plan resume')
        bindings = {**plan['source_files_sha256'], str(plan_path): plan_digest,
            plan['base_checkpoint']: plan['base_checkpoint_sha256'], plan['tokenizer']: plan['tokenizer_sha256'],
            str(Path(plan['code_pack']) / 'manifest.json'): plan['code_manifest_sha256'],
            str(Path(plan['code_pack']) / 'admission.json'): plan['code_admission_sha256'],
            str(Path(plan['replay']) / 'manifest.json'): plan['replay_manifest_sha256'],
            plan['replay_audit']: plan['replay_audit_sha256'], **bind_evaluations(plan)}
        replay_audit = json.loads(Path(plan['replay_audit']).read_text())
        if (replay_audit.get('replay_manifest_sha256') != plan['replay_manifest_sha256']
            or replay_audit.get('source_rows_dropped_or_duplicated') != 0
            or replay_audit.get('encoder_truncated_sequences') != 0
            or replay_audit.get('final_checkpoint_replay_match_verified') is not True
            or replay_audit.get('matched_checkpoint_sha256') != plan['base_checkpoint_sha256']):
            raise ValueError('Independent complete replay and exact final-base coverage binding required')
        code = CompleteCodePack(plan['code_pack'], 'train', purpose='training')
        replay = ReplayCorpus(plan['replay'])
        bindings.update({str(code.folder / name): digest for name, digest in code.manifest['files_sha256'].items()})
        bindings.update({str(replay.folder / name): digest for name, digest in replay.manifest['artifact_files_sha256'].items()})
        bindings[str(replay.source)] = replay.manifest['original_training_source_sha256']
        guard = FileBindings(bindings)
        tokenizer = Tokenizer.from_file(plan['tokenizer'])
        if (code.manifest['files_sha256']['tokenizer.json'] != plan['tokenizer_sha256']
            or replay.manifest['tokenizer_sha256'] != plan['tokenizer_sha256']):
            raise ValueError('One exact own tokenizer across model and both objectives required')
        budget = exposure_budget(len(code), options['passes'], world, options['batch_size'], options['gradient_accumulation'])
        planned_steps = budget['optimizer_steps']
        # Check source coverage before allocating the full own model/Adam state.
        coverage_replay_schedule(tuple((name,replay.size(name)) for name in replay.families),planned_steps,
                                 world*options['replay_batch_size']*options['replay_accumulation'])
        random.seed(options['seed'] + rank)
        np.random.seed((options['seed'] + rank) % 2**32)
        torch.manual_seed(options['seed'] + rank)
        previous = None
        if resume:
            previous = torch.load(output / 'resume.pt', map_location='cpu', weights_only=True, mmap=True)
            # A crash after the last resumable save can still finish the inference artifact.
            start = validate_resume(previous, plan_digest, world, planned_steps, allow_completed=True)
            raw = TinyMultimodalLM(ModelConfig(**previous['config']))
            raw.load_state_dict(previous['model'])
            raw = raw.to(device)
        else:
            raw, metadata = load_checkpoint(plan['base_checkpoint'], device, training_state=False)
            if metadata['tokenizer_sha256'] != plan['tokenizer_sha256']:
                raise ValueError('Starting own policy/tokenizer differs')
            start = 0
        if raw.config != ModelConfig.load(Path(plan['code_pack']) / 'model-config.json'):
            raise ValueError('Original exact packed-code model configuration required')
        trainable, _ = freeze_for_code_stage(raw)
        raw.train()
        raw.gradient_checkpointing = True
        # Every rank loads the same bound source state; no DDP bucket state is implicit.
        model = FixedReductionModel(raw)
        optimizer = torch.optim.AdamW(trainable, lr=options['learning_rate'], weight_decay=options['weight_decay'])
        metrics = []
        if previous is not None:
            optimizer.load_state_dict(previous['optimizer'])
            metrics = previous['metrics'].copy()
            restore_rng(previous['worker_rng_states'][rank], device)
            del previous
        code_encoder = packed_encoder(code, tokenizer, raw.config)
        replay_encoder = VerifiedReplayEncoder(tokenizer, raw.config, replay.source.parent, image_size=options['image_size'])
        if rank == 0:
            tokenizer.save(str(output / 'tokenizer.json'))
            atomic_json(output / 'bound-plan.json', {'run_plan_sha256': plan_digest, 'budget': budget, 'source': str(plan_path)})

        def amp():
            return torch.autocast(device_type='cuda', dtype=torch.bfloat16)

        def bundle(states, step, include_optimizer):
            data = {'model': {k: v.detach().cpu() for k, v in raw.state_dict().items()},
                'config': asdict(raw.config), 'stage': 'sft', 'training_kind': 'complete_code_rehearsal_v1',
                'run_plan_sha256': plan_digest, 'tokenizer_sha256': plan['tokenizer_sha256'],
                'data_sha256': plan['code_manifest_sha256'], 'replay_manifest_sha256': plan['replay_manifest_sha256'],
                'source_checkpoint_sha256': plan['base_checkpoint_sha256'], 'steps': step,
                'planned_steps': planned_steps, 'world_size': world, 'worker_rng_states': states,
                'training_options': options, 'trainable': 'language_only_vision_and_critic_frozen',
                'gradient_reduction': plan['gradient_reduction'],
                'replay_selection': plan['replay_selection'],
                'image_prompt_coordinates': 'letterbox input', 'metrics': metrics,
                'truncated_sequences': 0, 'automatic_candidate_promotion': False}
            if include_optimizer:
                data['optimizer'] = optimizer.state_dict()
            return data

        for step in range(start, planned_steps):
            guard.check()
            torch.cuda.synchronize(device)
            tick = time.monotonic()
            result = training_update(model, optimizer, code, code_encoder, replay, replay_encoder,
                                     step, options, device, amp, rank, world)
            torch.cuda.synchronize(device)
            if code_encoder.truncated or replay_encoder.truncated:
                raise ValueError('Any clipped training example invalidates this complete-mask stage')
            result['step_seconds'] = time.monotonic() - tick
            if (step + 1) % options['log_every'] == 0 or step + 1 == planned_steps:
                metrics.append(result)
                if rank == 0:
                    atomic_json(output / 'progress.json', {'status': 'training', 'step': step + 1,
                        'planned_steps': planned_steps, 'latest_metric': result,
                        'run_plan_sha256': plan_digest, 'updated_at': datetime.now(timezone.utc).isoformat(),
                        'candidate_promoted': False, 'goal_achieved': False})
                    print(json.dumps(result), flush=True)
            if (step + 1) % options['checkpoint_every'] == 0 or step + 1 == planned_steps:
                guard.check(full=True)
                save_collective(output / 'resume.pt', lambda states: bundle(states, step + 1, True),
                    device, world, rank, storage['maximum_checkpoint_bytes'], storage['minimum_free_bytes'])
        guard.check(full=True)
        save_collective(output / 'model.pt', lambda states: bundle(states, planned_steps, False),
            device, world, rank, storage['maximum_final_model_bytes'], storage['minimum_free_bytes'])
        if rank == 0:
            atomic_json(output / 'training-completed.json', {'completed_at': datetime.now(timezone.utc).isoformat(),
                'model_sha256': file_hash(output / 'model.pt'), 'run_plan_sha256': plan_digest,
                'completed_steps': planned_steps, 'capability_evaluations_pending': True,
                'candidate_promoted': False, 'goal_achieved': False})
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    train(args.plan, resume=args.resume)


if __name__ == '__main__':
    main()
