"""CPU-only full-size gradient audit; no optimizer, data admission or model save."""
from contextlib import nullcontext
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
from tokenizers import Tokenizer
import torch

from slm_lab.codenet_stage_v1 import Objective, freeze_for_code_stage, objective_gradients, exposure_budget
from slm_lab.codenet_token_pack_v1 import CompleteCodePack, packed_encoder
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json, pixel_hash
from slm_lab.recurrent_ablation import state_digest
from slm_lab.training import Encoder, load_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Explicitly hide every GPU for this gradient-only audit')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(93041)
    folder = ROOT / 'reports/codenet-full-model-gradients-v1'
    if folder.exists():
        raise ValueError('Preserve earlier full-model gradient evidence')
    folder.mkdir()
    def status(message, **fields):
        atomic_json(folder / 'status.json', {'status': message, 'updated_at': datetime.now(timezone.utc).isoformat(),
            'device': 'cpu', 'optimizer_step_taken': False, 'dataset_admitted_to_training': False,
            'new_model_training_started': False, 'goal_achieved': False, **fields})
    try:
        status('verifying original complete-code inputs and protected own checkpoint')
        checkpoint = ROOT / 'checkpoints/extended-curriculum-v1/language-pretraining-interrupted-1/step-00012000.pt'
        pack_folder = ROOT / 'data/codenet-durable-pack-v1'
        pack = CompleteCodePack(pack_folder, 'train', purpose='audit')
        tokenizer_path = pack_folder / 'tokenizer.json'
        image_rows_path = ROOT / 'data/extended-curriculum-v1/images.train.jsonl'
        with image_rows_path.open() as source:
            image_row = json.loads(source.readline())
        if image_row['split'] != 'train' or not image_row['image']:
            raise ValueError('Original training-only image rehearsal required')
        image_path = Path(image_row['image'])
        if file_hash(image_path) != image_row['stored_image_sha256']:
            raise ValueError('Original training image bytes differ')
        with Image.open(image_path) as image:
            actual_pixel_hash = pixel_hash(image, 448)
        if actual_pixel_hash != image_row['model_pixel_sha256_448']:
            raise ValueError('Actual448px rehearsal image differs')
        long_row = short_row = None
        for line in gzip.open(pack_folder / 'original-rows.jsonl.gz', 'rt', encoding='utf-8'):
            row = json.loads(line)
            if row['split'] != 'train' or row['language'] not in {'C++', 'Python'}:
                continue
            if short_row is None and row['target_tokens'] <= 128:
                short_row = row
            if long_row is None and row['target_tokens'] > 2048:
                long_row = row
            if long_row is not None and short_row is not None:
                break
        if long_row is None or short_row is None:
            raise ValueError('Actual complete long and short coding examples required')
        locals_by_global = {int(global_index): local for local, global_index in enumerate(pack.positions)}
        positions = [locals_by_global[row['token_pack_index']] for row in [long_row, short_row]]
        code_rows = [{'packed_code_index': position} for position in positions]
        actual_code = []
        for original, position in zip([long_row, short_row], positions):
            ids, labels = pack.sequence(position)
            if hashlib.sha256(np.asarray(ids, dtype='<u2').tobytes()).hexdigest() != original['complete_input_ids_sha256']:
                raise ValueError('Whole actual code context differs')
            actual_code.append({'id': original['id'], 'original_archive_member': original['original_archive_member'],
                'group_id': original['group_id'], 'split': 'train', 'local_pack_index': position,
                'complete_context_tokens': len(ids), 'complete_target_tokens': sum(value != -100 for value in labels),
                'complete_input_ids_sha256': original['complete_input_ids_sha256']})
        files = [Path(__file__), ROOT / 'src/slm_lab/codenet_stage_v1.py', ROOT / 'src/slm_lab/codenet_token_pack_v1.py',
            ROOT / 'src/slm_lab/training.py', ROOT / 'src/slm_lab/model.py', ROOT / 'src/slm_lab/recurrent_ablation.py',
            checkpoint, pack_folder / 'manifest.json', tokenizer_path, image_rows_path, image_path]
        bound = {path: file_hash(path) for path in files}
        states = {path: (path.stat().st_ino, path.stat().st_size, path.stat().st_mtime_ns) for path in bound}
        def guard(full=False):
            for path, digest in bound.items():
                s = path.stat()
                if (s.st_ino, s.st_size, s.st_mtime_ns) != states[path] or (full and file_hash(path) != digest):
                    raise ValueError('Bound full-model audit source changed')
        atomic_json(folder / 'plan.json', {'recorded_at': datetime.now(timezone.utc).isoformat(),
            'source_files_sha256': {str(p): digest for p, digest in bound.items()},
            'own_source_checkpoint': str(checkpoint), 'code_records': actual_code,
            'actual448px_image_id': image_row['id'], 'actual448px_image_pixels_sha256': actual_pixel_hash,
            'objective': 'Complete-code global token mean +0.5 original image-caption token mean.',
            'hypothetical_three_pass_four_worker_code_budget': exposure_budget(len(pack), 3, 4, 1, 16),
            'CPU_threads': 4, 'device': 'cpu', 'optimizer_exists': False,
            'all_model_tensors_must_remain_bitwise_unchanged': True,
            'dataset_admitted_to_training': False, 'new_model_training_started': False, 'goal_achieved': False})
        status('loading own preserved706M weights on CPU; active GPU training untouched', code_records=actual_code)
        model, metadata = load_checkpoint(checkpoint, 'cpu', training_state=False)
        if metadata['tokenizer_sha256'] != bound[tokenizer_path] or sum(p.numel() for p in model.parameters()) != 706012672:
            raise ValueError('Exact own706M source/tokenizer required')
        before = state_digest(model.state_dict())
        trainable, frozen = freeze_for_code_stage(model)
        model.gradient_checkpointing = True
        model.train()
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        code_encoder = packed_encoder(pack, tokenizer, model.config)
        replay_encoder = Encoder(tokenizer, model.config, image_rows_path.parent, image_size=448)
        replay_ids, replay_labels = replay_encoder.sequence(image_row, 'response')
        replay_targets = sum(value != -100 for value in replay_labels)
        stages, started = [], time.monotonic()
        for objective in [Objective('complete_code', code_encoder, [[code_rows[0]], [code_rows[1]]], 'response', 1.),
                          Objective('image_caption_rehearsal', replay_encoder, [[image_row]], 'response', .5)]:
            guard()
            status('computing actual own-model CPU gradients without an optimizer', current_objective=objective.name,
                   completed_objectives=[item['name'] for item in stages])
            tick = time.monotonic()
            result = objective_gradients(model, [objective], 'cpu', nullcontext)
            stages.append({'name': objective.name, 'seconds': time.monotonic() - tick, **result})
            print('gradient_objective_complete', objective.name, result, flush=True)
        if stages[0]['total_target_tokens'] != sum(row['complete_target_tokens'] for row in actual_code):
            raise ValueError('Long/short programme targets differ from actual gradient inputs')
        if stages[1]['total_target_tokens'] != replay_targets or code_encoder.truncated or replay_encoder.truncated:
            raise ValueError('Actual rehearsal masks differ or any source was clipped')
        if any(parameter.grad is not None for parameter in frozen):
            raise ValueError('Frozen vision or critic unexpectedly received gradients')
        nonzero, norm_squared = 0, 0.
        for parameter in trainable:
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise ValueError('Every language parameter needs finite actual gradients')
            norm = float(torch.linalg.vector_norm(parameter.grad.detach()))
            nonzero += norm > 0
            norm_squared += norm * norm
        if nonzero == 0 or not math.isfinite(norm_squared) or norm_squared <= 0:
            raise ValueError('Actual own-model gradient must be nonzero and finite')
        after = state_digest(model.state_dict())
        if before != after or torch.cuda.is_initialized() or any(p.device.type != 'cpu' for p in model.parameters()):
            raise ValueError('Audit changed weights or used GPU resources')
        guard(full=True)
        atomic_json(folder / 'results.json', {'completed_at': datetime.now(timezone.utc).isoformat(),
            'own_parameter_count': sum(p.numel() for p in model.parameters()),
            'trainable_language_parameter_count': sum(p.numel() for p in trainable),
            'frozen_vision_and_critic_parameter_count': sum(p.numel() for p in frozen),
            'trainable_tensors_with_finite_gradients': len(trainable), 'nonzero_gradient_tensors': nonzero,
            'frozen_tensors_with_gradients': 0, 'unclipped_combined_gradient_l2': math.sqrt(norm_squared),
            'before_model_state_sha256': before, 'after_model_state_sha256': after,
            'complete_code_records': actual_code, 'image_rehearsal_target_tokens': replay_targets,
            'objective_results': stages, 'elapsed_gradient_seconds': time.monotonic() - started,
            'plan_sha256': file_hash(folder / 'plan.json'), 'device': 'cpu', 'CUDA_initialized': False,
            'optimizer_step_taken': False, 'model_saved': False, 'dataset_admitted_to_training': False,
            'new_model_training_started': False, 'goal_achieved': False,
            'scope': 'Actual own706M CPU forward/backward/input-mask plumbing only. No optimizer update, policy promotion, capability gain, CUDA capacity or throughput claim. Source checkpoint is the protected own12000-step snapshot, not the live model.'})
        status('own706M complete-code/image-rehearsal CPU gradient audit finished; weights unchanged',
               actual_code_targets=stages[0]['total_target_tokens'], actual_rehearsal_targets=replay_targets)
    except Exception as error:
        status('full-model gradient audit stopped; inspect preserved evidence', error=str(error))
        raise


if __name__ == '__main__':
    main()
