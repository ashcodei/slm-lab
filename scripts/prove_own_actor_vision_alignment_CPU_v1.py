"""Real public-photo, full-resolution small-architecture mechanism proof.

This does not train/admit the current full-size student or measure held quality.
"""
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageOps
import torch
from tokenizers import Tokenizer

from slm_lab.codenet_checkpoint_v1 import capture_rng, restore_rng
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.local_source_closure_v2 import python_sources
from slm_lab.model import ModelConfig
from slm_lab.own_actor_vision_alignment_v1 import OwnActorVisionAlignment, pair_positives
from slm_lab.own_dense_actor_v1 import OwnDenseActor
from slm_lab.own_dense_growth_v1 import OwnDenseAssistant, new_backbone
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/own-actor-vision-alignment-CPU-v1'


def equal(a, b):
    if isinstance(a, torch.Tensor): return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict): return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)): return type(a) is type(b) and len(a) == len(b) and all(map(lambda ab: equal(*ab), zip(a, b)))
    return a == b


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or OUT.exists():
        raise ValueError('Fresh CPU-hidden proof directory required; no automatic restart')
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    source = ROOT / 'data/alignment-curriculum-v1'
    manifest = json.loads((source / 'manifest.json').read_text())
    populations = {}
    for split in ['train', 'valid']:
        path = source / f'contrastive.{split}.jsonl'
        if file_hash(path) != manifest['files_sha256'][path.name]: raise ValueError('Exact source caption file required')
        populations[split] = [json.loads(line) for line in path.read_text().splitlines()]
        if any(row['split'] != split for row in populations[split]): raise ValueError('Every original role must agree')
    overlap = {key: len({r[key] for r in populations['train']} & {r[key] for r in populations['valid']})
               for key in ['group_id', 'model_pixel_sha256_448', 'stored_image_sha256']}
    if any(overlap.values()): raise ValueError('Original held groups/pixels must not enter the train-photo proof')
    eligible = [r for r in populations['train'] if r.get('annotation_origin') ==
                'Original human spoken image description, manually transcribed.']
    rows = sorted(eligible, key=lambda r: hashlib.sha256(('vision-mechanism-source-only-v1:' + r['id']).encode()).hexdigest())[:4]
    if len(rows) != 4 or len({r['group_id'] for r in rows}) != 4: raise ValueError('Four fixed original human train photos required')
    tokenizer_path = ROOT / 'data/tokenizer-corpus/tokenizer.json'
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if (tokenizer.get_vocab_size() != 32768 or [tokenizer.token_to_id(s) for s in ['<pad>', '<bos>', '<eos>']] != [0, 1, 2]):
        raise ValueError('Original own complete tokenizer required')
    config = ModelConfig(vocab_size=32768, hidden_size=32, intermediate_size=64, num_layers=2,
        num_heads=4, num_kv_heads=2, max_text_length=4096, vision_hidden_size=32, vision_intermediate_size=64,
        vision_layers=1, vision_heads=4, image_size=224, patch_size=16, pool_size=2)
    def create():
        return OwnDenseActor(OwnDenseAssistant(new_backbone(config, 968417), reward_hidden=8,
            word_temporal=4, pointing_rank=8, router_hidden=8, detector_hidden=4, word_expansion=1),
            rank=2, seed=968517).eval()
    actor = create(); view = OwnActorVisionAlignment(actor).eval()
    pixels = []; sequences = []; texts = []; controls = []; inputs = [Path(__file__), tokenizer_path,
        source / 'manifest.json', source / 'contrastive.train.jsonl', source / 'contrastive.valid.jsonl',
        ROOT / 'tests/test_own_actor_vision_alignment_v1.py', ROOT / 'reports/own-actor-vision-alignment-v1-tests.log']
    for row in rows:
        path = Path(row['image'])
        if file_hash(path) != row['stored_image_sha256']: raise ValueError('Exact original stored image required')
        with Image.open(path) as image:
            image = ImageOps.pad(image.convert('RGB'), (448, 448), method=Image.Resampling.BICUBIC, color=(128, 128, 128))
            raw = np.asarray(image, dtype=np.uint8).copy()
        if hashlib.sha256(raw.tobytes()).hexdigest() != row['model_pixel_sha256_448']:
            raise ValueError('Full native448 image bytes must reconstruct')
        pixels.append(torch.from_numpy(raw.astype(np.float32) / 127.5 - 1).permute(2, 0, 1))
        prompt = [1, *tokenizer.encode('User: ' + row['prompt'] + '\nAssistant:').ids]
        response = [*tokenizer.encode(row['response']).ids, 2]
        text = [1, *tokenizer.encode(row['response']).ids, 2]
        if any(t in (0, 1, 2) for t in response[:-1]) or len(prompt + response) > 4096:
            raise ValueError('Whole unshortened human caption and genuine EOS must fit')
        sequences.append((prompt + response, [-100] * len(prompt) + response)); texts.append(text)
        controls.append(dict(id=row['id'], group_id=row['group_id'], source=row['source'], split=row['split'],
            annotation_origin=row['annotation_origin'], response=row['response'], image=str(path),
            stored_image_sha256=row['stored_image_sha256'], model_pixel_sha256_448=row['model_pixel_sha256_448'],
            complete_input_ids=prompt + response, complete_labels=[-100] * len(prompt) + response, caption_ids=text))
        inputs.append(path)
    def padded(seqs, pad):
        result = torch.full((len(seqs), max(map(len, seqs))), pad, dtype=torch.long)
        for i, seq in enumerate(seqs): result[i, :len(seq)] = torch.tensor(seq)
        return result
    ids = padded([r[0] for r in sequences], 0); labels = padded([r[1] for r in sequences], -100)
    text_ids = padded(texts, 0); text_mask = text_ids != 0
    batch = dict(input_ids=ids, labels=labels, attention_mask=ids != 0, pixel_values=torch.stack(pixels))
    positive = pair_positives([r['model_pixel_sha256_448'] for r in rows], [r['response'] for r in rows])
    inputs += python_sources(ROOT, [Path(__file__), ROOT / 'tests/test_own_actor_vision_alignment_v1.py'])
    bindings = {str(p): file_hash(p) for p in inputs}
    OUT.mkdir()
    atomic_json(OUT / 'plan.json', dict(at=datetime.now(timezone.utc).isoformat(), config=asdict(config),
        actual_counts=actor.counts(), source_files_sha256=bindings, original_train_photos=controls,
        original_train_rows=len(populations['train']), original_valid_rows=len(populations['valid']),
        human_spoken_train_rows=len(eligible), metadata_caption_train_rows=len(populations['train']) - len(eligible),
        exact_train_valid_overlap=overlap, image_side=448, native_visual_prefix_tokens=config.image_token_count(448),
        logical_updates=2, physical_updates=3, caption_CE_weight=1., matching_weight=.1, temperature=.07,
        own_small_initialization_not_full_current_student=True, held_quality_evaluation=False,
        all_original_train_records_consumed=False, full_GPU_admission=False, candidate_promoted=False, goal_achieved=False))
    selected = view.named_selected(); parameters = [p for _, p in selected]; chosen = {id(p) for p in parameters}
    frozen = lambda: tensor_digest((n, p) for n, p in actor.named_parameters() if id(p) not in chosen)
    frozen_before = frozen(); initial = {n: p.detach().clone() for n, p in actor.named_parameters()}
    with torch.no_grad():
        text_before = actor(text_ids, attention_mask=text_mask)['logits'].clone()
        if not torch.equal(actor(**batch)['logits'], view.native(**batch)['logits']):
            raise ValueError('Training view must preserve initial native image outputs exactly')
        word_before = actor.base.word_logits(batch['pixel_values']).clone()
    adam = torch.optim.AdamW(parameters, lr=1e-4, betas=(.9, .95), weight_decay=.01, foreach=False)
    random.seed(968617); np.random.seed(968617); torch.manual_seed(968617)
    def update():
        # Actually consume all three private RNGs so a missing stream breaks replay.
        position = (random.randrange(4) + int(np.random.randint(4)) + int(torch.randint(4, (1,)))) % 4
        order = torch.tensor([(position + i) % 4 for i in range(4)])
        native = {k: v[order] for k, v in batch.items()}
        objective = view.objective(native, text_ids[order], text_mask[order], positive[order][:, order])
        view.backward(objective)
        norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.))
        metrics = {k: float(v.detach()) if isinstance(v, torch.Tensor) else v for k, v in objective.items()}
        metrics.update(source_order=order.tolist(), gradient_norm_before_clip=norm,
            learned_448_position_gradient_nonzero=bool(actor.base.network.vision.position.grad.abs().sum() > 0))
        adam.step(); adam.zero_grad(set_to_none=True)
        if frozen() != frozen_before: raise ValueError('Language/actor/nonselected heads changed')
        with torch.no_grad():
            if not torch.equal(text_before, actor(text_ids, attention_mask=text_mask)['logits']):
                raise ValueError('Actual text-only native logits changed')
        return metrics
    first = update()
    snapshot = dict(base=copy.deepcopy(actor.base.state_dict()), parts=copy.deepcopy(actor.parts.state_dict()),
                    optimizer=copy.deepcopy(adam.state_dict()), RNG=capture_rng('cpu'), committed_updates=1)
    torch.save(snapshot, OUT / 'after-first.pt')
    second = update(); expected_base = copy.deepcopy(actor.base.state_dict()); expected_parts = copy.deepcopy(actor.parts.state_dict())
    expected_adam = copy.deepcopy(adam.state_dict()); expected_rng = capture_rng('cpu')
    restored = torch.load(OUT / 'after-first.pt', map_location='cpu', weights_only=True)
    actor.base.load_state_dict(restored['base'], strict=True); actor.parts.load_state_dict(restored['parts'], strict=True)
    adam.load_state_dict(restored['optimizer']); restore_rng(restored['RNG'], 'cpu')
    replayed = update()
    if not all([equal(second, replayed), equal(expected_base, actor.base.state_dict()),
        equal(expected_parts, actor.parts.state_dict()), equal(expected_adam, adam.state_dict()), equal(expected_rng, capture_rng('cpu'))]):
        raise ValueError('Actual next native update/Adam/three private RNG streams must replay exactly')
    if len(adam.state) != len(parameters) or any(float(s['step']) != 2 for s in adam.state.values()):
        raise ValueError('Every selected image and matching Adam clock must equal2')
    changed = [n for n, p in actor.named_parameters() if not torch.equal(initial[n], p)]
    if not changed or any(not n.startswith(('base.network.vision.', 'base.heads.pointing.')) for n in changed):
        raise ValueError('Only the explicitly selected original image and pointing values may change')
    torch.save(dict(base=actor.base.state_dict(), parts=actor.parts.state_dict(), own_small_mechanics_only=True,
                    full_size_student=False, candidate_promoted=False, goal_achieved=False), OUT / 'small-mechanics-student.pt')
    cold = create(); saved = torch.load(OUT / 'small-mechanics-student.pt', map_location='cpu', weights_only=True)
    cold.base.load_state_dict(saved['base'], strict=True); cold.parts.load_state_dict(saved['parts'], strict=True)
    with torch.no_grad():
        text_cold = torch.equal(actor(text_ids, attention_mask=text_mask)['logits'], cold(text_ids, attention_mask=text_mask)['logits'])
        image_cold = torch.equal(actor(**batch)['logits'], cold(**batch)['logits'])
        word_changed = not torch.equal(word_before, actor.base.word_logits(batch['pixel_values']))
    if not text_cold or not image_cold or torch.cuda.is_initialized():
        raise ValueError('Native cold text/image reload must be exact and CUDA must remain uninitialized')
    for path, digest in bindings.items():
        if file_hash(path) != digest: raise ValueError('Immutable proof source changed')
    atomic_json(OUT / 'complete.json', dict(at=datetime.now(timezone.utc).isoformat(), actual_counts=actor.counts(),
        source_files_sha256=bindings, actual_updates=[first, second], actual_replayed_update=replayed,
        logical_updates=2, physical_updates=3, selected_parameter_tensors=len(parameters),
        selected_neural_values=sum(p.numel() for p in parameters), changed_original_tensors=changed,
        added_neural_values=0, complete_original_caption_EOS_targets=int((labels[:, 1:] != -100).sum()),
        actual_full_resolution_real_train_photos=4, image_side=448, native_visual_prefix_tokens=196,
        all_nonselected_weights_unchanged=True, actual_text_only_logits_bitexact_after_both_updates=True,
        next_update_weights_Adam_Python_NumPy_TorchCPU_RNG_bitexact=True,
        actual_cold_native_text_logits_bitexact=text_cold, actual_cold_native_image_logits_bitexact=image_cold,
        word_head_weights_unchanged_but_shared_vision_output_changed=word_changed,
        held_image_knowledge_or_unassisted_coding_gain_measured=False, actual_full_size_student_trained=False,
        full_GPU_capacity_or_distributed_gradient_admission=False, candidate_promoted=False, goal_achieved=False,
        limitations=['Small own initialization tests mechanisms only; no current full-size model or held quality claim.',
            'Only four deterministically selected original human train photos; no complete source epoch.',
            'Exact pixel/group separation does not rule out perceptual/semantic overlap or older exposure.',
            'Shared vision changes may affect OCR, pointing and object behavior; independent retention tests remain required.',
            'No new full-size export format or fresh four-GPU admission has been implemented.']))
    print(json.dumps(dict(complete=str(OUT / 'complete.json'), actual_counts=actor.counts(),
        full_size_student_trained=False, held_gain_measured=False, goal_achieved=False)), flush=True)


if __name__ == '__main__': main()
