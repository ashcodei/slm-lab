"""Independently reconstruct public-photo inputs and replay the actual update.

Reuse the declared training module, but not the producer's input/update helpers.
This is an independent execution of the saved next update on the small model.
"""
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

from slm_lab.codenet_checkpoint_v1 import restore_rng
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.local_source_closure_v2 import python_sources
from slm_lab.model import ModelConfig
from slm_lab.own_actor_vision_alignment_v1 import OwnActorVisionAlignment, pair_positives
from slm_lab.own_dense_actor_v1 import OwnDenseActor
from slm_lab.own_dense_growth_v1 import OwnDenseAssistant, new_backbone

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'reports/own-actor-vision-alignment-CPU-v2'
OUT = ROOT / 'reports/own-actor-vision-alignment-CPU-v2-readback'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or OUT.exists(): raise ValueError('Fresh CPU-only readback required')
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    plan = json.loads((SOURCE / 'plan.json').read_text()); complete = json.loads((SOURCE / 'complete.json').read_text())
    if (complete['actual_full_size_student_trained'] or not complete['next_update_weights_Adam_Python_NumPy_TorchCPU_RNG_bitexact']
            or complete['added_neural_values'] != 0 or complete['logical_updates'] != 2):
        raise ValueError('Exact completed small-architecture mechanism proof required')
    bindings = dict(complete['source_files_sha256'])
    for name, digest in bindings.items():
        if file_hash(name) != digest: raise ValueError('Producer source changed')
    tokenizer = Tokenizer.from_file(str(ROOT / 'data/tokenizer-corpus/tokenizer.json'))
    rows = {r['id']: r for r in map(json.loads, (ROOT / 'data/alignment-curriculum-v1/contrastive.train.jsonl').read_text().splitlines())}
    native = []; captions = []; pixels = []; pixel_keys = []; responses = []
    for evidence in plan['original_train_photos']:
        row = rows[evidence['id']]
        if row.get('split') != 'train' or row.get('annotation_origin') != evidence['annotation_origin']:
            raise ValueError('Original explicit human training role required')
        prefix = [1, *tokenizer.encode('User: ' + row['prompt'] + '\nAssistant:').ids]
        answer = [*tokenizer.encode(row['response']).ids, 2]
        ids = prefix + answer; labels = [-100] * len(prefix) + answer
        caption = [1, *tokenizer.encode(row['response']).ids, 2]
        if ids != evidence['complete_input_ids'] or labels != evidence['complete_labels'] or caption != evidence['caption_ids']:
            raise ValueError('Every whole original caption, prompt, mask and EOS must independently reconstruct')
        path = Path(row['image'])
        if file_hash(path) != row['stored_image_sha256']: raise ValueError('Original image changed')
        with Image.open(path) as image:
            image = ImageOps.pad(image.convert('RGB'), (448, 448), method=Image.Resampling.BICUBIC, color=(128, 128, 128))
            array = np.asarray(image, dtype=np.uint8).copy()
        if hashlib.sha256(array.tobytes()).hexdigest() != row['model_pixel_sha256_448']:
            raise ValueError('Original full448 model pixels must reconstruct')
        pixels.append(torch.from_numpy(array.astype(np.float32) / 127.5 - 1).permute(2, 0, 1))
        native.append((ids, labels)); captions.append(caption); pixel_keys.append(row['model_pixel_sha256_448']); responses.append(row['response'])
    def padded(sequences, fill):
        out = torch.full((4, max(map(len, sequences))), fill, dtype=torch.long)
        for i, sequence in enumerate(sequences): out[i, :len(sequence)] = torch.tensor(sequence)
        return out
    ids = padded([r[0] for r in native], 0); labels = padded([r[1] for r in native], -100)
    text_ids = padded(captions, 0); mask = text_ids != 0
    batch = dict(input_ids=ids, labels=labels, attention_mask=ids != 0, pixel_values=torch.stack(pixels))
    positive = pair_positives(pixel_keys, responses)
    # These explicit small dimensions/seeds match the source-bound producer;
    # this does not instantiate the actual 984M dense student.
    actor = OwnDenseActor(OwnDenseAssistant(new_backbone(ModelConfig(**plan['config']), 968417), reward_hidden=8,
        word_temporal=4, pointing_rank=8, router_hidden=8, detector_hidden=4, word_expansion=1), rank=2, seed=968517).eval()
    snapshot = torch.load(SOURCE / 'after-first.pt', map_location='cpu', weights_only=True)
    actor.base.load_state_dict(snapshot['base'], strict=True); actor.parts.load_state_dict(snapshot['parts'], strict=True)
    view = OwnActorVisionAlignment(actor).eval(); parameters = [p for _, p in view.named_selected()]
    adam = torch.optim.AdamW(parameters, lr=1e-4, betas=(.9, .95), weight_decay=.01, foreach=False)
    adam.load_state_dict(snapshot['optimizer']); restore_rng(snapshot['RNG'], 'cpu')
    position = (random.randrange(4) + int(np.random.randint(4)) + int(torch.randint(4, (1,)))) % 4
    order = torch.tensor([(position + i) % 4 for i in range(4)])
    actual = view.objective({k: v[order] for k, v in batch.items()}, text_ids[order], mask[order], positive[order][:, order])
    for key in ['loss', 'caption_CE', 'matching_loss']:
        if float(actual[key]) != complete['actual_updates'][1][key]: raise ValueError('Actual fresh next loss differs')
    if order.tolist() != complete['actual_updates'][1]['source_order']: raise ValueError('Actual private RNG source order differs')
    view.backward(actual); torch.nn.utils.clip_grad_norm_(parameters, 1.); adam.step(); adam.zero_grad(set_to_none=True)
    final = torch.load(SOURCE / 'small-mechanics-student.pt', map_location='cpu', weights_only=True)
    for key, live in [('base', actor.base.state_dict()), ('parts', actor.parts.state_dict())]:
        if live.keys() != final[key].keys() or any(not torch.equal(tensor, final[key][name]) for name, tensor in live.items()):
            raise ValueError('Independent next update does not reproduce every retained final value exactly')
    if actor.counts() != complete['actual_counts'] or any(float(s['step']) != 2 for s in adam.state.values()):
        raise ValueError('Original actual parameter count and Adam clocks required')
    for path in [Path(__file__), SOURCE / 'plan.json', SOURCE / 'complete.json', SOURCE / 'after-first.pt', SOURCE / 'small-mechanics-student.pt',
                 *python_sources(ROOT, [Path(__file__)])]: bindings[str(path)] = file_hash(path)
    for path, digest in bindings.items():
        if file_hash(path) != digest: raise ValueError('Readback source changed')
    if torch.cuda.is_initialized(): raise ValueError('No CUDA initialization allowed')
    OUT.mkdir()
    atomic_json(OUT / 'complete.json', dict(at=datetime.now(timezone.utc).isoformat(), source_files_sha256=bindings,
        independently_reconstructed_original_train_photos=4, complete_native_EOS_targets=int((labels[:, 1:] != -100).sum()),
        all_original_stored_images_full448_pixels_prompt_caption_masks_EOS_exact=True,
        actual_fresh_next_update_losses_and_all_final_weights_bitexact=True, actual_selected_Adam_clocks=2,
        actual_student_neural_values=actor.counts()['actual_unique'], added_neural_values=0,
        actual_full_size_student_trained=False, held_quality_gain_measured=False, full_GPU_admission=False,
        candidate_promoted=False, goal_achieved=False))
    print(json.dumps(dict(complete=str(OUT / 'complete.json'), small_actual_next_update_reproduced=True)), flush=True)


if __name__ == '__main__': main()
