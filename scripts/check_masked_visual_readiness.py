"""Full-size CUDA resource/gradient check on real public training photos only."""
import json
from pathlib import Path
import time

import torch

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json, jsonl
from slm_lab.evaluation_provenance import capture, seal, pixels_sha256
from slm_lab.masked_visual_pretraining import SOURCES, from_base, training_pixels, state_digest

ROOT = Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(1); torch.manual_seed(42)
    device = 'cuda:3'; torch.cuda.set_per_process_memory_fraction(.25, 3)
    files = SOURCES + [Path(__file__)]
    provenance = capture(files)
    source = ROOT / 'data/alignment-curriculum-v1/images.train.jsonl'
    digest = file_hash(source)
    rows = []
    for row in jsonl(source):
        if row['curriculum_source'] == 'everyday_photos' and row['task'] == 'caption':
            rows.append({**row, 'visual_source': 'everyday_photos'})
            if len(rows) == 32:
                break
    if len(rows) != 32:
        raise ValueError('Expected 32 actual public training photos')
    model, metadata = from_base(ROOT / 'checkpoints/human-feedback-pilot-v1/dpo/model.pt')
    model = model.to(device).train()
    before = state_digest(model)
    pixels = training_pixels(rows, model.vision.config.image_size, device, 0, 42)
    noise = torch.rand(len(rows), 196, device=device)
    optimizer = torch.optim.AdamW((value for value in model.parameters() if value.requires_grad), lr=1e-4)
    torch.cuda.reset_peak_memory_stats(device)
    records = []
    for update in range(2):
        optimizer.zero_grad(set_to_none=True); started = time.monotonic()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            result = model(pixels, noise=noise, gradient_checkpointing=True)
        result['loss'].backward()
        gradients = {name: float(value.grad.float().norm()) for name, value in
                     [('patch', model.vision.patch.weight), ('position', model.vision.position),
                      ('mask_token', model.mask_token), ('prediction', model.predict.weight)]}
        if any(not torch.isfinite(value.grad).all() or value.grad.norm() <= 0 for value in
               [model.vision.patch.weight, model.vision.position, model.mask_token, model.predict.weight]):
            raise ValueError('Missing/nonfinite full-size reconstruction gradients')
        if model.vision.project.weight.grad is not None:
            raise ValueError('Unused language projector received reconstruction gradients')
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step(); torch.cuda.synchronize(device)
        records.append({'optimizer_update': update + 1, 'loss': float(result['loss'].detach()),
                        'unclipped_gradient_norm': float(norm), 'gradient_norms': gradients,
                        'step_seconds': time.monotonic() - started})
    after = state_digest(model)
    if before == after:
        raise ValueError('Readiness optimizer updates did not change component weights')
    report = {**metadata, 'batch_size': len(rows), 'visible_patches': 49, 'total_patches': 196,
              'mask_ratio': .75, 'optimizer_updates': 2, 'real_image_ids': [row['id'] for row in rows],
              'augmented_input_pixels_sha256': [pixels_sha256(value) for value in pixels],
              'before_component_sha256': before, 'after_component_sha256': after,
              'updates': records, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
              'gpu': torch.cuda.get_device_name(device), 'checkpoint_saved': False,
              'goal_achieved': False, 'production_promoted': False,
              'scope': 'Two temporary full-size component optimizer updates on 32 real training photos verify resources/gradients only; no semantic or held-out capability evidence.'}
    seal(report, provenance, files=files, data=source, digest=digest, device=device,
         tokenizer_sha256=metadata['tokenizer_sha256'])
    atomic_json(ROOT / 'reports/masked-visual-readiness-v1.json', report)
    print(json.dumps({key: value for key, value in report.items() if key not in
                      ['real_image_ids', 'augmented_input_pixels_sha256', 'evaluation_provenance']}, indent=2))


if __name__ == '__main__':
    main()
