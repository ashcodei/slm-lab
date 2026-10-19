"""Preserve an exact inference-only view of an existing own training snapshot."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil

import torch

from prepare_codenet_native_contract_pack_v2 import occupied_writer_reservation
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.recurrent_ablation import state_digest

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / 'checkpoints/extended-curriculum-v1/multilingual-code-instruction/step-00004000.pt'
    destination = ROOT / 'checkpoints/code-instruction-4000-assessment-v1'
    if destination.exists():
        raise ValueError('Preserve every earlier inference snapshot or partial extraction')
    if not source.is_file():
        raise ValueError('Require the actual committed4000-update checkpoint; never substitute another update')
    occupied = occupied_writer_reservation()
    reserve, maximum = 26*1024**3-occupied['credited_bytes'], 3*1024**3
    if shutil.disk_usage(ROOT).free < reserve + maximum + 64*1024**2:
        raise ValueError('Full inference artifact budget above remaining live writer commitments required')
    original_digest = file_hash(source)
    state = torch.load(source, map_location='cpu', weights_only=True, mmap=True)
    expected = dict(stage='sft', steps=4000, planned_steps=28343, world_size=4,
        data_sha256='e0101ece31a2cbe7b47e41d19741e34602e12e4a5565baea84a33a45f454ca24',
        tokenizer_sha256='0072bb6de61da87bd9ba5f4e6739b594cc3e60c62c75a59d25d499629a70088f')
    if any(state.get(k) != v for k, v in expected.items()) or 'optimizer' not in state:
        raise ValueError('Exact committed own code-instruction4000 training state required')
    token = source.parent / 'tokenizer.json'
    if file_hash(token) != expected['tokenizer_sha256']:
        raise ValueError('Original own trained tokenizer differs')
    before = state_digest(state['model'])
    payload = {k: v for k, v in state.items() if k != 'optimizer'}
    destination.mkdir()
    temporary = destination / 'model.pt.tmp'
    with temporary.open('xb') as f:
        torch.save(payload, f); f.flush(); os.fsync(f.fileno())
    if temporary.stat().st_size > maximum or shutil.disk_usage(ROOT).free < reserve + 64*1024**2:
        raise ValueError('Preserve inference artifact and remaining writer budgets')
    path = destination / 'model.pt'; os.replace(temporary, path)
    shutil.copyfile(token, destination / 'tokenizer.json')
    restored = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    if (state_digest(restored['model']) != before or state_digest(state['model']) != before
            or any(restored[k] != v for k, v in expected.items())
            or 'optimizer' in restored or set(restored) != set(payload)
            or file_hash(source) != original_digest):
        raise ValueError('Exact unchanged source/model and inference metadata roundtrip required')
    sources = [Path(__file__), ROOT / 'scripts/prepare_codenet_native_contract_pack_v2.py',
               ROOT / 'src/slm_lab/codenet_native_contracts_v1.py',
               ROOT / 'src/slm_lab/recurrent_ablation.py']
    receipt = dict(completed_at=datetime.now(timezone.utc).isoformat(),
        source_checkpoint=str(source), source_checkpoint_sha256=original_digest,
        source_checkpoint_bytes=source.stat().st_size, expected_metadata=expected,
        inference_checkpoint=str(path), inference_checkpoint_sha256=file_hash(path),
        inference_checkpoint_bytes=path.stat().st_size,
        original_and_restored_model_state_sha256=before, all_model_state_tensors_identical=True,
        excluded_fields=['optimizer'], source_checkpoint_modified=False,
        original_writer_reserve_bytes=26*1024**3, occupied_verified_writer=occupied,
        remaining_writer_reserve_bytes=reserve, source_files_sha256={str(p): file_hash(p) for p in sources},
        optimizer_updates=0, candidate_promoted=False, goal_achieved=False)
    atomic_json(destination / 'extraction.json', receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
