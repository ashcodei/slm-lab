"""Preserve the superseded 400/4754 reading snapshot in an exact byte archive."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil

import torch

from slm_lab.checkpoint_archive import chunks, equal_range, reconstruct, storage_ranges
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    checkpoint = ROOT / 'checkpoints/multimodal-pilot-v1/image-reading-original-box-prototype/step-00000400.pt'
    anchor = ROOT / 'checkpoints/multimodal-pilot-v1/instruction/model.pt'
    archive = checkpoint.with_name(checkpoint.name + '.archive')
    report = ROOT / 'reports/superseded-reading-snapshot-exact-archive-v1.json'
    if archive.exists() or report.exists() or checkpoint.stat().st_nlink != 1:
        raise ValueError('Require original unique superseded snapshot and fresh archive/report')
    protected = set()
    for path in (ROOT / 'checkpoints').rglob('manifest.json'):
        if not path.parent.name.endswith(('.xor', '.archive')):
            continue
        record = json.loads(path.read_text())
        protected.update(Path(record[key]).resolve() for key in ['anchor', 'source_checkpoint'] if record.get(key))
    if checkpoint.resolve() in protected:
        raise ValueError('Never remove a required full archive anchor')

    def no_readers():
        for process in Path('/proc').iterdir():
            if not process.name.isdigit() or int(process.name) == os.getpid():
                continue
            try:
                arguments = (process / 'cmdline').read_bytes().split(b'\0')
                for raw in arguments:
                    if not raw or len(raw) >= 4096:
                        continue
                    text = raw.decode()
                    if text.endswith('/train_multimodal_pilot.py') or Path(text).resolve() in {checkpoint, checkpoint.parent}:
                        raise ValueError('Superseded snapshot/producer still referenced by a live process')
                for descriptor in (process / 'fd').iterdir():
                    try:
                        if descriptor.resolve() == checkpoint:
                            raise ValueError('Superseded snapshot is open in another process')
                    except FileNotFoundError:
                        pass
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue

    no_readers()
    initial = checkpoint.stat(); original_sha = file_hash(checkpoint); anchor_sha = file_hash(anchor)
    snapshot = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    if (snapshot['stage'] != 'sft' or snapshot['trainable'] != 'vision'
            or snapshot['steps'] != 400 or snapshot['planned_steps'] != 4754
            or 'optimizer' not in snapshot or snapshot['world_size'] != 4
            or [row['step'] for row in snapshot['metrics']] != list(range(1, 401))):
        raise ValueError('Exact superseded optimizer snapshot budget differs')
    budget = {key: snapshot[key] for key in ['steps', 'planned_steps', 'stage', 'trainable', 'world_size', 'data_sha256']}
    del snapshot
    anchor_training = json.loads((anchor.parent / 'training.json').read_text())
    if 'completed' not in anchor_training['status'] or anchor_training['steps'] != anchor_training['planned_steps']:
        raise ValueError('Exact full source anchor is not completed')
    first, second = storage_ranges(anchor), storage_ranges(checkpoint)
    reuse = []
    with anchor.open('rb') as source, checkpoint.open('rb') as original:
        for key, (offset, length, crc) in second.items():
            prior = first.get(key)
            if prior is not None and prior[1:] == (length, crc) and equal_range(source, original, prior[0], offset, length):
                reuse.append((offset, length, prior[0]))
    if not reuse or sum(row[1] for row in reuse) < 2 * 1024**3:
        raise ValueError('Insufficient exact frozen-language reuse; leave original untouched')
    archive.mkdir()
    segments = []; position = 0; free_before = shutil.disk_usage(ROOT).free
    with checkpoint.open('rb') as original, (archive / 'payload.bin').open('xb') as payload:
        def store(start, length):
            if length <= 0:
                return
            offset = payload.tell()
            for block in chunks(original, start, length):
                payload.write(block)
            segments.append({'kind': 'payload', 'offset': offset, 'length': length})
        for offset, length, source_offset in sorted(reuse):
            if offset < position:
                raise ValueError('Overlapping archive ranges; original retained')
            store(position, offset - position)
            segments.append({'kind': 'source', 'offset': source_offset, 'length': length})
            position = offset + length
        store(position, initial.st_size - position)
        payload.flush(); os.fsync(payload.fileno())
    if file_hash(checkpoint) != original_sha or file_hash(anchor) != anchor_sha:
        raise ValueError('Snapshot or anchor changed; original retained')
    manifest = {'version': 1, 'created_at': datetime.now(timezone.utc).isoformat(),
        'original_checkpoint': str(checkpoint), 'original_sha256': original_sha,
        'original_bytes': initial.st_size, 'source_checkpoint': str(anchor), 'source_sha256': anchor_sha,
        'source_must_be_retained': True, 'payload_sha256': file_hash(archive / 'payload.bin'),
        'payload_bytes': (archive / 'payload.bin').stat().st_size, 'segments': segments,
        'reused_tensor_storages': len(reuse), 'all_reused_tensor_bytes_compared_exactly': True,
        'original_snapshot_budget': budget, 'source_training_completed': False,
        'original_removed': False, 'optimizer_and_rng_bytes_preserved': True,
        'scope': 'Superseded original-box-coordinate prototype, actual400 of4754 updates; not a completed training budget. Full optimizer/RNG/checkpoint bytes reconstruct exactly.',
        'limitation': 'Keep the full instruction anchor and payload; reconstruct to a fresh ordinary file before use.',
        'goal_achieved': False}
    atomic_json(archive / 'manifest.json', manifest)
    manifest['verification'] = reconstruct(archive)
    atomic_json(archive / 'manifest.json', manifest)
    no_readers()
    current = checkpoint.stat()
    if ((current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns)
            != (initial.st_dev, initial.st_ino, initial.st_size, initial.st_mtime_ns)
            or file_hash(checkpoint) != original_sha or file_hash(anchor) != anchor_sha):
        raise ValueError('Snapshot/anchor changed before removal; original retained')
    checkpoint.unlink(); manifest['original_removed'] = True
    atomic_json(archive / 'manifest.json', manifest)
    result = {key: value for key, value in manifest.items() if key != 'segments'}
    result.update(archive=str(archive), reclaimed_bytes=initial.st_size - manifest['payload_bytes'],
                  free_bytes_before=free_before, free_bytes_after=shutil.disk_usage(ROOT).free,
                  archive_source_sha256=file_hash(__file__), active_training_unchanged=True)
    atomic_json(report, result)
    print(json.dumps({'archive': str(archive), 'reclaimed_bytes': result['reclaimed_bytes'],
                      'actual_source_budget': budget, 'exact_original_bytes_verified': True}))


if __name__ == '__main__':
    main()
