"""Lossless completed-checkpoint archives reusing exact bytes in a retained anchor."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import struct
import zipfile

from .corpus import file_hash
from .diverse_images import atomic_json

BLOCK = 4 * 1024**2


def storage_ranges(path):
    with zipfile.ZipFile(path) as archive, Path(path).open('rb') as raw:
        result = {}
        for item in archive.infolist():
            parts = item.filename.split('/')
            if len(parts) != 3 or parts[-2] != 'data' or not parts[-1].isdigit():
                continue
            if item.compress_type != zipfile.ZIP_STORED:
                raise ValueError('Only uncompressed own PyTorch tensor archives are supported')
            raw.seek(item.header_offset)
            header = raw.read(30)
            if header[:4] != b'PK\x03\x04':
                raise ValueError('Invalid tensor entry header')
            name_size, extra_size = struct.unpack_from('<HH', header, 26)
            start = item.header_offset + 30 + name_size + extra_size
            if start + item.file_size > Path(path).stat().st_size:
                raise ValueError('Tensor entry exceeds checkpoint')
            result[parts[-1]] = (start, item.file_size, item.CRC)
        if not result:
            raise ValueError('Expected a PyTorch tensor checkpoint')
        return result


def chunks(stream, start, length):
    stream.seek(start)
    while length:
        chunk = stream.read(min(length, BLOCK))
        if not chunk:
            raise ValueError('Truncated checkpoint or archive segment')
        yield chunk
        length -= len(chunk)


def equal_range(first, second, a, b, length):
    # CRC is only a candidate filter. Actual byte equality authorizes reuse.
    return all(x == y for x, y in zip(chunks(first, a, length), chunks(second, b, length), strict=True))


def reconstruct(archive, target=None):
    archive = Path(archive).resolve()
    manifest = json.loads((archive / 'manifest.json').read_text())
    source = Path(manifest['source_checkpoint']).resolve()
    payload = archive / 'payload.bin'
    if file_hash(source) != manifest['source_sha256'] or file_hash(payload) != manifest['payload_sha256']:
        raise ValueError('Archive anchor or payload changed')
    output = None
    temporary = None
    if target is not None:
        target = Path(target).resolve()
        if target.exists() or target in {source, payload} or target.is_relative_to(archive):
            raise ValueError('Restore requires a fresh destination outside the archive')
        temporary = target.with_name(target.name + '.restoring')
        if temporary.exists():
            raise ValueError('Interrupted restore exists; inspect it before continuing')
        output = temporary.open('xb')
    digest = hashlib.sha256(); size = 0
    try:
        with source.open('rb') as anchor, payload.open('rb') as stored:
            for segment in manifest['segments']:
                if segment['kind'] not in {'source', 'payload'} or segment['offset'] < 0 or segment['length'] < 1:
                    raise ValueError('Invalid archive segment')
                stream = anchor if segment['kind'] == 'source' else stored
                for chunk in chunks(stream, segment['offset'], segment['length']):
                    digest.update(chunk); size += len(chunk)
                    if output is not None: output.write(chunk)
        if size != manifest['original_bytes'] or digest.hexdigest() != manifest['original_sha256']:
            raise ValueError('Restored checkpoint is not byte-identical to the original')
        if output is not None:
            output.flush(); os.fsync(output.fileno()); output.close(); output = None
            # Avoid overwriting a concurrently created file.
            os.link(temporary, target); temporary.unlink()
        return {'restored_sha256': digest.hexdigest(), 'restored_bytes': size,
                'exact_original_bytes_verified': True}
    finally:
        if output is not None: output.close()
        if temporary is not None and temporary.exists(): temporary.unlink()


def archive_completed(checkpoint, source, archive, remove_original=False, preparation=None):
    checkpoint, source, archive = map(lambda p: Path(p).resolve(), [checkpoint, source, archive])
    if checkpoint == source or archive.exists() or checkpoint.stat().st_ino == source.stat().st_ino:
        raise ValueError('Use distinct completed checkpoint/anchor and a fresh archive directory')
    prepared_record = None
    if preparation is not None:
        preparation = Path(preparation).resolve(); receipt = json.loads(preparation.read_text())
        matching = [record for record in receipt.get('branches', {}).values()
                    if Path(record['checkpoint']).resolve() == checkpoint]
        if len(matching) != 1 or matching[0]['checkpoint_sha256'] != file_hash(checkpoint):
            raise ValueError('Prepared snapshot does not match its exact preparation receipt')
        import torch
        snapshot = torch.load(checkpoint, weights_only=True, map_location='cpu', mmap=True)
        if ('optimizer' in snapshot or 'reference_model' in snapshot or 'model' not in snapshot
                or any(not torch.isfinite(value).all() for value in snapshot['model'].values())):
            raise ValueError('Prepared archive requires finite own weights-only snapshot')
        del snapshot
        prepared_record = {'receipt': str(preparation), 'sha256': file_hash(preparation),
                           'scope': 'Prepared immutable weights snapshot; does not claim its source training budget completed.'}
    for path in ([source] if prepared_record is not None else [checkpoint, source]):
        report = json.loads((path.parent / 'training.json').read_text())
        if 'completed' not in report.get('status', '') or report['steps'] != report['planned_steps']:
            raise ValueError('Only completed experimental checkpoints may be archived')
    for process in Path('/proc').iterdir():
        # The archive CLI necessarily names its own input. Other live readers
        # still prevent archival/removal; this process is the authorized writer.
        if not process.name.isdigit() or int(process.name) == os.getpid(): continue
        try: arguments = (process / 'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, PermissionError, ProcessLookupError): continue
        for raw in arguments:
            if not 0 < len(raw) < 4096: continue
            try: path = Path(raw.decode()).resolve()
            except (OSError, UnicodeError, ValueError): continue
            if path in {checkpoint, checkpoint.parent}:
                raise ValueError('Experimental checkpoint/output directory referenced by a live process')
    original_hash, source_hash = file_hash(checkpoint), file_hash(source)
    first_ranges, second_ranges = storage_ranges(source), storage_ranges(checkpoint)
    reuse = []
    with source.open('rb') as first, checkpoint.open('rb') as second:
        for key, (offset, length, crc) in second_ranges.items():
            candidate = first_ranges.get(key)
            if candidate is None or candidate[1:] != (length, crc): continue
            if equal_range(first, second, candidate[0], offset, length):
                reuse.append((offset, length, candidate[0]))
    if not reuse: raise ValueError('No exact common tensor bytes; leave the original intact')
    archive.mkdir(parents=True)
    segments = []; position = 0
    with checkpoint.open('rb') as original, (archive / 'payload.bin').open('xb') as payload:
        def store(start, length):
            if not length: return
            offset = payload.tell()
            for chunk in chunks(original, start, length): payload.write(chunk)
            segments.append({'kind': 'payload', 'offset': offset, 'length': length})
        for offset, length, shared_offset in sorted(reuse):
            if offset < position: raise ValueError('Overlapping tensor entries')
            store(position, offset - position)
            segments.append({'kind': 'source', 'offset': shared_offset, 'length': length})
            position = offset + length
        store(position, checkpoint.stat().st_size - position)
    if file_hash(checkpoint) != original_hash or file_hash(source) != source_hash:
        raise ValueError('Checkpoint changed while constructing archive; original retained')
    manifest = {'version': 1, 'created_at': datetime.now(timezone.utc).isoformat(),
                'original_checkpoint': str(checkpoint), 'original_sha256': original_hash,
                'original_bytes': checkpoint.stat().st_size, 'source_checkpoint': str(source),
                'source_sha256': source_hash, 'source_must_be_retained': True,
                'payload_sha256': file_hash(archive / 'payload.bin'),
                'payload_bytes': (archive / 'payload.bin').stat().st_size,
                'segments': segments, 'reused_tensor_storages': len(reuse),
                'all_reused_tensor_bytes_compared_exactly': True, 'original_removed': False,
                'prepared_snapshot_receipt': prepared_record,
                'goal_achieved': False,
                'limitation': 'Restore this archived research checkpoint before using its original model.pt path. Retain the exact source anchor and payload.'}
    atomic_json(archive / 'manifest.json', manifest)
    verification = reconstruct(archive)
    manifest['verification'] = verification
    atomic_json(archive / 'manifest.json', manifest)
    if remove_original:
        if file_hash(checkpoint) != original_hash or file_hash(source) != source_hash:
            raise ValueError('Checkpoint changed before archival; original retained')
        checkpoint.unlink(); manifest['original_removed'] = True
        atomic_json(archive / 'manifest.json', manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    create = sub.add_parser('archive')
    create.add_argument('--checkpoint', required=True); create.add_argument('--source', required=True)
    create.add_argument('--archive', required=True); create.add_argument('--remove-original', action='store_true')
    create.add_argument('--preparation', help='Exact own preparation receipt for an immutable weights-only snapshot')
    restore = sub.add_parser('restore')
    restore.add_argument('--archive', required=True); restore.add_argument('--output', required=True)
    verify = sub.add_parser('verify'); verify.add_argument('--archive', required=True)
    args = parser.parse_args()
    if args.command == 'archive':
        result = archive_completed(args.checkpoint, args.source, args.archive, args.remove_original, args.preparation)
        result = {key: value for key, value in result.items() if key != 'segments'}
    else:
        result = reconstruct(args.archive, args.output if args.command == 'restore' else None)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
