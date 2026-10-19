"""Copy selected original source bytes and explicit auxiliary members; never execute."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile

from .codenet_archive_inspection_v1 import BoundedReader


def copy_bodies(archive_path, output, selected, auxiliary, progress, *,
                maximum_retained_bytes=1024 ** 3, maximum_body_bytes=1024 ** 2,
                maximum_decompressed_bytes=128 * 1024 ** 3,
                minimum_free_bytes=12 * 1024 ** 3):
    if not selected or set(selected) & set(auxiliary):
        raise ValueError('Require distinct selected source and auxiliary members')
    output = Path(output)
    output.mkdir(exist_ok=False)
    found = set()
    copied_auxiliary = {}
    retained = members = original_code_headers = mismatched_sizes = invalid_utf8 = nul_bodies = 0
    languages = Counter()
    body_digest = hashlib.sha256()
    def reserve(count):
        nonlocal retained
        if retained + count > maximum_retained_bytes:
            raise ValueError('Selected source store exceeds the retained-byte bound')
        if shutil.disk_usage(output).free < minimum_free_bytes + count:
            raise ValueError('Preserve temporary model/archive reserve')
        retained += count
    with (output / 'source-bodies.bin').open('xb') as bodies, \
            (output / 'source-index.jsonl').open('x') as index, \
            gzip.open(archive_path, 'rb') as compressed:
        source = BoundedReader(compressed, maximum_decompressed_bytes)
        with tarfile.open(fileobj=source, mode='r|') as archive:
            while (member := archive.next()) is not None:
                archive.members.clear()
                members += 1
                if members > 20_000_000:
                    raise ValueError('Archive member count exceeds the source bound')
                path = PurePosixPath(member.name)
                if len(member.name) > 4096 or path.is_absolute() or '..' in path.parts or not path.parts or path.parts[0] != 'Project_CodeNet':
                    raise ValueError('Unexpected original archive path')
                if not (member.isdir() or member.isfile()):
                    raise ValueError('Original archive links/devices are not admitted')
                if member.isfile() and len(path.parts) > 1 and path.parts[1] == 'data':
                    original_code_headers += 1
                if member.name in selected:
                    if member.name in found or not member.isfile() or len(path.parts) != 5 or path.parts[1] != 'data':
                        raise ValueError('Duplicate/non-regular selected source member')
                    if member.size > maximum_body_bytes:
                        raise ValueError('Original selected source body exceeds its byte bound')
                    raw = archive.extractfile(member).read(member.size + 1)
                    if len(raw) != member.size:
                        raise ValueError('Incomplete original source body')
                    reserve(len(raw))
                    offset = bodies.tell()
                    if bodies.write(raw) != len(raw):
                        raise ValueError('Incomplete selected source write')
                    body_digest.update(raw)
                    valid_utf8 = True
                    try:
                        raw.decode('utf-8')
                    except UnicodeDecodeError:
                        valid_utf8 = False
                    declared = selected[member.name]
                    record = {'archive_member': member.name, 'body_offset': offset,
                        'body_bytes': len(raw), 'body_sha256': hashlib.sha256(raw).hexdigest(),
                        'publisher_declared_code_bytes': declared,
                        'publisher_declared_size_matches': declared == len(raw),
                        'utf8_valid': valid_utf8, 'contains_nul': b'\0' in raw}
                    encoded = json.dumps(record) + '\n'
                    reserve(len(encoded.encode()))
                    index.write(encoded)
                    found.add(member.name)
                    languages[path.parts[3]] += 1
                    mismatched_sizes += declared != len(raw)
                    invalid_utf8 += not valid_utf8
                    nul_bodies += b'\0' in raw
                elif member.name in auxiliary:
                    if member.name in copied_auxiliary or not member.isfile() or member.size != auxiliary[member.name]:
                        raise ValueError('Duplicate/changed auxiliary member identity or bytes')
                    destination = output / 'auxiliary' / Path(*path.parts[2:])
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    digest = hashlib.sha256()
                    count = 0
                    with destination.open('xb') as stream, archive.extractfile(member) as original:
                        while block := original.read(8 * 1024 ** 2):
                            reserve(len(block))
                            if stream.write(block) != len(block):
                                raise ValueError('Incomplete auxiliary source write')
                            digest.update(block)
                            count += len(block)
                    if count != member.size:
                        raise ValueError('Incomplete original auxiliary member')
                    copied_auxiliary[member.name] = {'cache_relative_path': str(destination.relative_to(output)),
                        'bytes': count, 'sha256': digest.hexdigest()}
                if members % 50000 == 0:
                    bodies.flush()
                    index.flush()
                    progress({'members': members, 'decompressed_bytes': source.bytes,
                        'copied_source_bodies': len(found), 'expected_source_bodies': len(selected),
                        'copied_auxiliary_members': len(copied_auxiliary), 'retained_bytes': retained})
        # Completion requires the entire archive, even when chosen members occur early.
        while source.read(8 * 1024 ** 2):
            pass
    if found != set(selected) or set(copied_auxiliary) != set(auxiliary):
        raise ValueError(f'Original source members missing: {len(set(selected)-found)} bodies, {len(set(auxiliary)-set(copied_auxiliary))} auxiliary')
    return {'members': members, 'decompressed_bytes': source.bytes,
        'original_code_headers': original_code_headers, 'source_bodies': len(found),
        'source_bodies_by_language': dict(languages), 'source_body_store_sha256': body_digest.hexdigest(),
        'retained_bytes': retained, 'publisher_declared_size_mismatches': mismatched_sizes,
        'non_utf8_bodies': invalid_utf8, 'bodies_with_nul': nul_bodies,
        'auxiliary_members': copied_auxiliary, 'whole_gzip_stream_drained_crc_checked': True,
        'source_code_executed': False, 'split_assignment_completed': False,
        'dataset_admitted_to_training': False, 'model_training_started': False}
