"""Inspect the original archive and retain bounded source metadata only."""
from collections import Counter
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile


class BoundedReader:
    def __init__(self, source, limit):
        self.source, self.limit, self.bytes = source, limit, 0

    def read(self, count):
        block = self.source.read(count)
        self.bytes += len(block)
        if self.bytes > self.limit:
            raise ValueError('Archive decompression exceeds the inspection budget')
        return block


def inspect_archive(archive_path, output, progress, *,
                    max_decompressed_bytes=128 * 1024 ** 3,
                    max_retained_bytes=3 * 1024 ** 3,
                    minimum_free_bytes=12 * 1024 ** 3):
    output = Path(output)
    output.mkdir(exist_ok=False)
    for name in ['metadata', 'problem_descriptions', 'derived']:
        (output / name).mkdir()
    section_counts = Counter()
    code_languages = Counter()
    code_problem_languages = Counter()
    metadata_languages = Counter()
    accepted_languages = Counter()
    statuses = Counter()
    retained = members = metadata_rows = description_files = metadata_files = 0
    derived_retained = derived_files = index_bytes = 0
    def index_record(stream, record):
        nonlocal index_bytes
        encoded = json.dumps(record) + '\n'
        index_bytes += len(encoded.encode())
        if retained + index_bytes > max_retained_bytes:
            raise ValueError('Retained metadata/index exceeds its total bound')
        stream.write(encoded)
    with (output / 'retained-members.jsonl').open('x') as journal, \
            (output / 'other-members.jsonl').open('x') as other, \
            gzip.open(archive_path, 'rb') as compressed:
        source = BoundedReader(compressed, max_decompressed_bytes)
        with tarfile.open(fileobj=source, mode='r|') as archive:
            while (member := archive.next()) is not None:
                # Python 3.11 otherwise caches every header even in stream mode.
                # Regular-member extraction below uses the supplied TarInfo directly.
                archive.members.clear()
                members += 1
                if members > 20_000_000:
                    raise ValueError('Archive member count exceeds the inspection budget')
                path = PurePosixPath(member.name)
                if len(member.name) > 4096 or path.is_absolute() or '..' in path.parts or not path.parts or path.parts[0] != 'Project_CodeNet':
                    raise ValueError('Unexpected archive root/path')
                if not (member.isdir() or member.isfile()):
                    raise ValueError('Archive links/devices are not admitted')
                section = path.parts[1] if len(path.parts) > 1 else 'root'
                if member.isfile():
                    section_counts[section] += 1
                    destination = None
                    if section == 'data':
                        if len(path.parts) != 5 or not re.fullmatch(r'p[0-9]{5}', path.parts[2]) or not re.fullmatch(r's[0-9]{9}\..+', path.name):
                            raise ValueError('Unexpected original submission path')
                        code_languages[path.parts[3]] += 1
                        code_problem_languages[(path.parts[2], path.parts[3])] += 1
                    elif section == 'metadata':
                        if len(path.parts) != 3 or not re.fullmatch(r'(?:p[0-9]{5}|problem_list)\.csv', path.name):
                            raise ValueError('Unexpected original metadata path')
                        if member.size > 128 * 1024 ** 2:
                            raise ValueError('Metadata member exceeds the per-file bound')
                        destination = output / 'metadata' / path.name
                    elif section == 'problem_descriptions':
                        if len(path.parts) != 3 or not re.fullmatch(r'p[0-9]{5}\.html', path.name) or member.size > 1024 ** 2:
                            raise ValueError('Unexpected/oversized original description')
                        destination = output / 'problem_descriptions' / path.name
                    elif section == 'README.md' and len(path.parts) == 2 and member.size < 1024 ** 2:
                        destination = output / 'archive-README.md'
                    elif section == 'derived':
                        index_record(other, {'member': member.name, 'bytes': member.size})
                        # Small original manifests aid subsequent cluster/sample inspection.
                        # Larger derived files remain available in the retained archive.
                        if derived_files < 8000 and member.size <= 1024 ** 2 and derived_retained + member.size <= 128 * 1024 ** 2:
                            relative = Path(*path.parts[2:])
                            destination = output / 'derived' / relative
                    else:
                        index_record(other, {'member': member.name, 'bytes': member.size})
                    if destination is not None:
                        if retained + index_bytes + member.size > max_retained_bytes:
                            raise ValueError('Retained source metadata exceeds its total bound')
                        if shutil.disk_usage(output).free < minimum_free_bytes + member.size:
                            raise ValueError('Preserve temporary model/archive reserve')
                        raw = archive.extractfile(member).read(member.size + 1)
                        if len(raw) != member.size:
                            raise ValueError('Incomplete original archive member')
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        with destination.open('xb') as stream:
                            stream.write(raw)
                        retained += len(raw)
                        index_record(journal, {'member': member.name, 'bytes': len(raw),
                            'sha256': hashlib.sha256(raw).hexdigest(),
                            'cache_relative_path': str(destination.relative_to(output))})
                        if section == 'derived':
                            derived_retained += len(raw)
                            derived_files += 1
                        elif section == 'problem_descriptions':
                            description_files += 1
                        elif section == 'metadata':
                            metadata_files += 1
                            if path.name != 'problem_list.csv':
                                reader = csv.DictReader(io.StringIO(raw.decode('utf-8-sig')))
                                if not {'submission_id', 'problem_id', 'language', 'status'} <= set(reader.fieldnames or []):
                                    raise ValueError('Original submission CSV fields differ')
                                for row in reader:
                                    if row['problem_id'] != path.stem or not re.fullmatch(r's[0-9]{9}', row['submission_id']):
                                        raise ValueError('Original CSV submission/problem identity differs')
                                    metadata_rows += 1
                                    statuses[row['status']] += 1
                                    metadata_languages[row['language']] += 1
                                    if row['status'] == 'Accepted':
                                        accepted_languages[row['language']] += 1
                if members % 50000 == 0:
                    journal.flush()
                    other.flush()
                    progress({'members': members, 'decompressed_bytes': source.bytes,
                              'retained_bytes': retained, 'description_files': description_files,
                              'metadata_files': metadata_files, 'metadata_rows': metadata_rows})
        # tar iteration can finish before the gzip trailer. Drain it to check CRC/size.
        while source.read(8 * 1024 ** 2):
            pass
    return {'members': members, 'decompressed_bytes': source.bytes,
        'retained_source_bytes': retained, 'section_regular_file_counts': dict(section_counts),
        'index_bytes': index_bytes, 'retained_derived_files': derived_files,
        'original_code_files_by_language': dict(code_languages),
        'original_code_problem_language_groups': len(code_problem_languages),
        'description_files': description_files, 'metadata_files': metadata_files,
        'metadata_submission_rows': metadata_rows, 'metadata_status_counts': dict(statuses),
        'metadata_submissions_by_language': dict(metadata_languages),
        'metadata_accepted_by_language': dict(accepted_languages),
        'code_header_and_metadata_row_counts_match': sum(code_languages.values()) == metadata_rows,
        'whole_gzip_stream_drained_crc_checked': True,
        'archive_paths_extracted_indiscriminately': False,
        'submission_code_executed': False, 'dataset_admitted_to_training': False}
