"""Inventory original APPS roles; retain only original publisher-train files.

No code execution, model updates, training admission or train/test mixing.
The immutable original archive preserves every test byte separately.
"""
from collections import Counter
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/public-apps-publisher-source-v1'
OUTPUT = ROOT / 'data/public-apps-original-inventory-v1'
KNOWN = {'solutions.json', 'question.txt', 'input_output.json', 'metadata.json', 'starter_code.py'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-existing-extraction', action='store_true')
    args = parser.parse_args()
    existing = OUTPUT.exists()
    if existing and (not args.audit_existing_extraction or (OUTPUT / 'inventory.json').exists()):
        raise ValueError('Preserve completed original inventory; explicit audit-only recovery of existing extraction required')
    collected = json.loads((SOURCE / 'collection.json').read_text())
    if collected['status'] != 'original public archive retained; not prepared or admitted':
        raise ValueError('Completed original archive transfer required')
    for name, receipt in collected['collected_files'].items():
        if file_hash(SOURCE / name) != receipt['sha256']: raise ValueError('Retained original publisher bytes differ')
    if shutil.disk_usage(ROOT).free < 34 * 1024**3:
        raise ValueError('Fund at most8GiB original train extraction above26GiB reserve')
    if not existing:
        OUTPUT.mkdir(); (OUTPUT / 'original-train').mkdir()
    catalog_name = 'members-recheck.jsonl' if existing else 'members.jsonl'
    problems = {'train': {}, 'test': {}}; names = set(); total = train_bytes = 0; files = []
    with tarfile.open(SOURCE / 'APPS.tar.gz', 'r|gz') as archive, (OUTPUT / catalog_name).open('x') as catalog:
        for member in archive:
            p = PurePosixPath(member.name)
            if p.is_absolute() or '..' in p.parts or member.name in names or member.issym() or member.islnk():
                raise ValueError('Unique original relative regular-file/directory archive paths required')
            names.add(member.name)
            if member.isdir(): continue
            if not member.isfile(): raise ValueError('Unexpected original archive member type')
            total += member.size
            if total > 16 * 1024**3: raise ValueError('Bounded16GiB decompressed original inventory exceeded')
            parts = p.parts
            role = parts[1] if len(parts) == 4 and parts[0] == 'APPS' and parts[1] in problems else None
            problem = parts[2] if role else None
            if role and (not problem.isdigit() or parts[3] not in KNOWN):
                raise ValueError('Unexpected original problem path/schema; preserve archive for review')
            target = None
            if role == 'train':
                train_bytes += member.size
                if train_bytes > 8 * 1024**3: raise ValueError('Bounded8GiB original train extraction exceeded')
                target = OUTPUT / 'original-train' / problem / parts[3]
                if not existing: target.parent.mkdir(exist_ok=True)
            source = archive.extractfile(member); digest = hashlib.sha256(); size = 0; retained = []
            inspect = role is not None and parts[3] in {'metadata.json', 'question.txt'}
            if inspect and member.size > 4 * 1024**2: raise ValueError('Unexpected original problem metadata size')
            stream = target.open('xb') if target and not existing else None
            try:
                while True:
                    chunk = source.read(1024**2)
                    if not chunk: break
                    size += len(chunk); digest.update(chunk)
                    if stream: stream.write(chunk)
                    if inspect: retained.append(chunk)
            finally:
                if stream: stream.close()
                source.close()
            if size != member.size: raise ValueError('Incomplete original archive member')
            if target and existing and (target.stat().st_size != size or file_hash(target) != digest.hexdigest()):
                raise ValueError('Every previously extracted training byte must match the actual original archive')
            record = dict(member=member.name, bytes=size, sha256=digest.hexdigest(), publisher_role=role,
                problem=problem, extracted_train_relative=str(target.relative_to(OUTPUT)) if target else None)
            catalog.write(json.dumps(record, separators=(',', ':')) + '\n'); files.append(record)
            if role:
                entry = problems[role].setdefault(problem, dict(files={}))
                entry['files'][parts[3]] = dict(bytes=size, sha256=digest.hexdigest())
                if inspect:
                    raw = b''.join(retained)
                    if parts[3] == 'metadata.json': entry['metadata'] = json.loads(raw)
                    else:
                        normalized = ' '.join(raw.decode('utf-8').split()).casefold()
                        entry['normalized_question_sha256'] = hashlib.sha256(normalized.encode()).hexdigest()
            elif member.name == 'APPS/README.txt':
                # Retain the exact archive declaration using a later random
                # read; all bytes are already kept in the original archive.
                pass
            if len(files) % 5000 == 0:
                print('original_APPS_members_inventoried', len(files), flush=True)
                if shutil.disk_usage(ROOT).free < 26 * 1024**3: raise ValueError('Retain original disk reserve')
    required = {'question.txt', 'metadata.json'}
    if (any(len(v) != 5000 for v in problems.values())
            or any(not required <= set(e['files']) for v in problems.values() for e in v.values())
            or any('solutions.json' not in e['files'] for e in problems['train'].values())):
        raise ValueError('All5000original train/test problems, original questions/metadata and every train solution file required')
    # The publisher archive itself omits195train test files and1235test
    # solution files. Record those source qualifications; invent neither.
    missing = {role: {field: [i for i, e in values.items() if field not in e['files']]
        for field in ['input_output.json', 'solutions.json', 'starter_code.py']} for role, values in problems.items()}
    if existing and file_hash(OUTPUT / 'members.jsonl') != file_hash(OUTPUT / catalog_name):
        raise ValueError('Independent complete archive/member catalog must reproduce the preserved first extraction exactly')
    test_questions = {e['normalized_question_sha256'] for e in problems['test'].values()}
    test_urls = {e['metadata']['url'] for e in problems['test'].values() if e['metadata'].get('url')}
    overlaps = dict(normalized_question_train_ids=[i for i, e in problems['train'].items() if e['normalized_question_sha256'] in test_questions],
        source_url_train_ids=[i for i, e in problems['train'].items() if e['metadata'].get('url') in test_urls])
    for role, values in problems.items(): atomic_json(OUTPUT / (role + '-problems.json'), values)
    report = dict(completed_at=datetime.now(timezone.utc).isoformat(), original_archive_sha256=collected['collected_files']['APPS.tar.gz']['sha256'],
        original_publisher_revision=collected['publisher_revision'], original_regular_members=len(files),
        original_uncompressed_bytes=total, retained_original_train_bytes=train_bytes,
        publisher_problem_counts={k: len(v) for k, v in problems.items()},
        train_difficulties=dict(Counter(e['metadata']['difficulty'] for e in problems['train'].values())),
        publisher_train_test_overlap=overlaps, test_files_extracted_for_training=0,
        original_source_missing_optional_files=missing,
        independent_second_archive_pass_reconciles_every_member_and_extracted_train_byte=existing,
        extracted_train_bytes_match_original_member_hashes=True,
        files_sha256={n: file_hash(OUTPUT / n) for n in set(['members.jsonl', catalog_name, 'train-problems.json', 'test-problems.json'])},
        preparation_pending=True, code_executed=False, source_admitted_to_training=False, model_updates=0, goal_achieved=False,
        qualifications=['Inventory alone does not establish original solution correctness or rights beyond actual publisher/source declarations.',
            'Before training, quarantine any publisher-test overlap and compare actual prepared training inputs with existing held sources.',
            'Keep original full prompt/starter/solution/test schemas; own-BPE over-window or unsupported Python/runtime cases require explicit reviews.'])
    atomic_json(OUTPUT / 'inventory.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__': main()
