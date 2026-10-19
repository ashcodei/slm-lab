"""Retain exact inactive repository-copy bytes in their unchanged aggregate files."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.source_segment_archive_v1 import archive_duplicate, no_readers, restore

ROOT = Path(__file__).resolve().parents[1]


def main():
    reports = ROOT / 'reports/public-code-source-copy-archives-v1'
    if reports.exists():
        raise ValueError('Inspect previous evidence before any continuation')
    protected = set(json.loads((ROOT / 'reports/nq-development-exposure-v1/audit.json').read_text())['source_files_sha256'])
    jobs, skipped, bound = [], [], {}
    for name in ['public-code-v1', 'public-code-extra-v1']:
        root = ROOT / 'data' / name
        manifest_path = root / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        anchor = root / 'all.jsonl'
        if file_hash(anchor) != manifest['all_sha256']:
            raise ValueError('Original aggregate bytes must remain exact')
        bound[str(anchor)] = manifest['all_sha256']
        bound[str(manifest_path)] = file_hash(manifest_path)
        data = anchor.read_bytes()
        for source in manifest['sources']:
            target = root / (source['name'] + '.jsonl')
            raw = target.read_bytes()
            if (hashlib.sha256(raw).hexdigest() != source['sha256']
                    or target.is_symlink() or target.stat().st_nlink != 1):
                raise ValueError('Exact unshared original repository bytes required')
            offset = data.find(raw[:min(1024, len(raw))])
            if offset < 0 or data[offset:offset + len(raw)] != raw or str(target) in protected:
                skipped.append(str(target))
                continue
            no_readers(target)
            jobs.append(dict(target=str(target), anchor=str(anchor), offset=offset,
                sha256=source['sha256'], anchor_sha256=manifest['all_sha256'], bytes=len(raw),
                archive_manifest=str(target.with_name(target.name + '.segment-v1.json'))))
        del data
    if len(jobs) != 19 or sum(row['bytes'] for row in jobs) != 345823295:
        raise ValueError('Exact reviewed nineteen-file duplicate inventory required')
    reports.mkdir()
    sources = [Path(__file__), ROOT / 'src/slm_lab/source_segment_archive_v1.py']
    bound.update({str(p): file_hash(p) for p in sources})
    atomic_json(reports / 'plan.json', dict(created_at=datetime.now(timezone.utc).isoformat(),
        jobs=jobs, skipped_noncontiguous_or_protected=skipped, retained_files_sha256=bound,
        every_original_byte_compared_before_any_removal=True,
        existing_training_data_images_weights_and_runtime_unchanged=True, goal_achieved=False))
    before = shutil.disk_usage(ROOT).free
    records = []
    for row in jobs:
        for path, expected in bound.items():
            if file_hash(path) != expected:
                raise ValueError('Retained source or archival implementation changed')
        result = archive_duplicate(row['target'], row['anchor'], row['offset'], row['archive_manifest'],
            row['sha256'], row['anchor_sha256'])
        records.append(dict(**result, archive_manifest=row['archive_manifest']))
        atomic_json(reports / 'status.json', dict(status='archiving inactive exact repository copies',
            updated_at=datetime.now(timezone.utc).isoformat(), completed=len(records), planned=len(jobs),
            goal_achieved=False))
    largest = max(records, key=lambda r: r['original_bytes'])
    with tempfile.TemporaryDirectory(prefix='slm-repository-copy-restore-') as scratch:
        output = Path(scratch) / 'original.jsonl'
        restored = restore(largest['archive_manifest'], output)
        if file_hash(output) != largest['original_sha256']:
            raise ValueError('Physical original-byte restoration differs')
        atomic_json(reports / 'physical-restore-check.json', dict(**restored,
            original_source=largest['original_path'], independently_rehashed=True, goal_achieved=False))
    atomic_json(reports / 'results.json', dict(completed_at=datetime.now(timezone.utc).isoformat(),
        archives=records, retained_files_sha256=bound, source_copy_count=len(records),
        original_bytes_retained=sum(r['original_bytes'] for r in records),
        reclaimed_allocated_bytes=sum(r['original_allocated_bytes'] for r in records),
        free_home_bytes_before=before, free_home_bytes_after=shutil.disk_usage(ROOT).free,
        every_original_byte_physically_present_in_retained_aggregates=True,
        original_preprocessing_requires_restore=True, model_capability_improved=False,
        full_future_training_capacity_funded=False, goal_achieved=False))
    atomic_json(reports / 'status.json', dict(status='nineteen inactive exact source copies archived',
        updated_at=datetime.now(timezone.utc).isoformat(), goal_achieved=False))
    print('exact_repository_source_copies_archived', len(records), flush=True)


if __name__ == '__main__':
    main()
