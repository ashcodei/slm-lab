"""Archive three fixed completed vision optimizer snapshots byte for byte."""
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
COMPONENTS = {'character-sequence-pilot-v1': 2126,
              'learned-character-sequence-v1': 6378,
              'masked-visual-pretraining-v1': 5724}


def assert_completed(snapshot, final, plan, steps):
    if (snapshot.get('steps') != steps or final.get('steps') != steps
            or plan.get('steps') != steps or snapshot.get('plan') != plan or final.get('plan') != plan
            or 'optimizer' not in snapshot or 'optimizer' in final
            or set(snapshot['model']) != set(final['model'])):
        raise ValueError('Exact completed component budget/plan or optimizer state differs')
    for key in snapshot['model']:
        if not torch.equal(snapshot['model'][key], final['model'][key]):
            raise ValueError('Final model tensors differ from the completed optimizer snapshot')


def no_readers(path, anchor):
    relevant = {path.resolve(), path.parent.resolve()}
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid(): continue
        try:
            for raw in (proc / 'cmdline').read_bytes().split(b'\0'):
                if not 0 < len(raw) < 4096: continue
                try: candidate = Path(raw.decode()).resolve()
                except (OSError, ValueError, UnicodeError): continue
                if candidate in relevant: raise ValueError('Completed component is named by another live process')
            for fd in (proc / 'fd').iterdir():
                try:
                    if fd.resolve() == path: raise ValueError('Optimizer snapshot has another live reader')
                except FileNotFoundError: pass
            if str(path) in (proc / 'maps').read_text():
                raise ValueError('Optimizer snapshot is memory-mapped by another live process')
        except (FileNotFoundError, PermissionError, ProcessLookupError): continue
    protected = set()
    for manifest in (ROOT / 'checkpoints').rglob('manifest.json'):
        if not manifest.parent.name.endswith(('.archive', '.xor')): continue
        d = json.loads(manifest.read_text())
        for field in ['anchor', 'source_checkpoint']:
            if d.get(field): protected.add(Path(d[field]).resolve())
    if path.resolve() in protected or path.stat().st_nlink != 1:
        raise ValueError('Never remove a full archive anchor or a multiply linked snapshot')
    if anchor.stat().st_nlink < 1: raise ValueError('Required final-model anchor is unavailable')


def main():
    torch.set_num_threads(1)
    report = ROOT / 'reports/completed-component-snapshot-archives-v1.json'
    if report.exists(): raise ValueError('Inspect earlier exact archives; do not relaunch')
    records = []
    for name, steps in COMPONENTS.items():
        path = ROOT / 'checkpoints' / name / 'latest.pt'
        anchor = path.with_name('model.pt'); archive = path.with_name('latest.pt.archive')
        plan_path = ROOT / 'reports' / name / 'plan.json'
        if archive.exists(): raise ValueError('Preserve earlier component archive')
        no_readers(path, anchor)
        initial = path.stat(); original_sha = file_hash(path); anchor_sha = file_hash(anchor)
        plan_sha = file_hash(plan_path); plan = json.loads(plan_path.read_text())
        snapshot = torch.load(path, weights_only=True, map_location='cpu', mmap=True)
        final = torch.load(anchor, weights_only=True, map_location='cpu', mmap=True)
        assert_completed(snapshot, final, plan, steps)
        del snapshot, final
        first, second = storage_ranges(anchor), storage_ranges(path)
        reuse = []
        with path.open('rb') as original, anchor.open('rb') as source:
            for _, (offset, length, crc) in second.items():
                for _, (prior, size, checksum) in first.items():
                    if length == size and crc == checksum and equal_range(original, source, offset, prior, length):
                        reuse.append((offset, length, prior)); break
        if sum(length for _, length, _ in reuse) < 190 * 1024**2:
            raise ValueError('Insufficient exact final-weight reuse; original retained')
        archive.mkdir(); segments = []; position = 0
        with path.open('rb') as original, (archive / 'payload.bin').open('xb') as payload:
            def store(start, length):
                if not length: return
                offset = payload.tell()
                for block in chunks(original, start, length): payload.write(block)
                segments.append({'kind': 'payload', 'offset': offset, 'length': length})
            for offset, length, prior in sorted(reuse):
                if offset < position: raise ValueError('Overlapping archive ranges')
                store(position, offset - position)
                segments.append({'kind': 'source', 'offset': prior, 'length': length})
                position = offset + length
            store(position, initial.st_size - position)
            payload.flush(); os.fsync(payload.fileno())
        if file_hash(path) != original_sha or file_hash(anchor) != anchor_sha or file_hash(plan_path) != plan_sha:
            raise ValueError('Original optimizer/final weights/plan changed; original retained')
        d = {'version': 1, 'created_at': datetime.now(timezone.utc).isoformat(),
             'original_checkpoint': str(path), 'original_sha256': original_sha,
             'original_bytes': initial.st_size, 'source_checkpoint': str(anchor), 'source_sha256': anchor_sha,
             'source_must_be_retained': True, 'payload_sha256': file_hash(archive / 'payload.bin'),
             'payload_bytes': (archive / 'payload.bin').stat().st_size, 'segments': segments,
             'original_removed': False, 'optimizer_and_rng_bytes_preserved': True,
             'actual_completed_steps': steps, 'planned_steps': steps, 'original_plan_sha256': plan_sha,
             'all_final_model_tensors_checked_equal': True, 'goal_achieved': False,
             'scope': 'Fixed completed component optimizer snapshot; byte-exact archival, no new model learning.'}
        atomic_json(archive / 'manifest.json', d)
        d['verification'] = reconstruct(archive)
        atomic_json(archive / 'manifest.json', d)
        no_readers(path, anchor)
        current = path.stat()
        if ((current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns)
                != (initial.st_dev, initial.st_ino, initial.st_size, initial.st_mtime_ns)
                or file_hash(path) != original_sha or file_hash(anchor) != anchor_sha):
            raise ValueError('Snapshot or retained final weights changed before removal')
        free_before = shutil.disk_usage(ROOT).free
        path.unlink(); d['original_removed'] = True; atomic_json(archive / 'manifest.json', d)
        records.append({k: v for k, v in d.items() if k != 'segments'})
        records[-1].update(archive=str(archive), reclaimed_bytes=initial.st_size - d['payload_bytes'],
                           free_bytes_before_unlink=free_before, free_bytes_after_unlink=shutil.disk_usage(ROOT).free)
        print(json.dumps({'component': name, 'reclaimed_bytes': records[-1]['reclaimed_bytes'], 'whole_original_verified': True}), flush=True)
    atomic_json(report, {'verified_at': datetime.now(timezone.utc).isoformat(), 'archives': records,
        'reclaimed_bytes': sum(r['reclaimed_bytes'] for r in records), 'archive_script_sha256': file_hash(__file__),
        'all_full_final_model_anchors_retained': True, 'live_training_sources_changed': False, 'goal_achieved': False})


if __name__ == '__main__': main()
