"""Full opaque-byte round trips, corruption and quotas on separate filesystems."""
import json
import os
from pathlib import Path
import tempfile
import zipfile

import pytest

from slm_lab.exact_checkpoint_raw_anchor_v4 import archive, restore, sha

ROOT = Path(__file__).resolve().parents[1]


def inputs(work):
    anchor, target = work / 'anchor.pt', work / 'target.pt'
    original = os.urandom(2*1024**2); changed = bytearray(original)
    for i in range(0, len(changed), 4096): changed[i] ^= 47
    for path, prefix, tensor in [(anchor, 'initial', original), (target, 'later', bytes(changed))]:
        with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_STORED) as z:
            z.writestr(prefix+'/data.pkl', b'opaque optimizer RNG metadata: '+prefix.encode())
            z.writestr(prefix+'/data/0', tensor)
            z.writestr(prefix+'/data/1', bytes(512*1024))
            if path == target: z.writestr(prefix+'/data/2', os.urandom(65536))
            z.writestr(prefix+'/version', b'3\n')
    return anchor, target


def pack(target, anchor, destination, scratch, **overrides):
    options = dict(expected_sha256=sha(target), expected_anchor_sha256=sha(anchor),
        maximum_payload_bytes=4*1024**2, minimum_free_bytes=0,
        minimum_reclaimed_bytes=1024**2, guard=lambda: None, retire_original=False)
    options.update(overrides)
    return archive(target, anchor, destination, scratch, **options)


def test_durable_raw_anchor_alone_restores_every_original_byte_after_retirement():
    with tempfile.TemporaryDirectory(dir=ROOT/'reports') as w, tempfile.TemporaryDirectory(dir='/tmp') as s:
        work, scratch = Path(w), Path(s); anchor, target = inputs(work)
        content, before = target.read_bytes(), anchor.read_bytes(); destination = work/'archive'
        result = pack(target, anchor, destination, scratch, retire_original=True)
        assert result['physical_restore_checked'] and result['original_retired'] and not target.exists()
        assert result['physical_restore']['independent_physical_file_rehash_passed']
        assert result['payload_bytes'] < len(content)//4
        assert result['reclaimed_allocated_bytes_if_retired'] > 1024**2
        assert {'raw', 'xor'} == {x['kind'] for x in result['segments']}
        restored = scratch/'new-original.pt'; check = restore(destination, restored)
        assert restored.read_bytes() == content and check['actual_restored_sha256'] == sha(restored)
        assert anchor.read_bytes() == before and result['anchor_must_be_retained']
        assert set(p.name for p in destination.iterdir()) == {'manifest.json', 'payload.zst'}


def test_changed_anchor_payload_and_existing_output_reject_without_publishing():
    with tempfile.TemporaryDirectory(dir=ROOT/'reports') as w, tempfile.TemporaryDirectory(dir='/tmp') as s:
        work, scratch = Path(w), Path(s); anchor, target = inputs(work); digest = sha(target)
        destination = work/'archive'; pack(target, anchor, destination, scratch)
        existing = scratch/'existing.pt'; existing.write_bytes(b'preserve')
        with pytest.raises(ValueError): restore(destination, existing)
        assert existing.read_bytes() == b'preserve'
        original_anchor = anchor.read_bytes(); anchor.write_bytes(original_anchor[:-1]+b'X')
        with pytest.raises(ValueError): restore(destination, scratch/'bad-anchor.pt')
        anchor.write_bytes(original_anchor)
        payload = destination/'payload.zst'; data = bytearray(payload.read_bytes()); data[len(data)//2] ^= 1; payload.write_bytes(data)
        with pytest.raises(ValueError): restore(destination, scratch/'bad-payload.pt')
        assert not (scratch/'bad-anchor.pt').exists() and not (scratch/'bad-payload.pt').exists()
        assert sha(target) == digest


def test_invalid_segment_bounds_and_decoded_length_remove_partial_physical_output():
    with tempfile.TemporaryDirectory(dir=ROOT/'reports') as w, tempfile.TemporaryDirectory(dir='/tmp') as s:
        work, scratch = Path(w), Path(s); anchor, target = inputs(work)
        destination = work/'archive'; pack(target, anchor, destination, scratch)
        p = destination/'manifest.json'; original = json.loads(p.read_text())
        for mutation in ['offset', 'anchor_range', 'expanded_length']:
            data = json.loads(json.dumps(original))
            part = next(x for x in data['segments'] if x['kind'] == 'xor')
            if mutation == 'offset': part['target_offset'] += 1
            elif mutation == 'anchor_range': part['anchor_offset'] = data['retained_raw_anchor']['bytes']
            else: part['length'] -= 1
            p.write_text(json.dumps(data)); output = scratch/(mutation+'.pt')
            with pytest.raises(ValueError): restore(destination, output)
            assert not output.exists() and not output.with_name(output.name+'.restoring').exists()


def test_payload_quota_savings_and_same_filesystem_rejections_keep_exact_original():
    with tempfile.TemporaryDirectory(dir=ROOT/'reports') as w, tempfile.TemporaryDirectory(dir='/tmp') as s:
        work, scratch = Path(w), Path(s); anchor, target = inputs(work); digest = sha(target)
        with pytest.raises(ValueError, match='reserve'):
            pack(target, anchor, work/'quota', scratch, maximum_payload_bytes=1, retire_original=True)
        with pytest.raises(ValueError, match='independent full physical scratch'):
            pack(target, anchor, work/'same-filesystem', work, retire_original=True)
        with pytest.raises(ValueError, match='original retained'):
            pack(target, anchor, work/'no-savings', scratch, minimum_reclaimed_bytes=8*1024**2, retire_original=True)
        assert sha(target) == digest and anchor.exists()
