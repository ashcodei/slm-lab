"""Opaque ZIP bytes on real separate filesystems; no tensor deserialization."""
import json
import os
from pathlib import Path
import tempfile
import zipfile

import pytest
import zstandard

from slm_lab.exact_checkpoint_zstd_archive_v2 import archive as retain_anchor
from slm_lab.exact_checkpoint_xor_zstd_archive_v3 import archive,restore,sha,segments

ROOT=Path(__file__).resolve().parents[1]


def inputs(work,scratch):
    anchor=work/'original-anchor.pt';target=work/'original-target.pt'
    values=os.urandom(2*1024**2);changed=bytearray(values)
    for position in range(0,len(changed),4096):changed[position]^=0x43
    def write(path,prefix,first,extra):
        with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_STORED) as z:
            z.writestr(prefix+'/data.pkl',b'opaque metadata optimizer state and private RNG\0'+extra)
            z.writestr(prefix+'/data/0',first)
            z.writestr(prefix+'/data/1',b'\0'*(512*1024))
            z.writestr(prefix+'/data/2',b'raw unequal-sized storage '+extra)
            if extra:z.writestr(prefix+'/data/3',os.urandom(65536))
            z.writestr(prefix+'/version',b'3\n')
    write(anchor,'earlier',values,b'');write(target,'different-prefix',bytes(changed),b'actual later metadata')
    retained=work/'retained-anchor'
    retain_anchor(anchor,retained,scratch,expected_sha256=sha(anchor),maximum_payload_bytes=4*1024**2,
        minimum_free_bytes=0,minimum_reclaimed_bytes=1,guard=lambda:None,retire_original=True)
    assert not anchor.exists()
    return retained,target


def pack(target,anchor,destination,scratch,**kwargs):
    return archive(target,anchor,destination,scratch,expected_sha256=sha(target),maximum_payload_bytes=4*1024**2,
        minimum_free_bytes=0,minimum_reclaimed_bytes=1024**2,guard=lambda:None,**kwargs)


def test_actual_compressed_anchor_and_target_physically_restore_every_original_byte_before_retirement():
    with tempfile.TemporaryDirectory(dir=ROOT/'reports') as w,tempfile.TemporaryDirectory(dir='/tmp') as s:
        work,scratch=Path(w),Path(s);anchor,target=inputs(work,scratch)
        content=target.read_bytes();digest=sha(target)
        before={p.name:sha(p) for p in anchor.iterdir()}
        result=pack(target,anchor,work/'target-archive',scratch,retire_original=True)
        assert not target.exists() and result['original_retired'] and result['physical_restore_checked']
        assert result['physical_restore']['retained_compressed_anchor_also_physically_restored']
        assert result['physical_restore']['independent_physical_file_rehash_passed']
        assert result['payload_bytes']<len(content)//4 and result['reclaimed_allocated_bytes_if_retired']>1024**2
        assert any(p['kind']=='raw' for p in result['segments']) and any(p['kind']=='xor' for p in result['segments'])
        check=restore(work/'target-archive',scratch/'fresh-independent-reader.pt',scratch)
        assert check['actual_restored_sha256']==digest and (scratch/'fresh-independent-reader.pt').read_bytes()==content
        assert {p.name:sha(p) for p in anchor.iterdir()}==before


def test_corrupt_payload_changed_anchor_and_existing_output_never_replace_original_or_output():
    with tempfile.TemporaryDirectory(dir=ROOT/'reports') as w,tempfile.TemporaryDirectory(dir='/tmp') as s:
        work,scratch=Path(w),Path(s);anchor,target=inputs(work,scratch);digest=sha(target)
        pack(target,anchor,work/'target-archive',scratch)
        existing=scratch/'existing.pt';existing.write_bytes(b'keep this output')
        with pytest.raises(ValueError):restore(work/'target-archive',existing,scratch)
        assert existing.read_bytes()==b'keep this output'
        payload=work/'target-archive/payload.zst';original=payload.read_bytes();changed=bytearray(original);changed[len(changed)//2]^=1;payload.write_bytes(changed)
        with pytest.raises(ValueError):restore(work/'target-archive',scratch/'corrupt.pt',scratch)
        payload.write_bytes(original)
        manifest=anchor/'manifest.json';manifest.write_text(manifest.read_text()+'\n')
        with pytest.raises(ValueError):restore(work/'target-archive',scratch/'changed-anchor.pt',scratch)
        assert sha(target)==digest and not (scratch/'corrupt.pt').exists() and not (scratch/'changed-anchor.pt').exists()


def test_payload_limit_shared_scratch_and_insufficient_saving_preserve_complete_original():
    with tempfile.TemporaryDirectory(dir=ROOT/'reports') as w,tempfile.TemporaryDirectory(dir='/tmp') as s:
        work,scratch=Path(w),Path(s);anchor,target=inputs(work,scratch);digest=sha(target)
        with pytest.raises(ValueError,match='reserve'):
            archive(target,anchor,work/'overflow',scratch,expected_sha256=digest,maximum_payload_bytes=1,
                minimum_free_bytes=0,minimum_reclaimed_bytes=1,guard=lambda:None,retire_original=True)
        with pytest.raises(ValueError,match='independent physical scratch'):
            pack(target,anchor,work/'same-device',work,retire_original=True)
        with pytest.raises(ValueError,match='original retained'):
            archive(target,anchor,work/'no-savings',scratch,expected_sha256=digest,maximum_payload_bytes=4*1024**2,
                minimum_free_bytes=0,minimum_reclaimed_bytes=8*1024**2,guard=lambda:None,retire_original=True)
        assert sha(target)==digest


def test_changed_segment_boundaries_reject_and_physical_output_is_removed():
    with tempfile.TemporaryDirectory(dir=ROOT/'reports') as w,tempfile.TemporaryDirectory(dir='/tmp') as s:
        work,scratch=Path(w),Path(s);anchor,target=inputs(work,scratch)
        pack(target,anchor,work/'target-archive',scratch)
        p=work/'target-archive/manifest.json';x=json.loads(p.read_text());x['segments'][0]['target_offset']=1;p.write_text(json.dumps(x))
        with pytest.raises(ValueError,match='contiguous'):restore(work/'target-archive',scratch/'broken.pt',scratch)
        assert not (scratch/'broken.pt').exists() and not (scratch/'broken.pt.restoring').exists()
