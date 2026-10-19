"""Losslessly archive an inactive raw cache; preserve original consumer paths on restore."""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from .corpus import file_hash
from .diverse_images import atomic_json
from .source_segment_archive_v1 import BLOCK, no_readers


def identity(path):
    s=Path(path).stat()
    return s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_nlink


def decoded_identity(path):
    h=hashlib.sha256();size=0
    with gzip.open(path,'rb') as source:
        while block:=source.read(BLOCK):
            h.update(block);size+=len(block)
    return size,h.hexdigest()


def verify(manifest):
    record=json.loads(Path(manifest).read_text())
    if record['format']!='slm-lossless-source-gzip-v1':raise ValueError('Known raw-source archive required')
    archive=Path(record['archive_path'])
    if file_hash(archive)!=record['archive_sha256']:raise ValueError('Compressed archive bytes changed')
    if decoded_identity(archive)!=(record['original_bytes'],record['original_sha256']):
        raise ValueError('Decoded archive differs from every original byte')
    return record


def archive(source,manifest,expected_sha256,maximum_compressed_bytes,minimum_free_bytes):
    source,manifest=Path(source).resolve(),Path(manifest).resolve()
    compressed=source.with_name(source.name+'.lossless-v1.gz')
    if manifest.exists() or compressed.exists() or manifest in {source,compressed}:
        raise ValueError('Preserve existing source archives and manifests')
    before=identity(source)
    if before[-1]!=1:raise ValueError('Do not remove a shared hard link')
    no_readers(source)
    if file_hash(source)!=expected_sha256:raise ValueError('Expected original raw-source checksum required')
    if shutil.disk_usage(source.parent).free<minimum_free_bytes+maximum_compressed_bytes:
        raise ValueError('Reserve all active writers plus the bounded archive output')
    fd,name=tempfile.mkstemp(prefix=source.name+'.compressing-',dir=source.parent)
    temporary=Path(name);h=hashlib.sha256();total=0
    try:
        with os.fdopen(fd,'wb') as raw,source.open('rb') as original:
            with gzip.GzipFile(filename='',mode='wb',fileobj=raw,compresslevel=1,mtime=0) as output:
                while block:=original.read(BLOCK):
                    output.write(block);h.update(block);total+=len(block)
                    if raw.tell()>maximum_compressed_bytes or shutil.disk_usage(source.parent).free<minimum_free_bytes:
                        raise ValueError('Compression output exceeded reserved storage budget')
            raw.flush();os.fsync(raw.fileno())
        if (temporary.stat().st_size>maximum_compressed_bytes or identity(source)!=before
                or (total,h.hexdigest())!=(before[2],expected_sha256)
                or decoded_identity(temporary)!=(total,expected_sha256)):
            raise ValueError('Exact complete original source must survive archive round trip')
        os.link(temporary,compressed)
    finally:
        temporary.unlink(missing_ok=True)
    record=dict(format='slm-lossless-source-gzip-v1',created_at=datetime.now(timezone.utc).isoformat(),
        original_path=str(source),original_bytes=before[2],original_sha256=expected_sha256,
        original_allocated_bytes=source.stat().st_blocks*512,archive_path=str(compressed),
        archive_bytes=compressed.stat().st_size,archive_sha256=file_hash(compressed),
        archive_allocated_bytes=compressed.stat().st_blocks*512,original_removed=False,
        scope='Inactive raw-source storage only; restore before rerunning consumers requiring the original path.',
        goal_achieved=False)
    atomic_json(manifest,record);verify(manifest);no_readers(source)
    if (identity(source)!=before or file_hash(source)!=expected_sha256
            or file_hash(compressed)!=record['archive_sha256']):
        raise ValueError('Original changed before removal; retain original and archive')
    source.unlink();record['original_removed']=True
    record['reclaimed_allocated_bytes']=record['original_allocated_bytes']-record['archive_allocated_bytes']
    atomic_json(manifest,record)
    return record


def restore(manifest,output=None):
    record=verify(manifest);compressed=Path(record['archive_path'])
    destination=Path(output or record['original_path']).resolve()
    if destination.exists() or destination in {compressed,Path(manifest).resolve()}:
        raise ValueError('Fresh restore destination required; never overwrite an existing file')
    destination.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix=destination.name+'.restoring-',dir=destination.parent)
    temporary=Path(name);h=hashlib.sha256();size=0
    try:
        with os.fdopen(fd,'wb') as target,gzip.open(compressed,'rb') as source:
            while block:=source.read(BLOCK):
                target.write(block);h.update(block);size+=len(block)
            target.flush();os.fsync(target.fileno())
        if ((size,h.hexdigest())!=(record['original_bytes'],record['original_sha256'])
                or file_hash(compressed)!=record['archive_sha256']):
            raise ValueError('Complete restored original bytes must match')
        os.link(temporary,destination)
        return dict(restored_path=str(destination),restored_bytes=size,restored_sha256=h.hexdigest())
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['archive','verify','restore'])
    parser.add_argument('--manifest',required=True);parser.add_argument('--source');parser.add_argument('--output')
    parser.add_argument('--expected-sha256');parser.add_argument('--maximum-compressed-bytes',type=int)
    parser.add_argument('--minimum-free-bytes',type=int)
    args=parser.parse_args()
    if args.action=='archive':
        if (not args.source or not args.expected_sha256 or args.maximum_compressed_bytes is None
                or args.maximum_compressed_bytes<1 or args.minimum_free_bytes is None or args.minimum_free_bytes<0):
            parser.error('Archival requires exact source/hash and positive bounded output/storage reservation')
        result=archive(args.source,args.manifest,args.expected_sha256,args.maximum_compressed_bytes,args.minimum_free_bytes)
    elif args.action=='verify':result=verify(args.manifest)
    else:result=restore(args.manifest,args.output)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
