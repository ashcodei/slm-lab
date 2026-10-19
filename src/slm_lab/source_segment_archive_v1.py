"""Lossless archival of inactive byte-identical source copies inside retained files."""
from datetime import datetime,timezone
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

from .corpus import file_hash
from .diverse_images import atomic_json

BLOCK=4*1024**2


def no_readers(target,maximum_wait_seconds=10):
    target=Path(target).resolve();stat=target.stat();identity=(stat.st_dev,stat.st_ino)
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
        deadline=time.monotonic()+maximum_wait_seconds
        while True:
            try:
                status=(proc/'status').read_text().splitlines()
                uid=next(s for s in status if s.startswith('Uid:')).split()[1:3]
                state=next(s for s in status if s.startswith('State:')).split()[1]
                # /proc directory ownership can change during a non-dumpable
                # exec; use actual process UIDs, and only authoritative exit.
                if str(stat.st_uid) not in uid or state in {'Z','X'}:break
                args=(proc/'cmdline').read_bytes().split(b'\0')
                for raw in args:
                    if not 0<len(raw)<4096:continue
                    try:p=Path(raw.decode()).resolve()
                    except (OSError,UnicodeError,ValueError):continue
                    if p in {target,target.parent}:
                        raise ValueError('Original source is named by a live process')
                for fd in (proc/'fd').iterdir():
                    try:s=fd.stat()
                    except FileNotFoundError:continue
                    if (s.st_dev,s.st_ino)==identity:
                        raise ValueError('Original source has a live open file descriptor')
                if str(target) in (proc/'maps').read_text():
                    raise ValueError('Original source has a live memory mapping')
                break
            except (FileNotFoundError,ProcessLookupError):break
            except PermissionError as error:
                if time.monotonic()>=deadline:
                    raise ValueError('Cannot inspect same-owner process '+proc.name+' before source archival') from error
                time.sleep(.05)


def segment_hash(anchor,offset,length):
    if type(offset) is not int or type(length) is not int or offset<0 or length<1:
        raise ValueError('Positive bounded byte segment required')
    anchor=Path(anchor)
    if offset+length>anchor.stat().st_size:raise ValueError('Segment exceeds retained source')
    digest=hashlib.sha256()
    with anchor.open('rb') as f:
        f.seek(offset);remaining=length
        while remaining:
            block=f.read(min(BLOCK,remaining))
            if not block:raise ValueError('Retained source segment truncated')
            digest.update(block);remaining-=len(block)
    return digest.hexdigest()


def verify(manifest_path):
    record=json.loads(Path(manifest_path).read_text())
    if record['format']!='slm-source-segment-archive-v1':raise ValueError('Known source archive format required')
    anchor=Path(record['retained_anchor'])
    if file_hash(anchor)!=record['retained_anchor_sha256']:
        raise ValueError('Retained source anchor changed')
    if segment_hash(anchor,record['offset'],record['original_bytes'])!=record['original_sha256']:
        raise ValueError('Retained source segment differs from original bytes')
    return record


def archive_duplicate(target,anchor,offset,manifest_path,expected_target_sha256,expected_anchor_sha256):
    target,anchor,manifest_path=map(lambda p:Path(p).resolve(),[target,anchor,manifest_path])
    if manifest_path.exists() or target==anchor or target==manifest_path or anchor==manifest_path:
        raise ValueError('Distinct original/retained files and a fresh source archive required')
    original=target.stat()
    if original.st_nlink!=1 or original.st_ino==anchor.stat().st_ino and original.st_dev==anchor.stat().st_dev:
        raise ValueError('Do not remove a shared hard link or retained anchor')
    no_readers(target)
    if file_hash(target)!=expected_target_sha256 or file_hash(anchor)!=expected_anchor_sha256:
        raise ValueError('Expected original and retained source checksums required')
    if type(offset) is not int or offset<0 or offset+original.st_size>anchor.stat().st_size:
        raise ValueError('Original complete source must fit the retained segment')
    with target.open('rb') as first,anchor.open('rb') as second:
        second.seek(offset);remaining=original.st_size
        while remaining:
            a=first.read(min(BLOCK,remaining));b=second.read(len(a))
            if not a or a!=b:raise ValueError('Original source and retained bytes are not exactly identical')
            remaining-=len(a)
    record=dict(format='slm-source-segment-archive-v1',created_at=datetime.now(timezone.utc).isoformat(),
        original_path=str(target),original_sha256=expected_target_sha256,original_bytes=original.st_size,
        original_allocated_bytes=original.st_blocks*512,retained_anchor=str(anchor),
        retained_anchor_sha256=expected_anchor_sha256,offset=offset,
        anchor_must_be_retained=True,every_original_byte_compared_exactly=True,
        original_removed=False,goal_achieved=False,
        scope='Inactive duplicate source bytes retained exactly in an immutable full source anchor. Restore before re-running an original consumer that needs the individual path.')
    atomic_json(manifest_path,record);verify(manifest_path)
    no_readers(target)
    current=target.stat()
    if ((current.st_dev,current.st_ino,current.st_size,current.st_mtime_ns)
            !=(original.st_dev,original.st_ino,original.st_size,original.st_mtime_ns)
            or file_hash(target)!=expected_target_sha256 or file_hash(anchor)!=expected_anchor_sha256):
        raise ValueError('Original/retained source changed before removal; original retained')
    target.unlink();record['original_removed']=True;atomic_json(manifest_path,record)
    return record


def restore(manifest_path,output=None):
    record=verify(manifest_path);anchor=Path(record['retained_anchor'])
    target=Path(output if output is not None else record['original_path']).resolve()
    if target.exists() or target in {anchor,Path(manifest_path).resolve()}:
        raise ValueError('Fresh restore destination required; never overwrite an existing file')
    target.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix=target.name+'.restoring-',dir=target.parent);temporary=Path(name)
    try:
        h=hashlib.sha256();remaining=record['original_bytes']
        with os.fdopen(fd,'wb') as out,anchor.open('rb') as source:
            source.seek(record['offset'])
            while remaining:
                block=source.read(min(BLOCK,remaining))
                if not block:raise ValueError('Retained source changed during restoration')
                out.write(block);h.update(block);remaining-=len(block)
            out.flush();os.fsync(out.fileno())
        if h.hexdigest()!=record['original_sha256'] or file_hash(anchor)!=record['retained_anchor_sha256']:
            raise ValueError('Restored source bytes do not match the original')
        os.link(temporary,target)
        return dict(restored_path=str(target),restored_sha256=h.hexdigest(),restored_bytes=record['original_bytes'])
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['verify','restore'])
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--output')
    args=parser.parse_args()
    if args.action=='verify':
        record=verify(args.manifest)
        result={k:record[k] for k in ['original_path','original_bytes','original_sha256','retained_anchor','offset']}
    else:result=restore(args.manifest,args.output)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
