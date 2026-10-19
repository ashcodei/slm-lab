"""Atomic hard-link consolidation of inactive, byte-identical frozen copies.

All paths and complete checkpoint contents remain. Live-bound destinations are
forbidden. Canonical inode/size/mtime stay unchanged; a destination's new inode
is explicit. These are hard links for archived immutable files, not COW clones.
A temporary original link permits rollback until every postcondition passes.
"""
import os
from pathlib import Path
import secrets
import stat

from .corpus import file_hash


def metadata(path):
    s=Path(path).stat()
    return dict(device=s.st_dev,inode=s.st_ino,size=s.st_size,mtime_ns=s.st_mtime_ns,mode=s.st_mode,uid=s.st_uid,gid=s.st_gid)


def consolidate(source,destination,*,expected_sha256,live_bound_paths):
    source,destination=Path(source).absolute(),Path(destination).absolute()
    if source.is_symlink() or destination.is_symlink() or source==destination:
        raise ValueError('Two distinct ordinary archived checkpoint paths are required')
    if str(destination.resolve()) in {str(Path(p).resolve()) for p in live_bound_paths}:
        raise ValueError('A live-bound destination inode must never be replaced')
    before_a,before_b=metadata(source),metadata(destination)
    if (before_a['device']!=before_b['device'] or before_a['inode']==before_b['inode'] or before_a['size']!=before_b['size']
            or any(before_a[k]!=before_b[k] for k in ['mode','uid','gid']) or not stat.S_ISREG(before_a['mode'])):
        raise ValueError('Distinct ordinary same-filesystem copies with identical sizes, permissions and ownership required')
    if file_hash(source)!=expected_sha256 or file_hash(destination)!=expected_sha256:
        raise ValueError('Both complete existing byte streams must match the exact same original SHA256')
    token=secrets.token_hex(12);backup=destination.with_name('.'+destination.name+'.original-copy-'+token)
    pending=destination.with_name('.'+destination.name+'.shared-copy-'+token)
    if backup.exists() or pending.exists():raise ValueError('Fresh private atomic links required')
    linked=False;committed=False
    try:
        os.link(destination,backup)
        os.link(source,pending)
        if metadata(source)!=before_a or metadata(destination)!=before_b:
            raise ValueError('Archived source/destination changed before atomic consolidation')
        os.replace(pending,destination);linked=True
        after_a,after_b=metadata(source),metadata(destination)
        if (after_a!=before_a or after_b!=before_a
                or file_hash(source)!=expected_sha256 or file_hash(destination)!=expected_sha256
                or file_hash(backup)!=expected_sha256):
            raise ValueError('Every original byte stream and canonical guard must remain exact; restore original destination')
        fd=os.open(destination.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
        committed=True
        backup.unlink()
        return dict(source=str(source),destination=str(destination),original_complete_sha256=expected_sha256,
            original_source_metadata=before_a,original_destination_metadata=before_b,actual_shared_metadata=after_b,
            every_original_checkpoint_path_and_byte_preserved=True,canonical_inode_size_and_mtime_preserved=True,
            inactive_destination_inode_and_mtime_explicitly_changed=True,
            logical_duplicate_bytes_consolidated=before_b['size'],shared_hard_links_for_frozen_archives_not_copy_on_write=True)
    except BaseException:
        if linked and not committed and backup.exists():os.replace(backup,destination)
        raise
    finally:
        pending.unlink(missing_ok=True)
        if not linked or not committed:backup.unlink(missing_ok=True)
