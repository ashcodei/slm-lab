"""Share only kernel-compared identical extents; preserve every logical file.

Linux FIDEDUPERANGE compares bytes instead of replacing a destination's data.
No unlink, rename, truncation, tensor rewrite or checkpoint-path change occurs.
The caller retains original complete file digests and inode/size/mtime guards.
"""
import fcntl
import os
from pathlib import Path
import platform
import stat
import struct

from .corpus import file_hash

FIDEDUPERANGE=0xc0189436
MAXIMUM_RANGE=16*1024**2
BLOCK=4096


def state(fd):
    s=os.fstat(fd)
    return dict(device=s.st_dev,inode=s.st_ino,size=s.st_size,mtime_ns=s.st_mtime_ns,mode=s.st_mode,uid=s.st_uid,gid=s.st_gid)


def compare_share_range(source_fd,destination_fd,offset,length):
    if (platform.system()!='Linux' or platform.machine()!='x86_64'
            or type(offset) is not int or type(length) is not int or offset<0 or offset%BLOCK
            or not 0<length<=MAXIMUM_RANGE or length%BLOCK):
        raise ValueError('Verified Linux x86_64 ABI and complete aligned bounded extent range required')
    request=bytearray(struct.pack('=QQHHI',offset,length,1,0,0)+struct.pack('=qQQiI',destination_fd,offset,0,0,0))
    fcntl.ioctl(source_fd,FIDEDUPERANGE,request,True)
    fd,destination_offset,shared,status,reserved=struct.unpack_from('=qQQiI',request,24)
    if (fd!=destination_fd or destination_offset!=offset or reserved or shared>length or shared%BLOCK):
        raise ValueError('The exact bounded destination descriptor/range and actual shared byte count must reconstruct')
    return dict(offset=offset,requested_bytes=length,actual_logical_bytes_compared_and_shared=shared,kernel_status=status)


def share_identical(source,destination,*,expected_sha256):
    source,destination=Path(source),Path(destination)
    if source.is_symlink() or destination.is_symlink():raise ValueError('Existing ordinary checkpoint files only; no symlink or path substitution')
    if file_hash(source)!=expected_sha256 or file_hash(destination)!=expected_sha256:
        raise ValueError('Both entire original checkpoint byte streams must already have the identical declared SHA256')
    source_fd=os.open(source,os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW)
    try:
        destination_fd=os.open(destination,os.O_RDWR|os.O_CLOEXEC|os.O_NOFOLLOW)
        try:
            before_source,before_destination=state(source_fd),state(destination_fd)
            if (before_source['device']!=before_destination['device'] or before_source['inode']==before_destination['inode']
                    or before_source['size']!=before_destination['size'] or before_source['size']<BLOCK
                    or not stat.S_ISREG(before_source['mode']) or not stat.S_ISREG(before_destination['mode'])):
                raise ValueError('Two distinct ordinary same-filesystem inodes and identical complete byte sizes are required')
            size=before_source['size'];limit=size-size%BLOCK;offset=0;operations=[]
            while offset<limit:
                observed=compare_share_range(source_fd,destination_fd,offset,min(MAXIMUM_RANGE,limit-offset))
                if observed['kernel_status']!=0 or observed['actual_logical_bytes_compared_and_shared']<BLOCK:
                    raise ValueError('The kernel did not verify and share this exact identical source/destination range: '+str(observed))
                operations.append(observed);offset+=observed['actual_logical_bytes_compared_and_shared']
            os.fsync(destination_fd)
            if state(source_fd)!=before_source or state(destination_fd)!=before_destination:
                raise ValueError('Storage sharing changed inode, size, mtime, ownership or mode; do not apply to other scientific files')
            if file_hash(source)!=expected_sha256 or file_hash(destination)!=expected_sha256:
                raise ValueError('Every original source/destination byte must retain its entire complete SHA256')
            return dict(source=str(source.resolve()),destination=str(destination.resolve()),complete_unchanged_sha256=expected_sha256,
                original_source_state=before_source,original_destination_state=before_destination,actual_kernel_range_operations=operations,
                logical_identical_bytes_compared_and_shared=offset,unchanged_unshared_final_partial_block_bytes=size-limit,
                all_file_paths_and_content_and_inode_size_mtime_ownership_mode_preserved=True,
                logical_shared_bytes_not_a_claim_about_isolated_filesystem_free_space_delta=True)
        finally:os.close(destination_fd)
    finally:os.close(source_fd)
