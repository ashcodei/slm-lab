"""Durable lossless archives for verified, inactive intermediate inference snapshots."""
from datetime import datetime,timezone
import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import zlib

import numpy as np
import torch

from .checkpoint_archive import storage_ranges
from .checkpoint_xor import restore as verify_xor
from .corpus import file_hash
from .diverse_images import atomic_json
from .recurrent_ablation import state_digest
from .source_segment_archive_v1 import no_readers


BLOCK=1024**2


def _identity(path):
    s=Path(path).stat()
    return s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_nlink


def _sync_dir(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def _durable_json(path,value):
    atomic_json(path,value)
    with Path(path).open('rb') as source:os.fsync(source.fileno())
    _sync_dir(Path(path).parent)


def _unprotected(checkpoint,checkpoint_root):
    if not checkpoint.is_relative_to(checkpoint_root) or checkpoint.stat().st_nlink!=1:
        raise ValueError('One unshared own-checkpoint snapshot required')
    for manifest in checkpoint_root.rglob('manifest.json'):
        value=json.loads(manifest.read_text())
        for key in ['anchor','source_checkpoint']:
            if value.get(key) and Path(value[key]).resolve()==checkpoint:
                raise ValueError('Never archive/remove a retained full checkpoint anchor')
    no_readers(checkpoint)


def restore_snapshot(archive,output,minimum_free_bytes=0):
    archive,output=Path(archive).resolve(),Path(output).resolve()
    manifest=json.loads((archive/'manifest.json').read_text())
    if manifest.get('format')!='slm-intermediate-inference-xor-archive-v1':
        raise ValueError('Explicit verified intermediate-inference archive required')
    if output.exists() or output.is_relative_to(archive) or output==Path(manifest['anchor']).resolve():
        raise ValueError('Fresh restore destination outside the archive and retained full anchor required')
    output.parent.mkdir(parents=True,exist_ok=True)
    if (type(minimum_free_bytes) is not int or minimum_free_bytes<0
            or shutil.disk_usage(output.parent).free<manifest['original_bytes']+minimum_free_bytes):
        raise ValueError('Whole original reconstruction above explicit scratch/storage reserve required')
    with tempfile.TemporaryDirectory(prefix='slm-inference-restore-',dir=output.parent) as temporary:
        pending=Path(temporary)/'original.pt';result=verify_xor(archive,pending)
        with pending.open('rb') as f:os.fsync(f.fileno())
        if file_hash(pending)!=manifest['original_sha256']:
            raise ValueError('Every physically restored original byte must match')
        os.link(pending,output);_sync_dir(output.parent)
    return dict(restored_path=str(output),**result)


def archive_snapshot(checkpoint,anchor,archive,*,receipt,checkpoint_root,expected_anchor_sha256,
                     maximum_payload_bytes,minimum_free_bytes,scratch_directory,
                     minimum_scratch_free_bytes=0,maximum_capacity_wait_seconds=60):
    checkpoint,anchor,archive,receipt,checkpoint_root,scratch_directory=map(lambda p:Path(p).resolve(),
        [checkpoint,anchor,archive,receipt,checkpoint_root,scratch_directory])
    if (checkpoint==anchor or archive.exists() or archive in {checkpoint,anchor}
            or not anchor.is_relative_to(checkpoint_root)
            or not archive.is_relative_to(checkpoint_root)
            or _identity(checkpoint)[:2]==_identity(anchor)[:2]):
        raise ValueError('Distinct own inference snapshot/full anchor and fresh archive required')
    if (type(maximum_payload_bytes) is not int or maximum_payload_bytes<1
            or type(minimum_free_bytes) is not int or minimum_free_bytes<0
            or type(minimum_scratch_free_bytes) is not int or minimum_scratch_free_bytes<0
            or type(maximum_capacity_wait_seconds) not in [int,float] or maximum_capacity_wait_seconds<0):
        raise ValueError('Explicit bounded payload, persistent/scratch reserve and wait budget required')
    _unprotected(checkpoint,checkpoint_root)
    original=_identity(checkpoint);anchor_identity=_identity(anchor);extraction=json.loads(receipt.read_text())
    original_sha=file_hash(checkpoint);receipt_sha=file_hash(receipt)
    expected=extraction.get('expected_metadata',{})
    if (Path(extraction.get('inference_checkpoint','')).resolve()!=checkpoint
            or extraction.get('inference_checkpoint_sha256')!=original_sha
            or extraction.get('inference_checkpoint_bytes')!=original[2]
            or extraction.get('all_model_state_tensors_identical') is not True
            or extraction.get('source_checkpoint_modified') is not False
            or extraction.get('excluded_fields')!=['optimizer']
            or type(expected.get('steps')) is not int or type(expected.get('planned_steps')) is not int
            or not 0<expected['steps']<expected['planned_steps']):
        raise ValueError('Exact original intermediate inference-extraction receipt required; no completed-budget claim')
    if file_hash(anchor)!=expected_anchor_sha256:raise ValueError('Exact retained full own anchor bytes required')
    snapshot=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
    previous=torch.load(anchor,map_location='cpu',weights_only=True,mmap=True)
    if ('optimizer' in snapshot or 'reference_model' in snapshot or not isinstance(snapshot.get('model'),dict)
            or any(snapshot.get(k)!=v for k,v in expected.items())
            or snapshot.get('config')!=previous.get('config')
            or snapshot.get('tokenizer_sha256')!=previous.get('tokenizer_sha256')
            or any(not bool(torch.isfinite(v).all()) for v in snapshot['model'].values())
            or state_digest(snapshot['model'])!=extraction.get('original_and_restored_model_state_sha256')):
        raise ValueError('Finite own weights-only original state/config/tokenizer/metadata must match extraction')
    del snapshot,previous
    if (shutil.disk_usage(checkpoint.parent).free<minimum_free_bytes+maximum_payload_bytes
            or shutil.disk_usage(scratch_directory).free<minimum_scratch_free_bytes+original[2]):
        raise ValueError('Whole bounded payload and physical restore must fit above all original commitments')
    target_ranges=storage_ranges(checkpoint);anchor_ranges=storage_ranges(anchor);parts=[];position=0
    for key,(offset,length,_) in sorted(target_ranges.items(),key=lambda item:item[1][0]):
        if offset<position:raise ValueError('Overlapping original tensor storage ranges')
        if offset>position:parts.append(('raw',position,offset-position,None))
        candidate=anchor_ranges.get(key);reusable=candidate is not None and candidate[1]==length
        parts.append(('xor' if reusable else 'raw',offset,length,candidate[0] if reusable else None));position=offset+length
    if position<original[2]:parts.append(('raw',position,original[2]-position,None))
    archive.mkdir(parents=True);segments=[]
    with checkpoint.open('rb') as current,anchor.open('rb') as base,(archive/'payload.bin').open('xb') as payload:
        def write(data):
            if payload.tell()+len(data)>maximum_payload_bytes:raise ValueError('Payload exceeded its declared capacity; original retained')
            deadline=time.monotonic()+maximum_capacity_wait_seconds
            while shutil.disk_usage(archive).free-len(data)<minimum_free_bytes:
                if time.monotonic()>=deadline:raise ValueError('Original writer reservation still occupied; original snapshot retained')
                time.sleep(.25)
            payload.write(data)
        for kind,start,length,anchor_offset in parts:
            current.seek(start)
            if kind=='xor':base.seek(anchor_offset)
            begin=payload.tell();remaining=length;compressor=zlib.compressobj(level=1)
            while remaining:
                raw=current.read(min(BLOCK,remaining))
                if not raw:raise ValueError('Original inference snapshot truncated')
                remaining-=len(raw)
                if kind=='xor':
                    old=base.read(len(raw))
                    if len(old)!=len(raw):raise ValueError('Retained full anchor truncated')
                    raw=np.bitwise_xor(np.frombuffer(raw,dtype=np.uint8),np.frombuffer(old,dtype=np.uint8)).tobytes()
                write(compressor.compress(raw))
            write(compressor.flush())
            segments.append(dict(kind=kind,target_offset=start,length=length,anchor_offset=anchor_offset,
                payload_offset=begin,payload_bytes=payload.tell()-begin))
        payload.flush();os.fsync(payload.fileno())
    if (_identity(checkpoint)!=original or _identity(anchor)!=anchor_identity or file_hash(checkpoint)!=original_sha
            or file_hash(anchor)!=expected_anchor_sha256 or file_hash(receipt)!=receipt_sha):
        raise ValueError('Original snapshot/anchor/extraction changed; original retained')
    payload=archive/'payload.bin'
    if payload.stat().st_size>=original[2]:raise ValueError('No positive storage recovery; original retained')
    manifest=dict(version=1,format='slm-intermediate-inference-xor-archive-v1',created_at=datetime.now(timezone.utc).isoformat(),
        original_checkpoint=str(checkpoint),original_sha256=original_sha,original_bytes=original[2],
        original_allocated_bytes=checkpoint.stat().st_blocks*512,anchor=str(anchor),anchor_sha256=expected_anchor_sha256,
        anchor_must_be_retained=True,payload_sha256=file_hash(payload),payload_bytes=payload.stat().st_size,
        payload_allocated_bytes=payload.stat().st_blocks*512,segments=segments,original_removed=False,
        extraction_receipt=str(receipt),extraction_receipt_sha256=receipt_sha,expected_original_metadata=expected,
        source_training_budget_completed=False,inference_snapshot_extracted_and_verified=True,
        maximum_payload_bytes=maximum_payload_bytes,minimum_persistent_free_bytes=minimum_free_bytes,
        old_restore_compatible=True,goal_achieved=False,
        scope='Lossless inactive intermediate inference-snapshot storage; no completed source-budget or capability claim. Restore before using the original model.pt path; retain exact full anchor and payload.')
    _durable_json(archive/'manifest.json',manifest);manifest['verification']=verify_xor(archive)
    with tempfile.TemporaryDirectory(prefix='slm-inference-archive-check-',dir=scratch_directory) as temporary:
        destination=Path(temporary)/'restored-original.pt'
        restored=restore_snapshot(archive,destination,minimum_scratch_free_bytes)
        if file_hash(destination)!=original_sha or destination.stat().st_size!=original[2]:
            raise ValueError('Independent physical full restore differs from original inference snapshot')
        manifest['physical_restore_verification']=dict(**restored,independent_file_sha256=file_hash(destination),temporary_copy_removed_after_check=True)
    _durable_json(archive/'manifest.json',manifest);_unprotected(checkpoint,checkpoint_root)
    if (_identity(checkpoint)!=original or _identity(anchor)!=anchor_identity or file_hash(checkpoint)!=original_sha
            or file_hash(anchor)!=expected_anchor_sha256 or file_hash(receipt)!=receipt_sha):
        raise ValueError('Original bytes/anchor/receipt changed before retirement; original retained')
    checkpoint.unlink();_sync_dir(checkpoint.parent);manifest['original_removed']=True
    manifest['reclaimed_allocated_bytes']=(manifest['original_allocated_bytes']-manifest['payload_allocated_bytes']
        -(archive/'manifest.json').stat().st_blocks*512)
    _durable_json(archive/'manifest.json',manifest)
    actual=(manifest['original_allocated_bytes']-manifest['payload_allocated_bytes']
        -(archive/'manifest.json').stat().st_blocks*512)
    if actual!=manifest['reclaimed_allocated_bytes']:
        manifest['reclaimed_allocated_bytes']=actual;_durable_json(archive/'manifest.json',manifest)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['verify','restore']);parser.add_argument('--archive',required=True)
    parser.add_argument('--output');parser.add_argument('--minimum-free-bytes',type=int,default=0)
    args=parser.parse_args()
    if args.action=='restore':
        if not args.output:parser.error('An explicit fresh output path is required')
        result=restore_snapshot(args.archive,args.output,args.minimum_free_bytes)
    else:
        manifest=json.loads((Path(args.archive)/'manifest.json').read_text())
        if manifest.get('format')!='slm-intermediate-inference-xor-archive-v1':
            raise ValueError('Explicit intermediate-inference archive required')
        result=verify_xor(args.archive)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
