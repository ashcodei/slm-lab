"""Exact compressed tensor-byte differences against an immutable retained checkpoint."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import zlib

import numpy as np
from .checkpoint_archive import storage_ranges
from .corpus import file_hash
from .diverse_images import atomic_json

BLOCK = 1 << 20


def restore(archive, output=None):
    archive = Path(archive); manifest = json.loads((archive/'manifest.json').read_text())
    anchor = Path(manifest['anchor']); payload = archive/'payload.bin'
    if file_hash(anchor) != manifest['anchor_sha256'] or file_hash(payload) != manifest['payload_sha256']:
        raise ValueError('Exact checkpoint anchor/payload changed')
    target = Path(output) if output is not None else None
    if target is not None and target.exists(): raise ValueError('Restore requires a fresh output path')
    writer = target.open('xb') if target is not None else None
    digest = hashlib.sha256(); total = 0
    try:
        with anchor.open('rb') as source, payload.open('rb') as compressed:
            for segment in manifest['segments']:
                if segment['target_offset'] != total: raise ValueError('Noncontiguous checkpoint archive segments')
                compressed.seek(segment['payload_offset']); remaining = segment['payload_bytes']; decoder = zlib.decompressobj(); count = 0
                if segment['kind'] == 'xor': source.seek(segment['anchor_offset'])
                while remaining:
                    block = compressed.read(min(BLOCK,remaining))
                    if not block: raise ValueError('Truncated checkpoint difference payload')
                    remaining -= len(block); pending = block
                    while pending:
                        raw = decoder.decompress(pending,BLOCK); pending = decoder.unconsumed_tail
                        if not raw and not pending: break
                        if count + len(raw) > segment['length']: raise ValueError('Checkpoint segment exceeded declared size')
                        if segment['kind'] == 'xor':
                            previous = source.read(len(raw))
                            if len(previous) != len(raw): raise ValueError('Anchor tensor bytes truncated')
                            raw = np.bitwise_xor(np.frombuffer(raw,dtype=np.uint8),np.frombuffer(previous,dtype=np.uint8)).tobytes()
                        digest.update(raw); count += len(raw); total += len(raw)
                        if writer is not None: writer.write(raw)
                if not decoder.eof or decoder.unused_data or count != segment['length']:
                    raise ValueError('Checkpoint difference segment is incomplete/oversized')
        if total != manifest['original_bytes'] or digest.hexdigest() != manifest['original_sha256']:
            raise ValueError('Reconstructed checkpoint differs from every original byte')
    finally:
        if writer is not None: writer.close()
    return {'original_bytes':total,'restored_sha256':digest.hexdigest(),'whole_file_verified_exact':True}


def archive_checkpoint(checkpoint, anchor, archive, remove_original=False):
    checkpoint,anchor,archive = [Path(path).resolve() for path in [checkpoint,anchor,archive]]
    if checkpoint==anchor or archive.exists() or checkpoint.stat().st_ino==anchor.stat().st_ino:
        raise ValueError('Archive requires distinct checkpoint/anchor and a fresh directory')
    training = json.loads((checkpoint.parent/'training.json').read_text())
    if 'completed' not in training.get('status','') or training['steps'] != training['planned_steps']:
        raise ValueError('Only completed experimental checkpoints may be archived')
    for process in Path('/proc').iterdir():
        if not process.name.isdigit() or int(process.name)==os.getpid(): continue
        try: arguments = (process/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError,PermissionError,ProcessLookupError): continue
        for value in arguments:
            if not 0 < len(value) < 4096: continue
            try: path=Path(value.decode()).resolve()
            except (ValueError,OSError,UnicodeError): continue
            if path in {checkpoint,checkpoint.parent}: raise ValueError('Completed checkpoint still referenced by another live process')
    original_sha,anchor_sha = file_hash(checkpoint),file_hash(anchor)
    ranges=storage_ranges(checkpoint); previous=storage_ranges(anchor); size=checkpoint.stat().st_size
    parts=[]; position=0
    for key,(offset,length,_) in sorted(ranges.items(),key=lambda item:item[1][0]):
        if offset < position: raise ValueError('Overlapping source tensor entries')
        if offset > position: parts.append(('raw',position,offset-position,None))
        candidate=previous.get(key)
        parts.append(('xor' if candidate is not None and candidate[1]==length else 'raw',offset,length,
                      candidate[0] if candidate is not None and candidate[1]==length else None))
        position=offset+length
    if position < size: parts.append(('raw',position,size-position,None))
    archive.mkdir(parents=True); segments=[]
    with checkpoint.open('rb') as current,anchor.open('rb') as source,(archive/'payload.bin').open('xb') as payload:
        for kind,start,length,source_offset in parts:
            current.seek(start)
            if kind=='xor': source.seek(source_offset)
            begin=payload.tell(); remaining=length; compressor=zlib.compressobj(level=1)
            while remaining:
                raw=current.read(min(BLOCK,remaining))
                if not raw: raise ValueError('Original checkpoint truncated during archival')
                remaining-=len(raw)
                if kind=='xor':
                    old=source.read(len(raw))
                    if len(old)!=len(raw): raise ValueError('Original anchor tensor truncated')
                    raw=np.bitwise_xor(np.frombuffer(raw,dtype=np.uint8),np.frombuffer(old,dtype=np.uint8)).tobytes()
                payload.write(compressor.compress(raw))
            payload.write(compressor.flush())
            segments.append({'kind':kind,'target_offset':start,'length':length,'anchor_offset':source_offset,
                             'payload_offset':begin,'payload_bytes':payload.tell()-begin})
    if file_hash(checkpoint)!=original_sha or file_hash(anchor)!=anchor_sha: raise ValueError('Checkpoint/anchor changed during archival')
    payload_size=(archive/'payload.bin').stat().st_size
    manifest={'version':1,'created_at':datetime.now(timezone.utc).isoformat(),'original_checkpoint':str(checkpoint),
              'original_sha256':original_sha,'original_bytes':size,'anchor':str(anchor),'anchor_sha256':anchor_sha,
              'anchor_must_be_retained':True,'payload_sha256':file_hash(archive/'payload.bin'),'payload_bytes':payload_size,
              'segments':segments,'original_removed':False,'goal_achieved':False,
              'scope':'Lossless storage operation; source anchor may be an interim snapshot, not completed model training.'}
    atomic_json(archive/'manifest.json',manifest)
    verification=restore(archive);manifest['verification']=verification
    if remove_original:
        if payload_size >= size: raise ValueError('Archive does not save space; original remains intact')
        if file_hash(checkpoint)!=original_sha or file_hash(anchor)!=anchor_sha: raise ValueError('Checkpoint changed before verified removal')
        checkpoint.unlink();manifest['original_removed']=True
    atomic_json(archive/'manifest.json',manifest);return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    create=commands.add_parser('archive');create.add_argument('--checkpoint',required=True);create.add_argument('--anchor',required=True)
    create.add_argument('--output',required=True);create.add_argument('--remove-original',action='store_true')
    recover=commands.add_parser('restore');recover.add_argument('--archive',required=True);recover.add_argument('--output',required=True)
    args=parser.parse_args()
    if args.command=='archive': result=archive_checkpoint(args.checkpoint,args.anchor,args.output,args.remove_original)
    else: result=restore(args.archive,args.output)
    print(json.dumps({key:value for key,value in result.items() if key!='segments'},indent=2))


if __name__=='__main__':main()
