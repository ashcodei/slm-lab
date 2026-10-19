"""Two-slot, fsynced shared actor/Adam commits with external RNG manifests.

A new tensor file and receipt become durable before LATEST is replaced. The
previous committed slot survives any earlier failure. Root performs I/O and
coordinates its result; readers verify exact receipt and checkpoint bytes.
"""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil

import torch

from .codenet_checkpoint_v1 import atomic_checkpoint
from .corpus import file_hash
from .parallel_function_repair_stage_v1 import manifest

RECEIPT='own-parallel-repair-durable-shared-state-receipt-v1'
POINTER='own-parallel-repair-durable-latest-pointer-v1'
SCOPE=('format','world_size','profile','mode','precision','planned','schedule_identity_sha256','options_sha256','sources_sha256','frozen_base_sha256')


def durable_json(path,value,reserve):
    path=Path(path);pending=path.with_suffix(path.suffix+'.pending')
    raw=(json.dumps(value,sort_keys=True,ensure_ascii=True,allow_nan=False,indent=2)+'\n').encode()
    if len(raw)>2*1024**2 or shutil.disk_usage(path.parent).free-len(raw)<reserve:
        raise ValueError('Complete bounded metadata must fit above the persistent reserve')
    if pending.exists():raise ValueError('An interrupted metadata write requires inspection')
    try:
        with pending.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
        os.replace(pending,path)
        fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
    except Exception:
        pending.unlink(missing_ok=True);raise


def regular(path):
    path=Path(path)
    if not path.is_file() or path.is_symlink():raise ValueError('Only complete regular committed artifacts are eligible')


def read_receipt(directory,*,expected_pointer_sha256=None):
    root=Path(directory).resolve();pointer_path=root/'LATEST.json';regular(pointer_path)
    pointer_sha=file_hash(pointer_path)
    if expected_pointer_sha256 is not None and pointer_sha!=expected_pointer_sha256:raise ValueError('The exact retained latest-pointer identity differs')
    pointer=json.loads(pointer_path.read_text());generation=pointer.get('generation');slot=pointer.get('slot')
    if (pointer.get('format')!=POINTER or type(generation) is not int or generation<1 or slot!=generation%2
            or type(pointer.get('completed')) is not int or pointer['completed']<1):raise ValueError('Exact committed pointer generation required')
    receipt_path=root/('receipt-slot-'+str(slot)+'.json');regular(receipt_path)
    if (Path(pointer.get('receipt','')).resolve()!=receipt_path or file_hash(receipt_path)!=pointer.get('receipt_sha256')):
        raise ValueError('The exact committed external receipt differs')
    receipt=json.loads(receipt_path.read_text());checkpoint=root/('state-slot-'+str(slot)+'.pt');regular(checkpoint)
    if (receipt.get('format')!=RECEIPT or receipt.get('generation')!=generation or receipt.get('slot')!=slot
            or receipt.get('completed')!=pointer['completed'] or Path(receipt.get('checkpoint','')).resolve()!=checkpoint
            or receipt.get('semantic_manifest',{}).get('completed')!=pointer['completed']
            or file_hash(checkpoint)!=receipt.get('checkpoint_sha256')):
        raise ValueError('The exact complete shared checkpoint and external manifest must agree before loading')
    if file_hash(pointer_path)!=pointer_sha:raise ValueError('Latest pointer changed during committed readback; inspect the current same owner')
    return receipt,pointer_sha


def read_latest(directory,*,expected_pointer_sha256=None):
    receipt,pointer_sha=read_receipt(directory,expected_pointer_sha256=expected_pointer_sha256)
    state=torch.load(receipt['checkpoint'],map_location='cpu',weights_only=True,mmap=True)
    if manifest(state)!=receipt['semantic_manifest']:raise ValueError('Committed tensor contents, every private RNG/cursor and external semantic identity differ')
    return state,receipt,pointer_sha


def commit(stage,directory,*,maximum_bytes,minimum_free_bytes):
    root=Path(directory).resolve();error=None
    try:
        if (type(maximum_bytes) is not int or maximum_bytes<1 or type(minimum_free_bytes) is not int or minimum_free_bytes<0):
            raise ValueError('Explicit bounded complete snapshot capacity and nonnegative reserve required')
        specification=dict(directory=str(root),maximum_bytes=maximum_bytes,minimum_free_bytes=minimum_free_bytes,completed=stage.completed)
    except Exception as exc:error=type(exc).__name__+': '+str(exc)
    stage.coordinated(error,'durable commit arguments')
    if any(v!=specification for v in stage.gather(specification)):raise ValueError('Every actual replica must commit the same complete generation')
    state,semantic=stage.snapshot();receipt=None;error=None
    if stage.rank==0:
        try:
            root.mkdir(parents=True,exist_ok=True)
            old=None
            if (root/'LATEST.json').exists():old,_=read_receipt(root)
            if old is not None and (old['completed']>=stage.completed
                    or any(old['semantic_manifest'].get(k)!=semantic.get(k) for k in SCOPE)):
                raise ValueError('Only a strictly advancing generation of the same full source experiment may commit')
            generation=1 if old is None else old['generation']+1;slot=generation%2
            if shutil.disk_usage(root).free<maximum_bytes+minimum_free_bytes+4*1024**2:
                raise ValueError('Complete tensor state and durable external receipts must fit above the shared reserve')
            checkpoint=root/('state-slot-'+str(slot)+'.pt');receipt_path=root/('receipt-slot-'+str(slot)+'.json')
            atomic_checkpoint(checkpoint,state,maximum_bytes,minimum_free_bytes)
            receipt=dict(format=RECEIPT,at=datetime.now(timezone.utc).isoformat(),generation=generation,slot=slot,
                completed=stage.completed,checkpoint=str(checkpoint),checkpoint_sha256=file_hash(checkpoint),semantic_manifest=semantic,
                all_rank_private_states_in_shared_payload=True,candidate_promoted=False,goal_achieved=False)
            durable_json(receipt_path,receipt,minimum_free_bytes)
            pointer=dict(format=POINTER,generation=generation,slot=slot,completed=stage.completed,
                receipt=str(receipt_path),receipt_sha256=file_hash(receipt_path),goal_achieved=False)
            durable_json(root/'LATEST.json',pointer,minimum_free_bytes)
        except Exception as exc:error=type(exc).__name__+': '+str(exc)
    stage.coordinated(error,'durable tensor/receipt/pointer publication')
    values=stage.gather(receipt)
    if values[0] is None or any(v is not None for v in values[1:]):raise ValueError('Exactly root must publish one complete shared receipt')
    return values[0]


def restore_latest(stage,directory,*,expected_pointer_sha256=None,allow_completed=False):
    error=None
    try:state,receipt,pointer_sha=read_latest(directory,expected_pointer_sha256=expected_pointer_sha256)
    except Exception as exc:error=type(exc).__name__+': '+str(exc)
    stage.coordinated(error,'durable external checkpoint read before mutation')
    identity=dict(pointer_sha256=pointer_sha,checkpoint_sha256=receipt['checkpoint_sha256'],generation=receipt['generation'])
    if any(v!=identity for v in stage.gather(identity)):raise ValueError('Every replica must read the exact same committed shared state and receipt')
    stage.resume(state,expected_manifest=receipt['semantic_manifest'],allow_completed=allow_completed)
    return receipt
