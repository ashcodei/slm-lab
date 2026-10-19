"""Actual experiment loops shared by small neural controls and owned GPU runs.

Recovery replays one real update, including private random draws, from the
external durable commit. Capacity calls are never counted as production.
"""
import copy
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from .codenet_checkpoint_v1 import capture_rng
from .parallel_function_repair_stage_v1 import manifest,value_digest
from .parallel_function_repair_v1 import signature
from .parallel_repair_durable_state_v1 import commit,read_latest,restore_latest
from .whole_human_sft_evidence_v1 import tensor_digest


def private_draws(device):
    values=[random.random(),float(np.random.random()),float(torch.rand(()))]
    if torch.device(device).type=='cuda':values.append(float(torch.rand((),device=device)))
    return values


def live(stage):
    return dict(parameters=signature(stage.model.parameters()),parts=tensor_digest(stage.model.parts.named_parameters()),
        optimizer=value_digest(stage.optimizer.state_dict()),rng=value_digest(capture_rng(stage.device)),
        Adam_identity=[(id(p),k,id(t),t.data_ptr(),t._version) for p,row in stage.optimizer.state.items() for k,t in row.items()],
        completed=stage.completed,metrics=value_digest(stage.metrics),failed=stage.failed)


def prove_next_update(stage,directory,*,maximum_bytes,minimum_free_bytes):
    """Run first, second, reject corrupt recovery, restore first, replay second."""
    if stage.mode!='capacity' or stage.completed or len(stage.entries)<3:
        raise ValueError('A fresh actual capacity schedule including a real tail is required')
    first=stage.update();commit(stage,directory,maximum_bytes=maximum_bytes,minimum_free_bytes=minimum_free_bytes)
    saved,receipt,pointer_sha=read_latest(directory)
    wanted_draws=private_draws(stage.device);second=stage.update();before=live(stage)
    bad=dict(saved,model=dict(saved['model'],sources={**saved['model']['sources'],'invalid_source_sha256':'0'*64}))
    corruptions=[(bad,manifest(bad))]
    bad=dict(saved,private_workers=copy.deepcopy(saved['private_workers']))
    field='cuda' if stage.device.type=='cuda' else 'torch'
    bad['private_workers'][-1]['rng'][field]=torch.zeros(1,dtype=torch.uint8)
    corruptions.append((bad,manifest(bad)))
    if stage.world>1:
        bad=dict(saved,private_workers=copy.deepcopy(saved['private_workers']))
        bad['private_workers'][0]['rng']=copy.deepcopy(saved['private_workers'][1]['rng'])
        corruptions.append((bad,receipt['semantic_manifest']))
    for bad,expected in corruptions:
        try:stage.resume(bad,expected_manifest=expected)
        except ValueError:pass
        else:raise ValueError('Bad source/private RNG must reject before any live mutation')
        if live(stage)!=before:raise ValueError('Rejected recovery changed live tensors, moments, sources or private RNG')
    restore_latest(stage,directory,expected_pointer_sha256=pointer_sha)
    if private_draws(stage.device)!=wanted_draws or stage.update()!=second:
        raise ValueError('Actual next private draws and complete global source update must replay exactly')
    after=live(stage)
    if any(after[k]!=before[k] for k in ['parts','optimizer','rng','completed','metrics','failed']):
        raise ValueError('Actual next actor/Adam/metrics/private RNG must replay bit exactly')
    return dict(first_actual_record=first,replayed_second_actual_record=second,
        exact_next_draws_global_update_actor_Adam_and_private_RNG=True,
        invalid_source_and_private_RNG_rejected_before_mutation=True,
        wrong_valid_peer_RNG_rejected_before_mutation=stage.world>1,
        physical_optimizer_calls=3,logical_updates=2,external_commit_pointer_sha256=pointer_sha)


def train(stage,directory,journal,*,maximum_bytes,minimum_free_bytes,every,preflight,progress):
    """Fresh full schedule, durable first/every/final commits; no implicit retry."""
    if stage.mode!='train' or stage.completed or type(every) is not int or every<1:
        raise ValueError('Only a fresh complete production schedule may enter this loop')
    stream=Path(journal).open('x') if stage.rank==0 else None;seconds=[];last_receipt=None
    try:
        for _ in stage.entries:
            preflight();start=time.monotonic();row=stage.update();elapsed=time.monotonic()-start;seconds.append(elapsed)
            error=None
            if stage.rank==0:
                try:stream.write(json.dumps(row)+'\n');stream.flush()
                except Exception as exc:error=repr(exc)
            stage.coordinated(error,'complete production record write')
            if stage.completed==1 or stage.completed%every==0 or stage.completed==len(stage.entries):
                error=None
                if stage.rank==0:
                    try:os.fsync(stream.fileno())
                    except Exception as exc:error=repr(exc)
                stage.coordinated(error,'production journal fsync')
                last_receipt=commit(stage,directory,maximum_bytes=maximum_bytes,minimum_free_bytes=minimum_free_bytes)
            error=None
            if stage.rank==0:
                try:progress(row,elapsed)
                except Exception as exc:error=repr(exc)
            stage.coordinated(error,'production progress publication')
        if last_receipt is None or last_receipt['completed']!=len(stage.entries):
            raise ValueError('A durable complete final source prefix is required')
        return dict(actual_complete_updates=stage.completed,actual_update_seconds=seconds,final_shared_receipt=last_receipt)
    finally:
        if stream is not None:stream.close()
