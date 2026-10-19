"""Two-generation joint commits storing selected values/Adam exactly once.

All ranks validate common state before root writes. Each actual rank's private
RNG is retained independently. An external receipt and LATEST commit follow
the bounded, fsynced tensor write. No full-size training admission is granted.
"""
import fcntl
import json
from pathlib import Path
import shutil

import torch

from .codenet_checkpoint_v1 import atomic_checkpoint
from .corpus import file_hash
from .parallel_function_repair_stage_v1 import value_digest
from .parallel_repair_durable_state_v1 import durable_json, regular
from .joint_indexed_checkpoint_v2 import (FORMAT as PRIVATE, agree, common_digest, control,
    gather, make_state, restore_state)

FORMAT = 'own-indexed-joint-one-common-selected-Adam-all-private-RNG-v1'
RECEIPT = FORMAT + '-receipt'
POINTER = FORMAT + '-latest'
PRIVATE_FIELDS = {'rank', 'private_RNG', 'private_RNG_sha256'}
COMMON_FIELDS = {'format', 'sources', 'world_size', 'model_metadata', 'cursor', 'history_receipts', 'history_configuration',
    'optimized_names', 'selected_weights', 'optimizer', 'grants_production_training_admission', 'candidate_promoted', 'goal_achieved'}


def manifest(shared):
    if type(shared) is not dict or set(shared) != {'format', 'common', 'private_states'} or shared['format'] != FORMAT:
        raise ValueError('One exact common-tensor/all-private-rank joint payload is required')
    common, private = shared['common'], shared['private_states']
    if (type(common) is not dict or set(common) != COMMON_FIELDS or common['format'] != PRIVATE
            or type(common['world_size']) is not int or not 1 <= common['world_size'] <= 4
            or type(private) is not list or len(private) != common['world_size']
            or any(common[k] is not False for k in ['grants_production_training_admission', 'candidate_promoted', 'goal_achieved'])):
        raise ValueError('The exact actual joint world and unpromoted complete state are required')
    private_hashes = []
    for rank, row in enumerate(private):
        if (type(row) is not dict or set(row) != PRIVATE_FIELDS or type(row['rank']) is not int or row['rank'] != rank
                or row['private_RNG_sha256'] != value_digest(row['private_RNG'])):
            raise ValueError('Every actual private rank and complete generator hash is required')
        private_hashes.append(row['private_RNG_sha256'])
    context = {k: common[k] for k in ['sources', 'world_size', 'model_metadata', 'optimized_names', 'history_configuration']}
    context['Adam_settings'] = common['optimizer']['param_groups']
    return dict(format=FORMAT, completed=common['cursor']['completed'], planned=common['cursor']['planned'],
        context_sha256=value_digest(context), actual_common_sha256=common_digest(common),
        all_actual_private_RNG_sha256=private_hashes)


def read_receipt(directory, *, expected_pointer_sha256=None):
    root = Path(directory).resolve(); pointer_path = root / 'LATEST.json'; regular(pointer_path)
    pointer_sha = file_hash(pointer_path)
    if expected_pointer_sha256 is not None and pointer_sha != expected_pointer_sha256:
        raise ValueError('The exact externally retained joint LATEST identity changed')
    pointer = json.loads(pointer_path.read_text()); generation = pointer.get('generation'); slot = pointer.get('slot')
    completed = pointer.get('completed')
    if (pointer.get('format') != POINTER or type(generation) is not int or generation < 1
            or slot != generation % 2 or type(completed) is not int or completed < 0):
        raise ValueError('The exact complete joint generation and cursor are required')
    receipt_path = root / ('receipt-slot-' + str(slot) + '.json'); regular(receipt_path)
    if Path(pointer.get('receipt', '')).resolve() != receipt_path or file_hash(receipt_path) != pointer.get('receipt_sha256'):
        raise ValueError('The exact committed external joint receipt differs')
    receipt = json.loads(receipt_path.read_text()); checkpoint = root / ('state-slot-' + str(slot) + '.pt'); regular(checkpoint)
    if (receipt.get('format') != RECEIPT or receipt.get('generation') != generation or receipt.get('slot') != slot
            or receipt.get('completed') != completed or Path(receipt.get('checkpoint', '')).resolve() != checkpoint
            or receipt.get('semantic_manifest', {}).get('completed') != completed
            or file_hash(checkpoint) != receipt.get('checkpoint_sha256')):
        raise ValueError('The exact shared joint tensor file, external receipt and cursor must agree')
    if file_hash(pointer_path) != pointer_sha: raise ValueError('Joint LATEST changed during the same readback')
    return receipt, pointer_sha


def read_latest(directory, *, expected_pointer_sha256=None):
    receipt, pointer_sha = read_receipt(directory, expected_pointer_sha256=expected_pointer_sha256)
    shared = torch.load(receipt['checkpoint'], map_location='cpu', weights_only=True, mmap=True)
    if manifest(shared) != receipt['semantic_manifest']:
        raise ValueError('Every joint tensor, Adam state, source metric and private generator must match the external manifest')
    if file_hash(Path(directory) / 'LATEST.json') != pointer_sha: raise ValueError('Joint LATEST changed during the tensor read')
    return shared, receipt, pointer_sha


def commit(view, optimizer, fixed, directory, *, sources, entries, completed, history_receipts, history_configuration, source_files, resource_guard,
           maximum_bytes, minimum_free_bytes, rank=0, world=1, device='cpu', control_group=None):
    actual_rank, actual_world = control(control_group); error = specification = None
    try:
        if (type(maximum_bytes) is not int or maximum_bytes < 1
                or type(minimum_free_bytes) is not int or minimum_free_bytes < 0):
            raise ValueError('Explicit complete joint tensor and other-writer reserve bounds are required')
        root = Path(directory).resolve()
        specification = value_digest(dict(directory=str(root), maximum_bytes=maximum_bytes,
            minimum_free_bytes=minimum_free_bytes, completed=completed))
    except Exception as exc: error = type(exc).__name__ + ': ' + str(exc)
    agree(error, specification, control_group, actual_world)
    local = make_state(view, optimizer, fixed, sources=sources, entries=entries, completed=completed,
        history_receipts=history_receipts,history_configuration=history_configuration,source_files=source_files,resource_guard=resource_guard, rank=rank, world=world, device=device, control_group=control_group)
    private = gather({k: local[k] for k in PRIVATE_FIELDS}, control_group, actual_world)
    shared = dict(format=FORMAT, common={k: local[k] for k in COMMON_FIELDS}, private_states=private)
    error = semantic = None
    try: semantic = manifest(shared)
    except Exception as exc: error = type(exc).__name__ + ': ' + str(exc)
    agree(error, None if semantic is None else value_digest(semantic), control_group, actual_world)
    error = receipt = None
    if actual_rank == 0:
        try:
            root.mkdir(parents=True, exist_ok=True)
            with (root / 'commit.lock').open('a+') as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                old = read_receipt(root)[0] if (root / 'LATEST.json').exists() else None
                if old is not None and (old['completed'] >= completed
                        or old['semantic_manifest']['context_sha256'] != semantic['context_sha256']):
                    raise ValueError('Only a strictly advancing generation of the same complete joint experiment may commit')
                if shutil.disk_usage(root).free < maximum_bytes + minimum_free_bytes + 4 * 1024**2:
                    raise ValueError('The complete joint tensor and receipts must fit above the shared reserve')
                generation = 1 if old is None else old['generation'] + 1; slot = generation % 2
                checkpoint = root / ('state-slot-' + str(slot) + '.pt'); receipt_path = root / ('receipt-slot-' + str(slot) + '.json')
                atomic_checkpoint(checkpoint, shared, maximum_bytes, minimum_free_bytes)
                receipt = dict(format=RECEIPT, generation=generation, slot=slot, completed=completed,
                    checkpoint=str(checkpoint), checkpoint_sha256=file_hash(checkpoint), semantic_manifest=semantic,
                    common_selected_weights_and_Adam_stored_once=True, frozen_language_critic_other_heads_not_stored=True,
                    grants_production_training_admission=False, candidate_promoted=False, goal_achieved=False)
                durable_json(receipt_path, receipt, minimum_free_bytes)
                durable_json(root / 'LATEST.json', dict(format=POINTER, generation=generation, slot=slot,
                    completed=completed, receipt=str(receipt_path), receipt_sha256=file_hash(receipt_path), goal_achieved=False), minimum_free_bytes)
        except Exception as exc: error = type(exc).__name__ + ': ' + str(exc)
    agree(error, None, control_group, actual_world)
    receipts = gather(receipt, control_group, actual_world)
    if receipts[0] is None or any(r is not None for r in receipts[1:]): raise ValueError('Exactly actual root must publish the joint receipt')
    return receipts[0]


def restore_latest(view, optimizer, fixed, directory, *, sources, entries, rank=0, world=1,
                   device='cpu', control_group=None, expected_pointer_sha256=None, allow_completed=False,history_configuration,source_files,resource_guard):
    actual_rank, actual_world = control(control_group); error = identity = shared = receipt = None
    try:
        shared, receipt, pointer_sha = read_latest(directory, expected_pointer_sha256=expected_pointer_sha256)
        if shared['common']['world_size'] != actual_world: raise ValueError('The actual complete joint worker population must match')
        identity = value_digest(dict(pointer_sha256=pointer_sha, checkpoint_sha256=receipt['checkpoint_sha256'],
            generation=receipt['generation'], completed=receipt['completed']))
    except Exception as exc: error = type(exc).__name__ + ': ' + str(exc)
    agree(error, identity, control_group, actual_world)
    local = {**shared['common'], **shared['private_states'][actual_rank]}
    completed, history_receipts = restore_state(view, optimizer, fixed, local, sources=sources, entries=entries,
        expected_common_sha256=receipt['semantic_manifest']['actual_common_sha256'],
        expected_private_RNG_sha256=receipt['semantic_manifest']['all_actual_private_RNG_sha256'][actual_rank],
        expected_completed=receipt['completed'], history_configuration=history_configuration,source_files=source_files,resource_guard=resource_guard,rank=rank, world=world, device=device,
        control_group=control_group, allow_completed=allow_completed)
    return completed, history_receipts, receipt
