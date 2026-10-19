"""Two-generation durable visual state with one common tensor payload.

All replicas first verify identical selected weights/Adam and global cursors.
Only private per-rank RNG/metrics are gathered. Root publishes common tensors,
an external semantic receipt, then LATEST. A failed receipt/pointer publication
keeps the preceding complete generation readable. No training admission.
"""
import fcntl
import json
from pathlib import Path
import shutil

import torch
from torch import distributed as dist

from .codenet_checkpoint_v1 import atomic_checkpoint
from .corpus import file_hash
from .joint_visual_private_checkpoint_v1 import (FORMAT as PRIVATE, digest, make_state, restore_state,
    private_rng_sha256, _common, _control, _agree)
from .parallel_repair_durable_state_v1 import durable_json, regular

FORMAT = 'own-joint-visual-one-common-selected-state-all-private-ranks-v1'
RECEIPT = FORMAT + '-receipt'
POINTER = FORMAT + '-latest'
PRIVATE_FIELDS = {'rank', 'metrics', 'private_RNG', 'private_RNG_sha256'}
COMMON_FIELDS = {'format', 'sources', 'mode', 'world_size', 'model_metadata', 'cursor',
    'optimized_names', 'selected_weights', 'optimizer', 'candidate_promoted', 'goal_achieved'}


def _gather(value, group, world):
    if world == 1:
        return [value]
    rows = [None] * world
    dist.all_gather_object(rows, value, group=group)
    return rows


def manifest(shared):
    if not isinstance(shared, dict) or set(shared) != {'format', 'common', 'private_states'} or shared['format'] != FORMAT:
        raise ValueError('One exact common-tensor/all-private-rank visual payload required')
    common, private = shared['common'], shared['private_states']
    if (set(common) != COMMON_FIELDS or common['format'] != PRIVATE
            or type(common['world_size']) is not int or not 1 <= common['world_size'] <= 4
            or not isinstance(private, list) or len(private) != common['world_size']
            or common['candidate_promoted'] is not False or common['goal_achieved'] is not False):
        raise ValueError('Exact shared visual format, actual world and nonpromoted source state required')
    private_rows = []
    for rank, value in enumerate(private):
        if (set(value) != PRIVATE_FIELDS or value['rank'] != rank
                or value['private_RNG_sha256'] != private_rng_sha256(value['private_RNG'])):
            raise ValueError('Every exact private rank and actual RNG boundary hash required')
        private_rows.append(dict(rank=rank, private_RNG_sha256=value['private_RNG_sha256'],
            complete_actual_private_metrics_sha256=digest(value['metrics'])))
    settings = [{k: v for k, v in group.items() if k != 'lr'} for group in common['optimizer']['param_groups']]
    context = dict(sources=common['sources'], mode=common['mode'], world=common['world_size'],
        model_metadata=common['model_metadata'], planned=common['cursor']['planned'],
        whole_schedule_entries_sha256=common['cursor']['whole_schedule_entries_sha256'],
        fixed_Adam_settings=settings)
    return dict(format=FORMAT, completed=common['cursor']['completed'], planned=common['cursor']['planned'],
        context_sha256=digest(context), actual_common_weights_Adam_global_cursor_sha256=_common(common),
        all_private_rank_boundaries=private_rows)


def read_receipt(directory, *, expected_pointer_sha256=None):
    root = Path(directory).resolve()
    pointer_path = root / 'LATEST.json'
    regular(pointer_path)
    pointer_sha = file_hash(pointer_path)
    if expected_pointer_sha256 is not None and pointer_sha != expected_pointer_sha256:
        raise ValueError('The exact externally retained visual commit pointer changed')
    pointer = json.loads(pointer_path.read_text())
    generation, completed, slot = pointer.get('generation'), pointer.get('completed'), pointer.get('slot')
    if (pointer.get('format') != POINTER or type(generation) is not int or generation < 1
            or type(completed) is not int or completed < 0 or slot != generation % 2):
        raise ValueError('Exact complete visual generation/cursor required')
    receipt_path = root / ('receipt-slot-' + str(slot) + '.json')
    regular(receipt_path)
    if Path(pointer.get('receipt', '')).resolve() != receipt_path or file_hash(receipt_path) != pointer.get('receipt_sha256'):
        raise ValueError('The exact committed external visual receipt differs')
    receipt = json.loads(receipt_path.read_text())
    checkpoint = root / ('state-slot-' + str(slot) + '.pt')
    regular(checkpoint)
    if (receipt.get('format') != RECEIPT or receipt.get('generation') != generation or receipt.get('slot') != slot
            or receipt.get('completed') != completed or Path(receipt.get('checkpoint', '')).resolve() != checkpoint
            or receipt.get('semantic_manifest', {}).get('completed') != completed
            or file_hash(checkpoint) != receipt.get('checkpoint_sha256')):
        raise ValueError('The exact shared visual tensor file, receipt and external cursor must agree')
    if file_hash(pointer_path) != pointer_sha:
        raise ValueError('Visual LATEST changed during readback; inspect the same actual controller')
    return receipt, pointer_sha


def read_latest(directory, *, expected_pointer_sha256=None):
    receipt, pointer_sha = read_receipt(directory, expected_pointer_sha256=expected_pointer_sha256)
    shared = torch.load(receipt['checkpoint'], map_location='cpu', weights_only=True, mmap=True)
    if manifest(shared) != receipt['semantic_manifest']:
        raise ValueError('Actual visual weights/Adam and every private RNG/metric must equal the external manifest')
    if file_hash(Path(directory) / 'LATEST.json') != pointer_sha:
        raise ValueError('The same exact visual commit must remain current throughout the tensor read')
    return shared, receipt, pointer_sha


def commit(view, optimizer, fixed, directory, *, sources, entries, completed, metrics, mode,
           maximum_bytes, minimum_free_bytes, rank=0, world=1, device='cpu', control_group=None):
    actual_rank, actual_world = _control(control_group)
    error = specification = None
    try:
        if (type(maximum_bytes) is not int or maximum_bytes < 1
                or type(minimum_free_bytes) is not int or minimum_free_bytes < 0):
            raise ValueError('Explicit complete tensor and other-writer reserve bounds required')
        root = Path(directory).resolve()
        specification = digest(dict(directory=str(root), maximum_bytes=maximum_bytes,
            minimum_free_bytes=minimum_free_bytes, completed=completed))
    except Exception as exc:
        error = type(exc).__name__ + ': ' + str(exc)
    _agree(error, specification, control_group, actual_world)
    state = make_state(view, optimizer, fixed, sources=sources, entries=entries, completed=completed,
        metrics=metrics, mode=mode, rank=rank, world=world, device=device, control_group=control_group)
    private = {key: state[key] for key in PRIVATE_FIELDS}
    private_rows = _gather(private, control_group, actual_world)
    # The common selected arrays/moments were already checked identical by
    # make_state. No634MiBreplica copies are sent through object collectives.
    shared = dict(format=FORMAT, common={k: state[k] for k in COMMON_FIELDS}, private_states=private_rows)
    error = semantic = None
    try:
        semantic = manifest(shared)
    except Exception as exc:
        error = type(exc).__name__ + ': ' + str(exc)
    _agree(error, None if semantic is None else digest(semantic), control_group, actual_world)
    receipt = None
    error = None
    if actual_rank == 0:
        try:
            root.mkdir(parents=True, exist_ok=True)
            with (root / 'commit.lock').open('a+') as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                old = read_receipt(root)[0] if (root / 'LATEST.json').exists() else None
                if old is not None and (old['completed'] >= completed
                        or old['semantic_manifest']['context_sha256'] != semantic['context_sha256']):
                    raise ValueError('Only a strictly advancing complete generation of the same full visual experiment may commit')
                if shutil.disk_usage(root).free < maximum_bytes + minimum_free_bytes + 4 * 1024**2:
                    raise ValueError('The complete common tensor state and external metadata must fit above the reserve')
                generation = 1 if old is None else old['generation'] + 1
                slot = generation % 2
                checkpoint = root / ('state-slot-' + str(slot) + '.pt')
                receipt_path = root / ('receipt-slot-' + str(slot) + '.json')
                atomic_checkpoint(checkpoint, shared, maximum_bytes, minimum_free_bytes)
                receipt = dict(format=RECEIPT, generation=generation, slot=slot, completed=completed,
                    checkpoint=str(checkpoint), checkpoint_sha256=file_hash(checkpoint), semantic_manifest=semantic,
                    common_selected_weights_and_Adam_stored_once=True,
                    frozen_language_actor_critic_weights_not_stored=True, candidate_promoted=False, goal_achieved=False)
                durable_json(receipt_path, receipt, minimum_free_bytes)
                pointer = dict(format=POINTER, generation=generation, slot=slot, completed=completed,
                    receipt=str(receipt_path), receipt_sha256=file_hash(receipt_path), goal_achieved=False)
                durable_json(root / 'LATEST.json', pointer, minimum_free_bytes)
        except Exception as exc:
            error = type(exc).__name__ + ': ' + str(exc)
    _agree(error, None, control_group, actual_world)
    receipts = _gather(receipt, control_group, actual_world)
    if receipts[0] is None or any(r is not None for r in receipts[1:]):
        raise ValueError('Exactly the actual root must publish a complete shared visual state')
    return receipts[0]


def restore_latest(view, optimizer, fixed, directory, *, sources, entries, mode, rank=0, world=1,
                   device='cpu', control_group=None, expected_pointer_sha256=None, allow_completed=False):
    actual_rank, actual_world = _control(control_group)
    error = identity = shared = receipt = None
    try:
        shared, receipt, pointer_sha = read_latest(directory, expected_pointer_sha256=expected_pointer_sha256)
        if shared['common']['world_size'] != actual_world:
            raise ValueError('The committed visual state must name the actual complete worker population')
        identity = digest(dict(pointer_sha256=pointer_sha, checkpoint_sha256=receipt['checkpoint_sha256'],
            generation=receipt['generation'], completed=receipt['completed']))
    except Exception as exc:
        error = type(exc).__name__ + ': ' + str(exc)
    _agree(error, identity, control_group, actual_world)
    private = shared['private_states'][actual_rank]
    expected = receipt['semantic_manifest']['all_private_rank_boundaries'][actual_rank]
    local = {**shared['common'], **private}
    completed, metrics = restore_state(view, optimizer, fixed, local, sources=sources, entries=entries,
        mode=mode, rank=rank, world=world, device=device, control_group=control_group,
        expected_completed=receipt['completed'], expected_private_RNG_sha256=expected['private_RNG_sha256'],
        allow_completed=allow_completed)
    return completed, metrics, receipt
