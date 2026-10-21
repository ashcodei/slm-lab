"""Validate every private broad-code state before any replica restores it.

The actual immutable stage still performs the restoration. This preflight
coordinates source/shape/moment/RNG failures across the four real replicas.
"""
import hashlib
import math
from pathlib import Path

import torch
from torch import distributed as dist

from .broad_anchored_execution_stage_v1 import FORMAT as STAGE_FORMAT
from .code_residual_checkpoint_v1 import _validate_adam, _validate_rng
from .distributed_dense_code_step_v1 import agree
from .own_dense_actor_checkpoint_v1 import FORMAT as CORE_FORMAT, selected


def validate(stage, payload, device):
    stage.reference.assert_bound(stage.model)
    core = payload.get('core', {}); completed = core.get('completed')
    named = selected(stage.model, stage.optimizer, ('actor',))
    if (payload.get('format') != STAGE_FORMAT or payload.get('settings') != stage.settings
            or payload.get('corpus_provenance') != stage.corpus.provenance
            or payload.get('reference_binding') != stage.reference.binding()
            or payload.get('private_state_is_not_whole_distributed_checkpoint') is not True
            or payload.get('full_trajectory_records_require_external_durable_journal') is not True
            or payload.get('candidate_promoted') is not False or payload.get('goal_achieved') is not False
            or core.get('format') != CORE_FORMAT or core.get('world_size') != 1
            or core.get('groups') != ['actor'] or core.get('optimized_names') != [n for n, _ in named]
            or core.get('planned') != len(stage.order) or type(completed) is not int
            or not 0 < completed < len(stage.order) or core.get('cursor') != stage.cursor(completed)
            or not isinstance(core.get('metrics'), list) or len(core['metrics']) != completed
            or core.get('candidate_promoted') is not False or core.get('goal_achieved') is not False):
        raise ValueError('Exact private stage, original source cursor, objective and actor partition required')
    for i, record in enumerate(core['metrics']):
        position = stage.order[i]; row = stage.corpus.row(position)
        if (record.get('step') != i+1 or record.get('position') != position or record.get('source_id') != row['id']
                or record.get('original_reference_used_regardless_of_rewards') is not True
                or record.get('source_reference_targets') != row['response_EOS_targets']
                or record.get('method') != stage.settings['method']
                or record.get('reference_binding') != stage.reference.binding()
                or len(record.get('global_rewards', [])) != stage.settings['world']*stage.settings['local_group_size']
                or any(not math.isfinite(value) or not 0 <= value <= 1 for value in record['global_rewards'])
                or len(record.get('global_trajectory_semantic_sha256', '')) != 64):
            raise ValueError('Every original consumed source and anchored objective record must reconstruct')
    stage.model.validate_export(core['model'], expected_sources=stage.sources)
    _validate_adam(core.get('optimizer'), [p for _, p in named], completed)
    _validate_rng(core.get('rng'), device)
    if core['optimizer']['param_groups'] != stage.optimizer.state_dict()['param_groups']:
        raise ValueError('Every exact original actor Adam option must match before any replica mutates')
    optimized = {n for n, _ in named}
    for name, value in stage.model.parts.state_dict().items():
        if name not in optimized and not torch.equal(value.detach().cpu(), core['model']['parts'][name]):
            raise ValueError('All unused actor/critic tensors must remain exact')


def restore(stage, path, *, expected_sha256, object_group=None):
    if (not dist.is_initialized() or dist.get_world_size() != 4
            or stage.settings['world'] != 4 or stage.settings['rank'] != dist.get_rank()):
        raise ValueError('Every exact original four-replica rank is required')
    error = None; payload = None
    try:
        with Path(path).open('rb') as pinned:
            actual = hashlib.file_digest(pinned, 'sha256').hexdigest()
            if actual != expected_sha256:
                raise ValueError('The actual original private checkpoint SHA differs')
            payload = torch.load('/proc/self/fd/'+str(pinned.fileno()), map_location='cpu', weights_only=True, mmap=True)
            validate(stage, payload, next(stage.model.parameters()).device)
            pinned.seek(0)
            if hashlib.file_digest(pinned, 'sha256').hexdigest() != actual:
                raise ValueError('The original pinned private file changed during validation')
    except Exception as exc:
        error = repr(exc)
    agree(error, object_group)
    error = None
    try:
        stage.resume(payload)
    except Exception as exc:
        error = repr(exc)
    agree(error, object_group)
    return stage.completed
