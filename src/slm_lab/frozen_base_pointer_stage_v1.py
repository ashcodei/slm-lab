"""Original-query attention learning with an immutable complete own backbone.

Reuse the original per-query objective, epoch and fixed-order collectives.
Only two query/patch projection matrices train. Snapshots reference the own
base checkpoint instead of copying or restoring frozen backbone tensors.
"""
from dataclasses import asdict
import random

import numpy as np
import torch

from .codenet_checkpoint_v1 import restore_rng, save_collective
from .point_grounding_evaluation_v1 import state_digest
from .point_grounding_head_v1 import PointGroundingHead
from .point_grounding_stage_v1 import PointerStage, collective_check, cpu

FORMAT = 'own-frozen-base-pointer-head-resume-v1'
PROJECTIONS = ('query_projection.weight', 'patch_projection.weight')


def make_head(base, rank=256):
    head = PointGroundingHead(base, rank=rank)
    for parameter in head.base.parameters():
        parameter.requires_grad_(False)
    head.base.gradient_checkpointing = False
    head.base.eval()
    assert_partition(head)
    return head


def assert_partition(head):
    if (type(head) is not PointGroundingHead or any(p.requires_grad for p in head.base.parameters())
            or tuple(name for name, p in head.named_parameters() if p.requires_grad) != PROJECTIONS):
        raise ValueError('Exactly the two own head matrices train; every original backbone parameter stays frozen')


def projections(head):
    assert_partition(head)
    return {name: cpu(head.state_dict()[name]) for name in PROJECTIONS}


def make_optimizer(head, options):
    assert_partition(head)
    # Keep the original query-stage optimizer-group/schedule interface. The
    # vision group is explicitly empty and has no learning rate or Adam state.
    return torch.optim.AdamW([
        dict(params=[], lr=0., base_lr=0., name='vision'),
        dict(params=[*head.query_projection.parameters(), *head.patch_projection.parameters()],
             lr=options['head_learning_rate'], base_lr=options['head_learning_rate'], name='head')],
        betas=tuple(options['betas']), weight_decay=options['weight_decay'], foreach=False)


class FrozenBasePointerStage(PointerStage):
    def __init__(self, head, optimizer, data, options, sources, device, amp=None):
        if data.role != 'train' or data.purpose != 'optimization':
            raise ValueError('Only the original train optimization role may update the frozen-base head')
        assert_partition(head)
        if options.get('vision_learning_rate') != 0. or not sources.get('base_checkpoint_sha256'):
            raise ValueError('Explicit immutable own-base reference and zero vision learning rate required')
        super().__init__(head, optimizer, data, options, sources, device, amp)
        if len(self.parameters) != 2 or optimizer.param_groups[0]['params']:
            raise ValueError('Exactly two head matrices and an empty vision optimizer group required')
        self.frozen_base_digest = state_digest(head.base)
        self.initial_groups = cpu(optimizer.state_dict()['param_groups'])

    def update(self):
        assert_partition(self.head)
        return super().update()

    def save(self, path, maximum_bytes, minimum_free_bytes):
        collective_check(self.validate_Adam)
        def bundle(states):
            return dict(format=FORMAT, projections=projections(self.head), optimizer=cpu(self.optimizer.state_dict()),
                config=asdict(self.head.base.config), head_rank=self.head.rank,
                frozen_base_digest=self.frozen_base_digest, actual_inventory=self.head.trainable_inventory(),
                steps=self.completed, planned_steps=self.steps, world_size=self.world, training_options=self.options,
                sources=self.sources, sampler_state=self.sampler_state(), worker_rng_states=states,
                metrics=self.metrics, frozen_backbone_is_immutable_reference=True,
                candidate_promoted=False, goal_achieved=False)
        save_collective(path, bundle, self.device, self.world, self.rank, maximum_bytes, minimum_free_bytes)

    def resume(self, state):
        assert_partition(self.head)
        if (state.get('format') != FORMAT or state['sources'] != self.sources
                or state['training_options'] != self.options or state['world_size'] != self.world
                or state['config'] != asdict(self.head.base.config) or state['head_rank'] != self.head.rank
                or state['planned_steps'] != self.steps or type(state['steps']) is not int
                or not 0 < state['steps'] < self.steps or len(state['metrics']) != state['steps']
                or len(state['worker_rng_states']) != self.world
                or state['actual_inventory'] != self.head.trainable_inventory()
                or state['frozen_base_digest'] != self.frozen_base_digest
                or state_digest(self.head.base) != self.frozen_base_digest):
            raise ValueError('Exact own frozen-base/head/source/worker/optimizer recovery contract required')
        cursor = dict(self.sampler_state(), completed_steps=state['steps'],
                      source_exposures=state['steps'] * self.options['global_queries_per_update'])
        if cursor != state['sampler_state']:
            raise ValueError('Every exact original full-query source cursor must match')
        current = self.head.state_dict()
        if tuple(state['projections']) != PROJECTIONS:
            raise ValueError('Exactly both original query/patch head matrices required')
        for name, tensor in state['projections'].items():
            if (tensor.shape != current[name].shape or tensor.dtype != current[name].dtype
                    or not bool(torch.isfinite(tensor).all())):
                raise ValueError('Complete finite exact own projection tensors required')
        saved = state['optimizer']; groups = saved['param_groups']
        if len(groups) != 2:
            raise ValueError('Exact empty-vision/head Adam groups required')
        ids = []
        for group, initial in zip(groups, self.initial_groups):
            if ({k: v for k, v in group.items() if k != 'lr'} != {k: v for k, v in initial.items() if k != 'lr'}
                    or group['lr'] != group['base_lr'] * self.schedule(state['steps'] - 1)):
                raise ValueError('Exact original Adam options/order/schedule required')
            ids.extend(group['params'])
        if set(saved['state']) != set(ids) or len(ids) != 2:
            raise ValueError('Only both trained projection matrices belong in Adam')
        for identity, parameter in zip(ids, self.parameters):
            entry = saved['state'][identity]
            if (float(entry['step']) != state['steps']
                    or any(entry[k].shape != parameter.shape or entry[k].dtype != parameter.dtype
                           or not bool(torch.isfinite(entry[k]).all()) for k in ['exp_avg', 'exp_avg_sq'])
                    or bool((entry['exp_avg_sq'] < 0).any())):
                raise ValueError('Complete finite exact-step projection Adam moments required')
        for rng in state['worker_rng_states']:
            try:
                random.Random().setstate(rng['python'])
                torch.Generator(device='cpu').set_state(rng['torch'].cpu())
                n = rng['numpy']
                np.random.RandomState(0).set_state((n['kind'], np.asarray(n['keys'], dtype=np.uint32),
                                                 n['position'], n['has_gauss'], n['cached_gaussian']))
                if (rng['cuda'] is None) != (self.device.type != 'cuda'):
                    raise ValueError('Original CPU/CUDA RNG contract required')
                if self.device.type == 'cuda':
                    torch.Generator(device=self.device).set_state(rng['cuda'].cpu())
            except (KeyError, TypeError, ValueError, RuntimeError) as error:
                raise ValueError('Valid complete original worker RNG states required before mutation') from error
        # Restore only learned matrices and their optimizer. Frozen tensors
        # are never copied into themselves, preserving their version counters
        # as well as their bytes and existing conditional-code caches.
        self.head.query_projection.load_state_dict({'weight': state['projections']['query_projection.weight']})
        self.head.patch_projection.load_state_dict({'weight': state['projections']['patch_projection.weight']})
        self.optimizer.load_state_dict(state['optimizer'])
        self.completed = state['steps']; self.metrics = list(state['metrics'])
        restore_rng(state['worker_rng_states'][self.rank], self.device)
