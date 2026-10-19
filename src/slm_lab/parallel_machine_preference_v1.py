"""Complete execution preferences with one global normalization on four replicas.

Rejected programs are preference-only targets. The reference snapshots our
actual starting actor; it is neither a human label nor a pretrained teacher.
"""
import math

import torch
from torch import distributed as dist

from .frozen_own_actor_reference_v1 import signature
from .machine_code_preference_stage_v1 import (
    MachineCodePreferenceStage, gradients, pair_information, validate_metric)

FORMAT = 'own-private-replica-full-machine-preference-v1'


def local_gradients(model, data, positions, reference, options, *, rank=0,
                    world=1, device='cpu', amp=None):
    if (type(world) is not int or world < 1 or type(rank) is not int
            or not 0 <= rank < world or not positions
            or len(set(positions)) != len(positions)
            or any(type(p) is not int or not 0 <= p < len(data) for p in positions)):
        raise ValueError('Distinct complete global machine positions and exact replica identity required')
    information = [pair_information(data, p) for p in positions]
    selected = positions[rank::world]
    if not selected:
        reference.assert_bound(model)
        for _, parameter in model.named_selected(('actor',)):
            if parameter.grad is not None:
                raise ValueError('An empty tail replica cannot contain stale gradients')
            parameter.grad = torch.zeros_like(parameter)
        return []
    result = gradients(model, data, selected, reference, method=options['method'],
        beta=options['beta'], chosen_CE_weight=options['chosen_CE_weight'],
        device=device, amp=amp, normalization_pairs=len(positions),
        normalization_chosen_targets=sum(r['chosen_response_EOS_targets'] for r in information))
    return result['original_machine_pairs']


class ParallelMachinePreferenceStage(MachineCodePreferenceStage):
    def __init__(self, model, optimizer, data, reference, options, sources, *,
                 world=1, rank=0, gradient_group=None, control_group=None,
                 amp=None, CPU_order=None):
        if (type(world) is not int or world < 1 or type(rank) is not int
                or not 0 <= rank < world or (world > 1 and
                (not dist.is_initialized() or dist.get_world_size(gradient_group) != world
                 or dist.get_world_size(control_group) != world))):
            raise ValueError('Exact initialized machine preference replicas required')
        super().__init__(model, optimizer, data, reference, options, sources, CPU_order=CPU_order)
        self.world, self.rank = world, rank
        self.gradient_group, self.control_group = gradient_group, control_group
        self.amp = amp

    def gather(self, value):
        if self.world == 1:
            return [value]
        result = [None] * self.world
        dist.all_gather_object(result, value, group=self.control_group)
        return result

    def cursor(self, completed):
        return dict(super().cursor(completed), replica_rank=self.rank, world_size=self.world)

    def update(self):
        positions = self.positions(self.completed)
        self.reference.assert_bound(self.model)
        if signature(self.fixed) != self.fixed_signature:
            raise ValueError('Every frozen dense, vision, word and unused critic value must stay exact')
        identity = dict(options=self.options, sources=self.sources, positions=positions,
            pairs=[pair_information(self.data, p) for p in positions], reference=self.reference.binding())
        if any(value != identity for value in self.gather(identity)):
            raise ValueError('Every replica must agree on full original programs, global denominators and own reference')
        self.optimizer.zero_grad(set_to_none=True)
        try:
            local = local_gradients(self.model, self.data, positions, self.reference,
                self.options, rank=self.rank, world=self.world, device=self.device, amp=self.amp)
            if (any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in self.parameters)
                    or any(p.grad is not None for p in self.fixed)):
                raise ValueError('Every actor and no frozen base/head/critic leaf must have finite gradients')
            if self.world > 1:
                flat = torch.cat([p.grad.reshape(-1) for p in self.parameters])
                dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=self.gradient_group)
                offset = 0
                for parameter in self.parameters:
                    parameter.grad.copy_(flat[offset:offset+parameter.numel()].view_as(parameter))
                    offset += parameter.numel()
                del flat
            norm = float(torch.nn.utils.clip_grad_norm_(self.parameters,
                self.options['gradient_clip'], error_if_nonfinite=True))
            rows = [r for group in self.gather(local) for r in group]
            by_position = {r['original_train_position']: r for r in rows}
            if len(rows) != len(positions) or set(by_position) != set(positions):
                raise ValueError('Every complete original machine pair must occur exactly once')
            rows = [by_position[p] for p in positions]
            chosen = sum(r['chosen_response_EOS_targets'] for r in rows)
            logp = math.fsum(r['policy_chosen_logp'] for r in rows)
            mean = math.fsum(r['pair_DPO'] for r in rows) / len(positions)
            record = dict(method=self.options['method'], original_source_positions=positions,
                original_machine_pairs=rows, beta=self.options['beta'],
                chosen_CE_weight=self.options['chosen_CE_weight'], normalization_pairs=len(positions),
                normalization_chosen_targets=chosen, chosen_response_EOS_targets=chosen,
                rejected_response_EOS_targets=sum(r['rejected_response_EOS_targets'] for r in rows),
                complete_context_tokens=sum(r['complete_context_tokens'] for r in rows),
                pair_mean_DPO=mean, chosen_token_mean_NLL=-logp/chosen,
                summed_objective=mean-self.options['chosen_CE_weight']*logp/chosen,
                own_starting_actor_reference_queried=self.options['method'] == 'machine_DPO',
                human_preferences=False, rejected_programs_never_positive_CE_targets=True,
                critic_and_frozen_dense_not_trained=True, step=self.completed+1,
                learning_rate=self.options['learning_rate'], gradient_norm_before_clip=norm,
                global_pair_and_chosen_token_denominators=True,
                gradient_reduction='SUM_without_second_world_average',
                per_replica_original_pairs=[len(positions[r::self.world]) for r in range(self.world)])
            validate_metric(self.data, positions, self.options, record, self.completed)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
        except BaseException:
            self.optimizer.zero_grad(set_to_none=True)
            raise
        self.completed += 1
        self.metrics.append(record)
        self.reference.assert_bound(self.model)
        return record

    def state(self):
        return dict(format=FORMAT, core=super().state(), replicas=self.world, replica_rank=self.rank,
            private_state_is_not_whole_distributed_checkpoint=True, goal_achieved=False)

    def resume(self, state, *, allow_completed=False):
        if (state.get('format') != FORMAT or state.get('replicas') != self.world
                or state.get('replica_rank') != self.rank
                or state.get('private_state_is_not_whole_distributed_checkpoint') is not True
                or state.get('goal_achieved') is not False):
            raise ValueError('Exact private machine replica and all-worker recovery required')
        for record in state.get('core', {}).get('core', {}).get('metrics', []):
            positions = record['original_source_positions']
            if (record.get('global_pair_and_chosen_token_denominators') is not True
                    or record.get('gradient_reduction') != 'SUM_without_second_world_average'
                    or record.get('per_replica_original_pairs') !=
                    [len(positions[r::self.world]) for r in range(self.world)]):
                raise ValueError('Actual global denominators and every empty-tail replica count must reconstruct')
        return super().resume(state['core'], allow_completed=allow_completed)
