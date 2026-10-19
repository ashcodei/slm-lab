"""Whole human pairs with global pair/token denominators and gradient SUM.

The same counted frozen own backbone provides the DPO reference. Public human
rankings of other-model responses are not new judgments of our model outputs.
An empty tail replica contributes zero gradients without dropping any pair.
"""
from contextlib import nullcontext
import math

import torch
from torch import distributed as dist
from torch.nn import functional as F

from .dense_human_preference_v1 import pair_information, sequence_logp
from .dense_human_preference_stage_v1 import (
    DenseHumanPreferenceStage, actual_exposures, signature, validate_metric)
from .own_dense_actor_v1 import OwnDenseActor
from .whole_human_data_v2 import WholeHumanData

FORMAT = 'own-private-replica-complete-human-DPO-global-normalization-v2'


def local_gradients(model, data, positions, *, rank=0, world=1, DPO_weight=1,
                    beta=.1, chosen_CE_weight=.1, device='cpu', amp=None):
    if (type(model) is not OwnDenseActor or type(data) is not WholeHumanData
            or (data.family, data.role, data.purpose) != ('preferences', 'train', 'optimization')
            or not positions or len(set(positions)) != len(positions)
            or any(type(p) is not int or not 0 <= p < len(data) for p in positions)
            or type(world) is not int or world < 1 or type(rank) is not int or not 0 <= rank < world
            or type(DPO_weight) is not int or DPO_weight not in (0, 1)
            or not math.isfinite(beta) or not 0 < beta <= 1
            or not math.isfinite(chosen_CE_weight) or not 0 < chosen_CE_weight <= 1):
        raise ValueError('Exact own actor, complete distinct train-only human pairs and global replica profile required')
    info = [pair_information(data, p) for p in positions]
    denominator = sum(r['chosen_response_EOS_targets'] for r in info)
    parameters = [p for _, p in model.named_selected(('actor',))]
    amp = amp or nullcontext
    records = []
    model.train()
    model.policy.gradient_checkpointing = True
    for row in info[rank::world]:
        values, reference = {}, {}
        fields = ('chosen', 'rejected') if DPO_weight else ('chosen',)
        for field in fields:
            batch = data.batch([row['original_train_position']], field, device)
            inputs = {k: v for k, v in batch.items() if k != 'labels'}
            with amp():
                policy = model(**inputs)['logits']
                if DPO_weight:
                    with torch.no_grad():
                        original = model.base(**inputs)['logits']
            values[field], count = sequence_logp(policy, batch['labels'])
            if count != row[field+'_response_EOS_targets']:
                raise ValueError('Every complete original response and EOS target must be used')
            if DPO_weight:
                reference[field], actual = sequence_logp(original, batch['labels'])
                if actual != count:
                    raise ValueError('Own reference must use the identical response mask')
        margin = ((values['chosen']-values['rejected'])-(reference['chosen']-reference['rejected'])) if DPO_weight else None
        pair = F.softplus(-beta*margin) if DPO_weight else None
        loss = -chosen_CE_weight*values['chosen']/denominator
        if DPO_weight:
            loss = loss+pair/len(positions)
        if not bool(torch.isfinite(loss)):
            raise ValueError('Globally normalized human preference loss must be finite')
        loss.backward(inputs=parameters)
        records.append(dict(**row, policy_chosen_logp=float(values['chosen'].detach()),
            policy_rejected_logp=float(values['rejected'].detach()) if DPO_weight else None,
            reference_chosen_logp=float(reference['chosen']) if DPO_weight else None,
            reference_rejected_logp=float(reference['rejected']) if DPO_weight else None,
            implicit_preference_margin=float(margin.detach()) if DPO_weight else None,
            pair_DPO=float(pair.detach()) if DPO_weight else 0.))
    if not records:
        for parameter in parameters:
            if parameter.grad is not None:
                raise ValueError('An empty tail replica must begin without stale gradients')
            parameter.grad = torch.zeros_like(parameter)
    return records


class ParallelHumanPreferenceStage(DenseHumanPreferenceStage):
    def __init__(self, model, optimizer, data, options, sources, device='cpu', amp=None,
                 *, world=1, rank=0, gradient_group=None, control_group=None):
        if (type(world) is not int or world < 1 or type(rank) is not int or not 0 <= rank < world
                or (world > 1 and (not dist.is_initialized()
                    or dist.get_world_size(gradient_group) != world
                    or dist.get_world_size(control_group) != world))):
            raise ValueError('Exact initialized human preference replicas required')
        super().__init__(model, optimizer, data, options, sources, device, amp)
        self.world, self.rank = world, rank
        self.gradient_group, self.control_group = gradient_group, control_group

    def gather(self, value):
        if self.world == 1:
            return [value]
        values = [None]*self.world
        dist.all_gather_object(values, value, group=self.control_group)
        return values

    def cursor(self, completed):
        return dict(super().cursor(completed), replica_rank=self.rank,
                    world_size=self.world)

    def update(self):
        positions = self.positions(self.completed)
        if signature(self.fixed) != self.fixed_signature:
            raise ValueError('Frozen own base, heads and unused critic must stay exact')
        identity = dict(options=self.options, sources=self.sources, positions=positions,
            pairs=[pair_information(self.data, p) for p in positions])
        if any(row != identity for row in self.gather(identity)):
            raise ValueError('Every replica must agree on complete human source identities and global denominators')
        self.optimizer.zero_grad(set_to_none=True)
        try:
            local = local_gradients(self.model, self.data, positions, rank=self.rank, world=self.world,
                DPO_weight=self.options['DPO_weight'], beta=self.options['DPO_beta'],
                chosen_CE_weight=self.options['chosen_CE_weight'], device=self.device, amp=self.amp)
            if (any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in self.parameters)
                    or any(p.grad is not None for p in self.fixed)):
                raise ValueError('Every actor gradient and no frozen-reference/critic gradient required')
            if self.world > 1:
                flat = torch.cat([p.grad.reshape(-1) for p in self.parameters])
                dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=self.gradient_group)
                offset = 0
                for p in self.parameters:
                    p.grad.copy_(flat[offset:offset+p.numel()].view_as(p))
                    offset += p.numel()
                del flat
            norm = float(torch.nn.utils.clip_grad_norm_(self.parameters, self.options['gradient_clip'], error_if_nonfinite=True))
            rows = [r for group in self.gather(local) for r in group]
            by_position = {r['original_train_position']: r for r in rows}
            if len(rows) != len(positions) or set(by_position) != set(positions):
                raise ValueError('No original human pair may be dropped or duplicated')
            rows = [by_position[p] for p in positions]
            chosen = sum(r['chosen_response_EOS_targets'] for r in rows)
            chosen_sum = math.fsum(r['policy_chosen_logp'] for r in rows)
            pair_mean = math.fsum(r['pair_DPO'] for r in rows)/len(rows)
            record = dict(original_source_positions=positions, original_human_pairs=rows,
                DPO_beta=self.options['DPO_beta'], chosen_CE_weight=self.options['chosen_CE_weight'],
                chosen_response_EOS_targets=chosen,
                rejected_response_EOS_targets=sum(r['rejected_response_EOS_targets'] for r in rows),
                complete_context_tokens=sum(r['complete_context_tokens'] for r in rows),
                pair_mean_DPO=pair_mean, chosen_token_mean_NLL=-chosen_sum/chosen,
                summed_objective=pair_mean-self.options['chosen_CE_weight']*chosen_sum/chosen,
                original_frozen_own_reference_shares_the_one_counted_dense_backbone=True,
                human_ranked_other_model_responses_not_fresh_own_output_judgments=True,
                policy_critic_not_trained_by_this_DPO_objective=True,
                step=self.completed+1, learning_rate=self.options['learning_rate'],
                gradient_norm_before_clip=norm, global_pair_and_chosen_token_denominators=True,
                gradient_reduction='SUM_without_second_world_average',
                per_replica_original_pairs=[len(positions[r::self.world]) for r in range(self.world)],
                **actual_exposures(self.data, positions, self.options))
            validate_metric(self.data, positions, self.options, record, self.completed)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
        except BaseException:
            self.optimizer.zero_grad(set_to_none=True)
            raise
        self.completed += 1
        self.metrics.append(record)
        return record

    def state(self):
        return dict(format=FORMAT, core=super().state(), replicas=self.world, replica_rank=self.rank,
                    private_state_is_not_whole_distributed_checkpoint=True, goal_achieved=False)

    def resume(self, state, *, allow_completed=False):
        if (state.get('format') != FORMAT or state.get('replicas') != self.world
                or state.get('replica_rank') != self.rank
                or state.get('private_state_is_not_whole_distributed_checkpoint') is not True
                or state.get('goal_achieved') is not False):
            raise ValueError('Exact private preference replica state and all-worker recovery required')
        for record in state.get('core', {}).get('core', {}).get('metrics', []):
            positions = record['original_source_positions']
            if (record.get('global_pair_and_chosen_token_denominators') is not True
                    or record.get('gradient_reduction') != 'SUM_without_second_world_average'
                    or record.get('per_replica_original_pairs') != [len(positions[r::self.world]) for r in range(self.world)]):
                raise ValueError('Every actual global denominator and empty-tail replica count must reconstruct')
        return super().resume(state['core'], allow_completed=allow_completed)
