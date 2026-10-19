"""Four actual shared-native PPO replicas with global token normalization.

Every original prompt is visited once. Actual own-sampled trajectories are
gathered through an explicit CPU control group; GAE whitening spans every
actual response token. Local token-weighted gradients SUM across the real
replicas before one common clipped Adam update, including genuine empty ranks.
"""
import copy
import math

import torch
from torch import distributed as dist

from .dense_ppo_cached_v2 import validate_cached
from .dense_ppo_v1 import action_outputs, validate_rollouts
from .native_parallel_ppo_history_v2 import History
from .native_shared_ppo_v1 import NativeSharedPPOStage
from .objectives import normalize_rollout_advantages, ppo_loss
from .parallel_function_repair_stage_v1 import value_digest

FORMAT = 'own-four-rank-current-native-shared-human-reward-PPO-v2'


def parallel_binding(kernel):
    options = copy.deepcopy(kernel.options)
    options['steps'] = math.ceil(len(kernel.data)/(4*kernel.options['prompts_per_update']))
    binding = {**copy.deepcopy(kernel.binding), 'parallel_format': FORMAT, 'world_size': 4,
        'planned_updates': options['steps'], 'options': copy.deepcopy(options),
        'local_kernel_binding_sha256': kernel.binding_sha256,
        'global_normalization': 'all_actual_response_tokens_across_all_four_ranks',
        'gradient_reduction': 'SUM_of_local_losses_divided_by_global_action_tokens'}
    return options, binding


class Collective:
    def __init__(self, device, *, control_group, gradient_group=None):
        self.device = torch.device(device); self.control_group = control_group; self.gradient_group = gradient_group
        if (not dist.is_initialized() or dist.get_backend(control_group) != 'gloo'
                or dist.get_world_size(control_group) != 4 or dist.get_world_size(gradient_group) != 4
                or dist.get_rank(control_group) != dist.get_rank(gradient_group)
                or dist.get_backend(gradient_group) != ('nccl' if self.device.type == 'cuda' else 'gloo')):
            raise ValueError('The actual four replicas require explicit Gloo control and matching CPU/GPU gradient groups')
        self.rank, self.world = dist.get_rank(control_group), 4

    def gather(self, value):
        rows = [None]*4; dist.all_gather_object(rows, value, group=self.control_group); return rows

    def agree(self, error, common=None):
        rows = self.gather((error, common))
        if any(row[0] is not None for row in rows):
            raise ValueError('Coordinated shared-native PPO rejection: '+repr([r[0] for r in rows]))
        if any(row[1] != rows[0][1] for row in rows):
            raise ValueError('The exact shared PPO source/cursor/common state differs across actual ranks')

    def sum_gradients(self, parameters):
        flat = torch.cat([p.grad.reshape(-1) for p in parameters])
        dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=self.gradient_group)
        offset = 0
        for p in parameters:
            p.grad.copy_(flat[offset:offset+p.numel()].reshape_as(p)); offset += p.numel()


class ParallelNativePPO:
    def __init__(self, kernel, collective):
        if type(kernel) is not NativeSharedPPOStage or type(collective) is not Collective:
            raise ValueError('An exact shared-native PPO local kernel and actual four-rank collective are required')
        self.kernel, self.collective = kernel, collective; self.rank, self.world = collective.rank, 4
        if kernel.completed != 0 or kernel.history is not None or kernel.optimizer.state:
            raise ValueError('Each parallel replica must start at its fresh exact shared-native boundary')
        self.options, self.binding = parallel_binding(kernel)
        self.binding_sha256 = value_digest(self.binding)
        self.history_binding = {**self.binding, 'physical_rank': self.rank}
        self.initial_global_record_sha256 = value_digest(dict(format=FORMAT, binding_sha256=self.binding_sha256))
        self.last_global_record_sha256 = self.initial_global_record_sha256
        self.completed = 0; self.history = None; self.previous_prefix = None; self.failed = False
        collective.agree(None, self.binding_sha256)

    def check(self, *, full=False):
        self.kernel.check(full=full)
        if (self.failed or value_digest(self.binding) != self.binding_sha256
                or self.options != self.binding['options'] or self.collective.rank != self.rank
                or self.kernel.completed != 0 or self.kernel.history is not None):
            raise ValueError('The fixed actual four-rank source/horizon and unchanged local collection kernel must remain exact')

    def positions(self, step, rank=None):
        if type(step) is not int or not 0 <= step < self.options['steps'] or rank is not None and rank not in range(4):
            raise ValueError('An actual remaining original four-rank PPO source position is required')
        width = 4*self.options['prompts_per_update']; start = step*width
        rows = [self.kernel.order.at(i) for i in range(start, min(start+width, len(self.kernel.data)))]
        if rank is None: return rows
        count = self.options['prompts_per_update']; return rows[rank*count:(rank+1)*count]

    def cursor(self, completed):
        if type(completed) is not int or not 0 <= completed <= self.options['steps']:
            raise ValueError('The actual unrepeated four-rank PPO cursor is required')
        draws = min(completed*4*self.options['prompts_per_update'], len(self.kernel.data))
        return dict(completed_updates=completed, original_prompt_population=len(self.kernel.data),
            actual_prompt_draws=draws, global_prompts_per_update=4*self.options['prompts_per_update'],
            complete_prompt_passes=draws//len(self.kernel.data), next_pass_prompts=draws%len(self.kernel.data),
            seed=self.options['seed'], world_size=4, repeated_rounding_prompts=0)

    def start_history(self, directory, budgets, *, checked=lambda: None):
        self.check(full=True)
        if self.history is not None: raise ValueError('Never reopen or replace the actual parallel PPO trajectory writer')
        self.history = History(directory, binding=self.history_binding, budgets=budgets,
            previous=self.previous_prefix, guard=lambda: (checked(), self.check()))

    @torch.no_grad()
    def collect(self):
        local = None; error = None
        try:
            positions = self.positions(self.completed, self.rank)
            local = self.kernel.collect(positions) if positions else []
            for row in local: row['sampling_rank'] = self.rank
        except Exception as exc: error = type(exc).__name__+': '+str(exc)
        gathered = self.collective.gather((error, local))
        if any(row[0] is not None for row in gathered):
            raise ValueError('An actual four-rank sampled collection failed: '+repr([r[0] for r in gathered]))
        rows = [row for _, rank_rows in gathered for row in rank_rows]
        for row in rows: row['advantages'] = torch.tensor(row['raw_advantages'], dtype=torch.float32)
        normalize_rollout_advantages(rows)
        for row in rows: row['advantages'] = row['advantages'].tolist()
        self.validate_rows(self.completed, rows)
        return rows

    def validate_rows(self, step, rows):
        kernel = self.kernel; positions = self.positions(step)
        validate_rollouts(kernel.data, positions, self.options, rows, kernel.reward.provenance,
            kernel.model.config.vocab_size, kernel.model.config.max_text_length)
        validate_cached(rows)
        expected = [r for r in range(4) for _ in self.positions(step, r)]
        if [row.get('sampling_rank') for row in rows] != expected:
            raise ValueError('Every actual sampled trajectory must keep its original physical rank and whole source')

    def validate_record(self, record):
        step = record.get('step')
        if type(step) is not int or not 1 <= step <= self.options['steps']:
            raise ValueError('Every exact complete four-rank PPO update must remain in its original horizon')
        rows = record.get('rollouts', []); self.validate_rows(step-1, rows)
        wanted = dict(format=FORMAT, world_size=4, source_positions=self.positions(step-1),
            actual_action_tokens=sum(len(r['response_ids']) for r in rows),
            actual_complete_prompt_tokens=sum(len(r['prompt_ids']) for r in rows),
            rank_source_positions=[self.positions(step-1, rank) for rank in range(4)],
            rank_action_tokens=[sum(len(r['response_ids']) for r in rows if r['sampling_rank'] == rank) for rank in range(4)],
            policy_unique_parameters=self.kernel.model.counts()['actual_unique'], shared_native_reward_counts=self.kernel.reward.counts(),
            additional_reward_encoder_parameters=0, learning_rate=self.options['learning_rate'],
            advantages_normalized_over_all_actual_response_tokens=True, gradients_summed_after_global_token_weighting=True,
            own_sampled_responses_not_new_human_judgments=True)
        if any(record.get(k) != v for k, v in wanted.items()):
            raise ValueError('Every actual global token denominator, empty rank and shared-native count must reconcile')
        losses = record.get('rank_token_weighted_losses')
        if (not isinstance(losses, list) or len(losses) != 4 or any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in losses)
                or record.get('token_weighted_PPO_loss') != math.fsum(losses)
                or not isinstance(record.get('gradient_norm_before_clip'), (int, float))
                or not math.isfinite(record['gradient_norm_before_clip']) or record['gradient_norm_before_clip'] < 0
                or record.get('global_record_sha256') != value_digest({k:v for k,v in record.items() if k != 'global_record_sha256'})):
            raise ValueError('Every actual four-rank loss, gradient and common mathematical record hash must remain finite and exact')

    def update(self):
        error = None; rows = None; adam_returned = False; kernel = self.kernel; c = self.collective
        try:
            try:
                self.check()
                if self.history is None or self.history.starting+self.history.writer.records != self.completed:
                    raise ValueError('Every real rank requires its funded full trajectory writer at the exact cursor')
                if self.completed >= self.options['steps']: raise ValueError('The whole original source epoch already completed')
                self.history.reserve_next()
            except Exception as exc: error = type(exc).__name__+': '+str(exc)
            c.agree(error, (self.completed, self.last_global_record_sha256))
            rows = self.collect(); total = sum(len(r['response_ids']) for r in rows)
            kernel.optimizer.zero_grad(set_to_none=True); kernel.model.train(); kernel.model.policy.gradient_checkpointing = True
            losses = []; error = None
            try:
                with torch.autocast(device_type=kernel.device.type, enabled=False):
                    for row in rows:
                        if row['sampling_rank'] != self.rank: continue
                        ids = torch.tensor([row['prompt_ids']+row['response_ids']], device=kernel.device)
                        logps, values = action_outputs(kernel.model, ids, len(row['prompt_ids']), kernel.amp)
                        old, adv, old_value, returns = [torch.tensor(row[k], device=kernel.device)
                            for k in ['old_logps', 'advantages', 'old_values', 'returns']]
                        loss = ppo_loss(logps, old, adv, values, old_value, returns,
                            self.options['clip'], self.options['value_coefficient'])*len(logps)/total
                        if not bool(torch.isfinite(loss)): raise ValueError('Every actual local weighted PPO loss must be finite')
                        loss.backward(inputs=kernel.parameters); losses.append(float(loss.detach()))
                if not self.positions(self.completed, self.rank):
                    for p in kernel.parameters: p.grad = torch.zeros_like(p)
                if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in kernel.parameters):
                    raise ValueError('Every actual actor/critic contribution, including genuine zero ranks, must be finite')
                self.check()
            except Exception as exc: error = type(exc).__name__+': '+str(exc)
            c.agree(error); c.sum_gradients(kernel.parameters)
            error = None; norm = None
            try:
                norm = float(torch.nn.utils.clip_grad_norm_(kernel.parameters, self.options['gradient_clip'], error_if_nonfinite=True))
            except Exception as exc: error = type(exc).__name__+': '+str(exc)
            gathered = c.gather((error, norm, math.fsum(losses)))
            if any(r[0] is not None for r in gathered) or any(r[1] != gathered[0][1] for r in gathered):
                raise ValueError('Every actual shared global clipped gradient must agree across all four replicas')
            record = dict(format=FORMAT, step=self.completed+1, world_size=4, source_positions=self.positions(self.completed),
                rollouts=rows, actual_action_tokens=total, actual_complete_prompt_tokens=sum(len(r['prompt_ids']) for r in rows),
                rank_source_positions=[self.positions(self.completed, r) for r in range(4)],
                rank_action_tokens=[sum(len(r['response_ids']) for r in rows if r['sampling_rank'] == rnk) for rnk in range(4)],
                rank_token_weighted_losses=[r[2] for r in gathered], token_weighted_PPO_loss=math.fsum(r[2] for r in gathered),
                gradient_norm_before_clip=norm, learning_rate=self.options['learning_rate'],
                policy_unique_parameters=kernel.model.counts()['actual_unique'], shared_native_reward_counts=kernel.reward.counts(),
                additional_reward_encoder_parameters=0, advantages_normalized_over_all_actual_response_tokens=True,
                gradients_summed_after_global_token_weighting=True, own_sampled_responses_not_new_human_judgments=True,
                previous_global_record_sha256=self.last_global_record_sha256)
            record['global_record_sha256'] = value_digest(record); error = None
            try: self.validate_record(record); self.history.append(record)
            except Exception as exc: error = type(exc).__name__+': '+str(exc)
            c.agree(error, record['global_record_sha256']); error = None
            try:
                kernel.optimizer.step(); adam_returned = True
                kernel.optimizer.zero_grad(set_to_none=True); self.check()
            except Exception as exc: error = type(exc).__name__+': '+str(exc)
            c.agree(error); error = None
            try: self.history.acknowledge(record['step'])
            except Exception as exc: error = type(exc).__name__+': '+str(exc)
            c.agree(error)
            self.completed += 1; self.last_global_record_sha256 = record['global_record_sha256']; return record
        except BaseException as error:
            self.failed = True; kernel.optimizer.zero_grad(set_to_none=True)
            if self.history is not None and self.history.writer.failure_slot is not None:
                try:
                    self.history.failure(dict(rank=self.rank, world_size=4, attempted_step=self.completed+1,
                        source_positions=self.positions(self.completed), rollouts=rows, exception=repr(error),
                        optimizer_step_returned=adam_returned, completed_global_policy_update_not_certified=True))
                except BaseException as secondary: error.add_note('Independent exact failure preservation failed: '+repr(secondary))
            raise
