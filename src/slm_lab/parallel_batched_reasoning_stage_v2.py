"""Four-replica fresh reasoning groups with complete-source shared recovery.

Each active rank owns one original source and batches its private attempts.
Gradients are averaged over actual source groups before one global SUM/Adam
commit. Every original answer/EOS remains in the unconditional CE anchor.
This stage owns no resources and cannot admit itself to GPU production.
"""
import copy
import hashlib
import math
import struct

import torch
from torch import distributed as dist

from .batched_reasoning_group_gradients_v2 import group_gradients, FORMAT as KERNEL
from .codenet_checkpoint_v1 import capture_rng, restore_rng
from .code_residual_checkpoint_v1 import _cpu, _validate_adam, _validate_rng
from .execution_group_controlled_v2 import advantages
from .frozen_own_actor_reference_v1 import FrozenOwnActorReference, signature
from .own_dense_actor_checkpoint_v1 import selected
from .parallel_function_repair_stage_v1 import value_digest, manifest
from .public_reasoning_train_data_v1 import PublicReasoningTrainData
from .whole_human_sft_evidence_v1 import tensor_digest

FORMAT = 'own-parallel-batched-public-reasoning-shared-Adam-private-RNG-v2'
OPTIONS = {'learning_rate', 'gradient_clip', 'group_size', 'KL_weight', 'anchor_weight',
    'cache_absolute_bound', 'maximum_cache_bytes', 'gradient_bucket_values', 'seed', 'epochs', 'CPU_mechanics_horizon'}


def private_seeds(seed, step, position, count):
    return [int.from_bytes(hashlib.sha256(struct.pack('<QQQQ', seed, step, position, i)).digest()[:8], 'little') % 2**63
        for i in range(count)]


def sum_gradients(parameters, device, world, bucket_values, group):
    """Explicit actual gradient group: NCCL/CUDA or Gloo/CPU, never control."""
    if world == 1:
        return sum(p.grad is not None for p in parameters)
    present = torch.tensor([int(p.grad is not None) for p in parameters], dtype=torch.int32, device=device)
    dist.all_reduce(present, op=dist.ReduceOp.MAX, group=group)
    active, bucket, values = present.cpu().tolist(), [], 0
    def flush():
        flat = torch.cat([p.grad.reshape(-1) for p in bucket])
        dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=group)
        offset = 0
        for p in bucket:
            p.grad.copy_(flat[offset:offset+p.numel()].view_as(p))
            offset += p.numel()
        bucket.clear()
    for p, on in zip(parameters, active):
        if not on:
            continue
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        if bucket and values + p.numel() > bucket_values:
            flush()
            values = 0
        bucket.append(p)
        values += p.numel()
        if values >= bucket_values:
            flush()
            values = 0
    if bucket:
        flush()
    return sum(active)


class ParallelBatchedReasoningStage:
    def __init__(self, model, optimizer, data, entries, profile, options, sources, *, mode,
                 rank=0, world=1, control_group=None, gradient_group=None):
        if (type(world) is not int or not 1 <= world <= 4 or type(rank) is not int or not 0 <= rank < world
                or world == 1 and dist.is_initialized()
                or world > 1 and (not dist.is_initialized() or dist.get_world_size() != world or dist.get_world_size(control_group) != world
                    or dist.get_rank(control_group) != rank or dist.get_world_size(gradient_group) != world
                    or dist.get_rank(gradient_group) != rank or dist.get_backend(control_group) != 'gloo')):
            raise ValueError('Exact actual replica ranks and separate Gloo control group required')
        self.rank, self.world, self.control_group, self.gradient_group = rank, world, control_group, gradient_group
        self.failure_observations = []
        error = initial = None
        try:
            if (type(data) is not PublicReasoningTrainData or data.role != 'train' or data.purpose != 'optimization'
                    or profile not in ['anchor_CE', 'anchor_RL'] or mode not in ['mechanics', 'capacity', 'train']
                    or not isinstance(entries, list) or not entries or optimizer.state or set(options) != OPTIONS
                    or any(not isinstance(e, list) or not 1 <= len(e) <= world or len(e) != len(set(e)) for e in entries)
                    or not sources or any(not isinstance(k, str) or not isinstance(v, str) or len(v) != 64
                        or any(c not in '0123456789abcdef' for c in v) for k, v in sources.items())):
                raise ValueError('Exact admitted full train data, complete source groups, fresh Adam, mode and bindings required')
            if (any(type(options[k]) is not int or options[k] < 1 for k in ['seed', 'epochs', 'group_size', 'maximum_cache_bytes', 'gradient_bucket_values'])
                    or not 2 <= options['group_size'] <= 8 or options['seed'] >= 2**64
                    or any(type(options[k]) not in [int, float] or not math.isfinite(options[k]) or options[k] <= 0
                        for k in ['learning_rate', 'gradient_clip', 'anchor_weight', 'cache_absolute_bound'])
                    or options['cache_absolute_bound'] > .001 or not math.isfinite(options['KL_weight']) or options['KL_weight'] < 0):
                raise ValueError('Complete positive finite unchanged precision/objective/source-group settings required')
            self.model, self.optimizer, self.data = model, optimizer, data
            self.entries, self.options, self.sources = copy.deepcopy(entries), copy.deepcopy(options), dict(sources)
            self.profile, self.mode, self.precision = profile, mode, 'fp32'
            self.device = next(model.parameters()).device
            if world > 1 and dist.get_backend(gradient_group) != ('nccl' if self.device.type == 'cuda' else 'gloo'):
                raise ValueError('Actual device gradients require their matching NCCL/CUDA or Gloo/CPU group')
            self.named = selected(model, optimizer, ('actor',))
            self.parameters = [p for _, p in self.named]
            if (any(p.dtype != torch.float32 or p.device != self.device or p.grad is not None for p in model.parameters())
                    or optimizer.param_groups[0]['lr'] != options['learning_rate']):
                raise ValueError('One clear-gradient actual FP32 own replica and matching ordinary Adam learning rate required')
            if mode == 'mechanics':
                if self.device.type != 'cpu' or torch.cuda.is_initialized():
                    raise ValueError('Own-small mechanics must remain explicitly CPU-only')
            elif (world != 4 or self.device != torch.device('cuda', rank) or not torch.are_deterministic_algorithms_enabled()
                    or torch.backends.cuda.matmul.allow_tf32 or options['CPU_mechanics_horizon'] is not None
                    or model.counts()['actual_unique'] != 999706442 or model.base.counts()['total_unique'] != 984399690):
                raise ValueError('Full capacity/training requires four actual existing full-size deterministic FP32 GPUreplicas')
            if mode == 'train' and entries != data.complete_schedule(epochs=options['epochs'], width=world, seed=options['seed']):
                raise ValueError('Every original source, whole native horizon and declared complete epoch must be retained')
            data.budget(entries)
            if (options['CPU_mechanics_horizon'] is not None and (mode != 'mechanics'
                    or type(options['CPU_mechanics_horizon']) is not int
                    or not 1 <= options['CPU_mechanics_horizon'] <= min(data.information(p)['full_training_action_budget'] for e in entries for p in e))):
                raise ValueError('Only an explicit valid own-small CPU mechanics horizon may shorten source budgets')
            chosen = {id(p) for p in self.parameters}
            self.fixed = [p for p in model.parameters() if id(p) not in chosen]
            self.fixed_signature = signature(self.fixed)
            self.base_digest = tensor_digest(model.base.named_parameters())
            self.reference = FrozenOwnActorReference(model, sources=sources)
            self.reference_binding = self.reference.binding()
            self.schedule_identity = data.identity(entries)
            self.optimizer_settings = {k: copy.deepcopy(v) for k, v in optimizer.param_groups[0].items() if k != 'params'}
            for k in ['lr', 'eps', 'weight_decay']:
                if not math.isfinite(self.optimizer_settings[k]) or self.optimizer_settings[k] < 0 or k != 'weight_decay' and self.optimizer_settings[k] == 0:
                    raise ValueError('Every actual Adam setting must be finite and valid')
            if any(not 0 <= b < 1 for b in self.optimizer_settings['betas']):
                raise ValueError('Both actual Adam betas must be in[0,1)')
            self.expected = [[data.information(p) for p in row] for row in entries]
            self.initial_binding = copy.deepcopy(self.binding())
            self.options_digest = value_digest(self.initial_binding)
            self.completed, self._metrics, self.failed = 0, [], False
            initial = dict(binding=self.initial_binding, base=self.base_digest,
                actor=tensor_digest(model.parts.named_parameters()), actual_counts=model.counts(), device_type=self.device.type)
        except Exception as exc:
            error = repr(exc)
        self.coordinated(error, 'construction')
        if any(v != initial for v in self.gather(initial)):
            raise ValueError('All replicas must share the same initializer, reference, optimizer and complete source schedule')

    def gather(self, value):
        if self.world == 1:
            return [value]
        result = [None] * self.world
        dist.all_gather_object(result, value, group=self.control_group)
        return result

    def coordinated(self, error, phase):
        errors = self.gather(error)
        if any(e is not None for e in errors):
            self.failure_observations.append(copy.deepcopy(dict(phase=phase, actual_rank_errors=errors)))
            raise ValueError('Coordinated batched reasoning ' + phase + ' rejection: ' + repr(errors))

    @property
    def metrics(self):
        return copy.deepcopy(self._metrics)

    def binding(self):
        return dict(format=FORMAT, profile=self.profile, mode=self.mode, precision=self.precision, world=self.world,
            options=self.options, sources=self.sources, schedule=self.schedule_identity,
            complete_entries_sha256=value_digest(self.entries), reference=self.reference_binding,
            optimizer_settings=self.optimizer_settings,
            normalization='equal_source_mean_of_complete_local_action_and_gold_token_means')

    def intact(self):
        current = selected(self.model, self.optimizer, ('actor',))
        if (type(self.data) is not PublicReasoningTrainData or self.data.role != 'train' or self.data.purpose != 'optimization'
                or type(self.completed) is not int or not 0 <= self.completed <= len(self.entries) or len(self._metrics) != self.completed
                or [(n, id(p)) for n, p in current] != [(n, id(p)) for n, p in self.named]
                or self.binding() != self.initial_binding or signature(self.fixed) != self.fixed_signature
                or {k: v for k, v in self.optimizer.param_groups[0].items() if k != 'params'} != self.optimizer_settings
                or any(p.grad is not None for p in self.model.parameters())):
            raise ValueError('Immutable source/settings/reference/fixed values and a clear-gradient boundary required')
        self.reference.assert_bound(self.model)
        self.data.guard.check()
        if self.completed == 0:
            if self.optimizer.state:
                raise ValueError('A zero cursor requires fresh Adam state')
        elif (len(self.optimizer.state) != len(self.parameters)
                or any(float(self.optimizer.state[p]['step']) != self.completed for p in self.parameters)):
            raise ValueError('All actual selected Adam clocks must equal the exact complete global source prefix')

    def validate_kernel(self, row, position, step):
        info = self.data.information(position)
        ids, _ = self.data.sequence(position)
        prefix = ids[:info['original_prompt_positions']]
        horizon = self.options['CPU_mechanics_horizon'] or info['full_training_action_budget']
        seeds = private_seeds(self.options['seed'], step, position, self.options['group_size'])
        if (row.get('format') != KERNEL or row.get('profile') != self.profile or row.get('complete_original_source') != info
                or row.get('original_complete_prompt_ids') != prefix or row.get('actual_policy_private_generator_seeds') != seeds
                or row.get('actual_group_size') != len(seeds) or row.get('actual_delivered_action_horizon') != horizon
                or row.get('full_original_source_action_budget') != info['full_training_action_budget']
                or row.get('CPU_mechanics_horizon') != self.options['CPU_mechanics_horizon']
                or row.get('KL_weight') != self.options['KL_weight'] or row.get('anchor_weight') != self.options['anchor_weight']
                or row.get('cached_full_probability_bound') != self.options['cache_absolute_bound']
                or row.get('reference_binding') != self.reference_binding
                or row.get('resident_training_inventory') != self.reference.resident_inventory(self.model)):
            raise ValueError('Exact original source, fresh private seeds, complete native budgets and own-reference objective required')
        items = row['trajectories']
        sampling = row['complete_batched_sampling_observation']
        if len(items) != len(seeds) or len(sampling['trajectories']) != len(seeds):
            raise ValueError('Every actual private attempt must remain present')
        rewards, actions = [], 0
        for i, (item, sampled) in enumerate(zip(items, sampling['trajectories'])):
            emitted = item['actual_generated_token_ids']
            verdict = self.data.score(position, emitted, horizon=horizon)
            if (item['sample_position'] != i or item['private_seed'] != seeds[i] or item['original_prompt_ids'] != prefix
                    or item['actual_strict_source_verifier'] != verdict or any(item.get(k) != v for k, v in sampled.items())
                    or item['actual_native_action_count'] != len(emitted) or item['actual_stopped_on_EOS'] != (emitted[-1] == 2)):
                raise ValueError('Every actual sample/behavior/verifier/native delivery must reconstruct')
            cached = item['actual_cached_behavior_logps']
            for key in ['actual_full_forward_behavior_logps', 'actual_gradient_forward_behavior_logps']:
                full = item[key]
                if (len(cached) != len(emitted) or len(full) != len(emitted)
                        or any(type(x) not in [float, int] or not math.isfinite(x) or x > 0 for x in cached + full)
                        or max(abs(a - b) for a, b in zip(cached, full)) > self.options['cache_absolute_bound']):
                    raise ValueError('Every actual cached/full/gradient log probability must be finite and inside the unchanged bound')
            rewards.append(verdict['reward'])
            actions += len(emitted)
        lengths = [len(t['actual_generated_token_ids']) for t in items]
        if (row['actual_strict_rewards'] != rewards or row['actual_group_relative_advantages'] != advantages(rewards)
                or row['actual_sampled_action_tokens'] != actions or not sampling['behavior_rescore_passed']
                or sampling['nonfinite_full_forward_behavior_observed']
                or sampling['actual_prefill_batch_sizes'] != [len(seeds)]
                or sampling['actual_decode_batch_sizes'] != [sum(n > k for n in lengths) for k in range(1, max(lengths))]):
            raise ValueError('Every real group advantage, action denominator and shrinking native batch must reconcile')
        anchor = row['unconditional_original_answer_CE_anchor']
        if (anchor['complete_sources'] != [info] or anchor['response_EOS_targets'] != info['response_EOS_targets']
                or not anchor['all_whole_question_answer_and_true_EOS_masks_preserved']
                or not math.isfinite(anchor['original_CE_sum']) or anchor['original_CE_sum'] < 0
                or anchor['original_token_mean_NLL'] != anchor['original_CE_sum'] / info['response_EOS_targets']):
            raise ValueError('Every complete original CE/EOS anchor is unconditional on reward')

    def validate_records(self, rows):
        if not isinstance(rows, list) or not 1 <= len(rows) <= len(self.entries):
            raise ValueError('The complete nonempty original source-prefix records are required')
        for i, row in enumerate(rows):
            self.validate_record(row, i)

    def validate_record(self, row, index):
        positions = self.entries[index]
        if (row['step'] != index + 1 or row['global_original_sources'] != self.expected[index]
                or row['actual_original_source_positions'] != positions or row['actual_world_size'] != self.world
                or row['normalization'] != self.initial_binding['normalization']
                or row['original_response_EOS_targets'] != sum(i['response_EOS_targets'] for i in self.expected[index])
                or row['complete_original_context_tokens'] != sum(i['complete_context_tokens'] for i in self.expected[index])
                or row['learning_rate'] != self.options['learning_rate']
                or not math.isfinite(row['gradient_norm_before_clip']) or row['gradient_norm_before_clip'] < 0
                or not row['private_global_RNG_unchanged_by_update'] or not row['one_global_gradient_SUM_and_Adam_commit']
                or len(row['actual_rank_observations']) != self.world):
            raise ValueError('Every complete global source group, original target and ordinary optimizer update must reconcile')
        for rank, actual in enumerate(row['actual_rank_observations']):
            local = positions[rank:rank+1]
            if actual['rank'] != rank or actual['actual_original_positions'] != local:
                raise ValueError('Each actual rank must retain its exact original source or genuine empty tail')
            if local:
                self.validate_kernel(actual['kernel'], local[0], index + 1)
            elif actual['kernel'] is not None:
                raise ValueError('Empty tail ranks cannot fabricate a forward, source or sampled attempt')

    def update(self):
        error = None
        try:
            if self.failed or self.completed >= len(self.entries):
                raise ValueError('One healthy remaining whole-source update is required')
            self.intact()
            if self.expected[self.completed] != [self.data.information(p) for p in self.entries[self.completed]]:
                raise ValueError('The actual original source identity changed before sampling')
        except Exception as exc:
            error = repr(exc)
        self.coordinated(error, 'update preflight')
        positions = self.entries[self.completed]
        before_rng = value_digest(capture_rng(self.device))
        kernel = None
        error = None
        try:
            if self.rank < len(positions):
                position = positions[self.rank]
                kernel = group_gradients(self.model, self.data, position, self.reference,
                    profile=self.profile, seeds=private_seeds(self.options['seed'], self.completed + 1, position, self.options['group_size']),
                    purpose='CPU-mechanics' if self.mode == 'mechanics' else 'GPU-runtime',
                    **{k: self.options[k] for k in ['maximum_cache_bytes', 'KL_weight', 'anchor_weight', 'cache_absolute_bound', 'CPU_mechanics_horizon']})
                self.validate_kernel(kernel, position, self.completed + 1)
                for p in self.parameters:
                    p.grad.div_(len(positions))
            else:
                self.model.train()
                self.model.policy.gradient_checkpointing = True
            if value_digest(capture_rng(self.device)) != before_rng:
                raise ValueError('Private seeded attempts must not consume the original global RNG')
        except Exception as exc:
            error = dict(exception_type=type(exc).__name__, message=str(exc),
                complete_actual_failure_observation=copy.deepcopy(getattr(exc, 'report', None)))
        try:
            self.coordinated(error, 'whole source sampling/verifier/gradient before reduction')
            observed = self.gather(dict(rank=self.rank, actual_original_positions=positions[self.rank:self.rank+1], kernel=kernel))
            error = None
            try:
                count = sum_gradients(self.parameters, self.device, self.world, self.options['gradient_bucket_values'], self.gradient_group)
                if count != len(self.parameters) or any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in self.parameters):
                    raise ValueError('All selected gradients must receive exactly one globally normalized source-group SUM')
                gradient_sha = tensor_digest((name, p.grad) for name, p in self.named)
                norm = float(torch.nn.utils.clip_grad_norm_(self.parameters, self.options['gradient_clip'], error_if_nonfinite=True))
            except Exception as exc:
                error = repr(exc)
            self.coordinated(error, 'summed gradient validation before Adam')
            row = dict(step=self.completed + 1, global_original_sources=self.expected[self.completed],
                actual_original_source_positions=list(positions), actual_world_size=self.world,
                normalization=self.initial_binding['normalization'], actual_rank_observations=observed,
                original_response_EOS_targets=sum(i['response_EOS_targets'] for i in self.expected[self.completed]),
                complete_original_context_tokens=sum(i['complete_context_tokens'] for i in self.expected[self.completed]),
                actual_unclipped_global_actor_gradient_sha256=gradient_sha, gradient_norm_before_clip=norm,
                learning_rate=self.options['learning_rate'], private_global_RNG_unchanged_by_update=True,
                one_global_gradient_SUM_and_Adam_commit=True, genuine_empty_ranks_have_no_source_forward=True)
            error = None
            try:
                self.validate_record(row, self.completed)
            except Exception as exc:
                error = repr(exc)
            self.coordinated(error, 'global source record before Adam')
            if any(v != value_digest(row) for v in self.gather(value_digest(row))):
                raise ValueError('Actual shared source records and all summed gradients must agree before Adam')
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            error = None
            try:
                if (signature(self.fixed) != self.fixed_signature or any(not bool(torch.isfinite(p).all()) for p in self.parameters)
                        or any(float(self.optimizer.state[p]['step']) != self.completed + 1 for p in self.parameters)):
                    raise ValueError('Every selected weight/clock and unchanged fixed value must preserve the completed update')
                self.reference.assert_bound(self.model)
            except Exception as exc:
                error = repr(exc)
            self.coordinated(error, 'post-Adam completed boundary')
        except BaseException:
            self.failed = True
            self.optimizer.zero_grad(set_to_none=True)
            raise
        self.completed += 1
        self._metrics.append(copy.deepcopy(row))
        return copy.deepcopy(row)

    def cursor(self, completed, rank=None):
        value = dict(completed=completed, profile=self.profile, mode=self.mode, precision=self.precision, world_size=self.world,
            schedule_identity_sha256=self.schedule_identity, all_consumed_global_source_batches=self.entries[:completed],
            original_source_role='train', private_seed_derivation='sha256(root_seed,global_step,source_position,sample_index)')
        if rank is not None:
            value.update(rank=rank, all_consumed_local_source_batches=[e[rank:rank+1] for e in self.entries[:completed]])
        return value

    def snapshot(self):
        error = None
        try:
            self.intact()
            if self.failed or self.completed < 1:
                raise ValueError('Only a healthy complete optimizer boundary may publish a shared snapshot')
            self.validate_records(self._metrics)
            if tensor_digest(self.model.base.named_parameters()) != self.base_digest:
                raise ValueError('Every original frozen neural value must remain unchanged')
            adam = _cpu(self.optimizer.state_dict())
            _validate_adam(adam, self.parameters, self.completed)
            rng = capture_rng(self.device)
            _validate_rng(rng, self.device)
            local = dict(rank=self.rank, local_source_cursor=self.cursor(self.completed, self.rank), rng=rng,
                replica=dict(frozen_base_sha256=self.base_digest, model_parts_sha256=tensor_digest(self.model.parts.named_parameters()),
                    optimizer_sha256=value_digest(adam), global_metrics_sha256=value_digest(self._metrics)))
        except Exception as exc:
            error = repr(exc)
        self.coordinated(error, 'shared snapshot preflight')
        workers = self.gather(local)
        if any(w['replica'] != workers[0]['replica'] for w in workers):
            raise ValueError('All actual model/Adam/source-prefix replicas must be identical before shared storage')
        state = receipt = None
        error = None
        if self.rank == 0:
            try:
                payload = self.model.export(self.sources)
                self.model.validate_export(payload, expected_sources=self.sources)
                state = dict(format=FORMAT, world_size=self.world, profile=self.profile, mode=self.mode, precision=self.precision,
                    completed=self.completed, planned=len(self.entries), schedule_identity_sha256=self.schedule_identity,
                    options_sha256=self.options_digest, groups=['actor'], optimized_names=[n for n, _ in self.named],
                    model=payload, optimizer=adam, global_cursor=self.cursor(self.completed), metrics=_cpu(self._metrics),
                    private_workers=_cpu(workers), reference_binding=copy.deepcopy(self.reference_binding),
                    one_shared_policy_and_Adam_copy_verified_across_all_replicas=True,
                    all_private_RNGs_and_source_cursors_retained=True, candidate_promoted=False, goal_achieved=False)
                receipt = manifest(state)
            except Exception as exc:
                error = repr(exc)
        self.coordinated(error, 'root shared payload serialization')
        receipts = self.gather(receipt)
        return state, receipts[0]

    def resume(self, state, *, expected_manifest, allow_completed=False):
        error = None
        try:
            current = selected(self.model, self.optimizer, ('actor',))
            self.data.guard.check()
            if ([(n, id(p)) for n, p in current] != [(n, id(p)) for n, p in self.named]
                    or type(self.data) is not PublicReasoningTrainData or self.data.role != 'train' or self.data.purpose != 'optimization'
                    or self.binding() != self.initial_binding or signature(self.fixed) != self.fixed_signature
                    or manifest(state) != expected_manifest or not expected_manifest
                    or state['format'] != FORMAT or state['world_size'] != self.world or state['profile'] != self.profile
                    or state['mode'] != self.mode or state['precision'] != self.precision or state['planned'] != len(self.entries)
                    or type(state['completed']) is not int or not 0 < state['completed'] <= len(self.entries)
                    or state['completed'] == len(self.entries) and not allow_completed
                    or state['schedule_identity_sha256'] != self.schedule_identity or state['options_sha256'] != self.options_digest
                    or state['global_cursor'] != self.cursor(state['completed']) or len(state['metrics']) != state['completed']
                    or state['groups'] != ['actor'] or state['optimized_names'] != [n for n, _ in self.named]
                    or state['reference_binding'] != self.reference_binding or state['candidate_promoted'] is not False
                    or state['goal_achieved'] is not False or len(state['private_workers']) != self.world
                    or state['one_shared_policy_and_Adam_copy_verified_across_all_replicas'] is not True
                    or state['all_private_RNGs_and_source_cursors_retained'] is not True):
                raise ValueError('Exact independently retained shared manifest/reference/source/optimizer prefix required')
            self.reference.assert_bound(self.model)
            self.validate_records(state['metrics'])
            self.model.validate_export(state['model'], expected_sources=self.sources)
            _validate_adam(state['optimizer'], self.parameters, state['completed'])
            if state['optimizer']['param_groups'] != self.optimizer.state_dict()['param_groups']:
                raise ValueError('Every actual ordinary Adam option/order must remain unchanged')
            selected_names = {n for n, _ in self.named}
            if any(not torch.equal(t.detach().cpu(), state['model']['parts'][n]) for n, t in self.model.parts.state_dict().items() if n not in selected_names):
                raise ValueError('Every unused critic value must remain unchanged')
            expected_replica = dict(frozen_base_sha256=self.base_digest,
                model_parts_sha256=tensor_digest(state['model']['parts'].items()),
                optimizer_sha256=value_digest(state['optimizer']), global_metrics_sha256=value_digest(state['metrics']))
            for rank, worker in enumerate(state['private_workers']):
                if (worker['rank'] != rank or worker['local_source_cursor'] != self.cursor(state['completed'], rank)
                        or worker['replica'] != expected_replica):
                    raise ValueError('Every actual private worker/source cursor must remain exact')
                _validate_rng(worker['rng'], self.device)
        except Exception as exc:
            error = repr(exc)
        self.coordinated(error, 'complete shared/private recovery before mutation')
        if any(m != expected_manifest for m in self.gather(expected_manifest)):
            raise ValueError('Every replica must retain the same independent external semantic manifest')
        with torch.no_grad():
            for name, p in self.named:
                p.copy_(state['model']['parts'][name])
        self.optimizer.load_state_dict(_cpu(state['optimizer']))
        self.optimizer.zero_grad(set_to_none=True)
        restore_rng(state['private_workers'][self.rank]['rng'], self.device)
        self.completed, self._metrics, self.failed = state['completed'], _cpu(state['metrics']), False
        self.intact()
        return self.completed
