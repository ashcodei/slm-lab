"""Complete-response coding CE with explicit source weights and shared recovery.

This stage owns no resources. Mechanics fixtures cannot authorize full-model
training. Public synthetic tests are never used as a correctness/RL oracle.
"""
import copy
import hashlib
import math
import struct

import torch
from torch import distributed as dist
from torch.nn import functional as F

from .codenet_checkpoint_v1 import capture_rng, restore_rng
from .code_residual_checkpoint_v1 import _cpu, _validate_adam, _validate_rng
from .frozen_own_actor_reference_v1 import signature
from .mixed_coding_SFT_data_v1 import MixedCodingSFTData, PROFILES
from .own_dense_actor_checkpoint_v1 import selected
from .parallel_batched_reasoning_stage_v2 import sum_gradients
from .parallel_function_repair_stage_v1 import value_digest, manifest
from .whole_human_sft_evidence_v1 import tensor_digest

FORMAT = 'own-complete-mixed-coding-supervised-shared-Adam-RNG-v1'
OPTIONS = {'learning_rate', 'gradient_clip', 'maximum_microbatch_examples', 'maximum_padded_tokens',
    'gradient_bucket_values', 'epochs', 'width', 'seed'}


class ParallelMixedCodingSFTStage:
    def __init__(self, model, optimizer, data, entries, profile, options, sources, *, mode,
                 rank=0, world=1, control_group=None, gradient_group=None):
        if (type(world) is not int or not 1 <= world <= 4 or type(rank) is not int or not 0 <= rank < world
                or world == 1 and dist.is_initialized()
                or world > 1 and (not dist.is_initialized() or dist.get_world_size() != world
                    or dist.get_world_size(control_group) != world or dist.get_rank(control_group) != rank
                    or dist.get_world_size(gradient_group) != world or dist.get_rank(gradient_group) != rank
                    or dist.get_backend(control_group) != 'gloo')):
            raise ValueError('Exact replica ranks and Gloo control with matching actual gradient group required')
        self.rank, self.world, self.control_group, self.gradient_group = rank, world, control_group, gradient_group
        self.failure_observations = []
        error = initial = None
        try:
            if (type(data) is not MixedCodingSFTData or data.role != 'train' or data.purpose != 'optimization'
                    or data.use_as_RL_reward_oracle_forbidden is not True or profile not in PROFILES
                    or mode not in ['mechanics', 'capacity', 'train'] or set(options) != OPTIONS
                    or optimizer.state or not sources or any(not isinstance(k, str) or not isinstance(v, str)
                        or len(v) != 64 or any(c not in '0123456789abcdef' for c in v) for k, v in sources.items())):
                raise ValueError('Exact complete train-only mixed data, fresh Adam, fixed profile/options and bindings required')
            if (any(type(options[k]) is not int or options[k] < 1 for k in OPTIONS - {'learning_rate', 'gradient_clip'})
                    or any(type(options[k]) not in [int, float] or not math.isfinite(options[k]) or options[k] <= 0
                        for k in ['learning_rate', 'gradient_clip'])
                    or options['width'] % world or options['maximum_padded_tokens'] < 4096):
                raise ValueError('Finite positive settings and complete4096 native context support required')
            data.budget(entries)
            if any(len(e) > options['width'] for e in entries): raise ValueError('A batch exceeds the declared complete width')
            self.model, self.optimizer, self.data = model, optimizer, data
            self.entries, self.options, self.sources = copy.deepcopy(entries), copy.deepcopy(options), dict(sources)
            self.profile, self.mode, self.precision = profile, mode, 'fp32'
            self.device = next(model.parameters()).device
            if world > 1 and dist.get_backend(gradient_group) != ('nccl' if self.device.type == 'cuda' else 'gloo'):
                raise ValueError('NCCL/CUDA or Gloo/CPU actual gradient transport required')
            self.named = selected(model, optimizer, ('actor',))
            self.parameters = [p for _, p in self.named]
            if (any(p.dtype != torch.float32 or p.device != self.device or p.grad is not None for p in model.parameters())
                    or optimizer.param_groups[0]['lr'] != options['learning_rate']):
                raise ValueError('Clear-gradient matching ordinary Adam and actual FP32 own replica required')
            if mode == 'mechanics':
                if self.device.type != 'cpu' or torch.cuda.is_initialized(): raise ValueError('Explicit own-small CPU mechanics only')
            elif (world != 4 or self.device != torch.device('cuda', rank) or not torch.are_deterministic_algorithms_enabled()
                    or torch.backends.cuda.matmul.allow_tf32 or model.counts()['actual_unique'] != 999706442
                    or model.base.counts()['total_unique'] != 984399690):
                raise ValueError('Four deterministic actual full-size own GPUreplicas required for capacity/train')
            if mode == 'train' and entries != data.complete_schedule(**{k: options[k] for k in ['epochs', 'width', 'seed']}):
                raise ValueError('All complete source epochs must remain in their original global batches')
            chosen = {id(p) for p in self.parameters}
            self.fixed = [p for p in model.parameters() if id(p) not in chosen]
            self.fixed_signature = signature(self.fixed)
            self.base_digest = tensor_digest(model.base.named_parameters())
            self.schedule_identity = data.identity(entries)
            self.expected = [data.metadata(e, profile) for e in entries]
            self.optimizer_settings = {k: copy.deepcopy(v) for k, v in optimizer.param_groups[0].items() if k != 'params'}
            for k in ['lr', 'eps', 'weight_decay']:
                v = self.optimizer_settings[k]
                if type(v) not in [int, float] or not math.isfinite(v) or v < 0 or k != 'weight_decay' and v == 0:
                    raise ValueError('Finite actual ordinary Adam settings required')
            if any(type(b) not in [int, float] or not math.isfinite(b) or not 0 <= b < 1 for b in self.optimizer_settings['betas']):
                raise ValueError('Both actual Adam betas must be finite and in[0,1)')
            self.initial_binding = copy.deepcopy(self.binding())
            self.options_digest = value_digest(self.initial_binding)
            self.completed, self._metrics, self.failed = 0, [], False
            initial = dict(binding=self.initial_binding, base=self.base_digest,
                actor=tensor_digest(model.parts.named_parameters()), counts=model.counts(), device_type=self.device.type)
        except Exception as exc: error = repr(exc)
        self.coordinated(error, 'construction')
        if any(v != initial for v in self.gather(initial)):
            raise ValueError('All actual replica initializers/options/complete sources must agree')

    def gather(self, value):
        if self.world == 1: return [value]
        result = [None] * self.world
        dist.all_gather_object(result, value, group=self.control_group)
        return result

    def coordinated(self, error, phase):
        errors = self.gather(error)
        if any(e is not None for e in errors):
            self.failure_observations.append(copy.deepcopy(dict(phase=phase, actual_rank_errors=errors)))
            raise ValueError('Coordinated mixed coding SFT ' + phase + ' rejection: ' + repr(errors))

    @property
    def metrics(self): return copy.deepcopy(self._metrics)

    def binding(self):
        return dict(format=FORMAT, profile=self.profile, mode=self.mode, precision=self.precision, world=self.world,
            options=self.options, sources=self.sources, schedule=self.schedule_identity,
            complete_entries_sha256=value_digest(self.entries), optimizer_settings=self.optimizer_settings,
            normalization='positive_source_weighted_complete_response_CE_SUM/global_weighted_target_count',
            synthetic_generated_tests_never_rewarded=True, epoch_loss_fraction_not_guaranteed=True)

    def intact(self):
        current = selected(self.model, self.optimizer, ('actor',))
        if (type(self.data) is not MixedCodingSFTData or self.data.role != 'train' or self.data.purpose != 'optimization'
                or self.data.use_as_RL_reward_oracle_forbidden is not True
                or [(n, id(p)) for n, p in current] != [(n, id(p)) for n, p in self.named]
                or self.binding() != self.initial_binding or signature(self.fixed) != self.fixed_signature
                or type(self.completed) is not int or not 0 <= self.completed <= len(self.entries)
                or len(self._metrics) != self.completed or any(p.grad is not None for p in self.model.parameters())
                or {k: v for k, v in self.optimizer.param_groups[0].items() if k != 'params'} != self.optimizer_settings):
            raise ValueError('Immutable source/options/fixed parameters and complete clear-gradient boundary required')
        self.data.guard.check()
        if self.completed == 0:
            if self.optimizer.state: raise ValueError('A zero source cursor requires empty Adam')
        elif (len(self.optimizer.state) != len(self.parameters)
                or any(float(self.optimizer.state[p]['step']) != self.completed for p in self.parameters)):
            raise ValueError('Every selected actual Adam clock must match the exact complete source prefix')

    def _local_gradients(self, positions, denominator):
        observations = self._active_local_observations
        self.model.train()
        self.model.policy.gradient_checkpointing = True
        remaining = list(positions)
        while remaining:
            micro, longest = [], 0
            while remaining:
                length = self.data.information(remaining[0])['complete_context_tokens']
                if micro and (len(micro) >= self.options['maximum_microbatch_examples']
                        or (len(micro) + 1) * max(longest, length) > self.options['maximum_padded_tokens']): break
                micro.append(remaining.pop(0))
                longest = max(longest, length)
            batch = self.data.batch(micro, self.device)
            for j, p in enumerate(micro):
                info = self.data.information(p)
                ids, labels = self.data.sequence(p)
                start, length = info['original_prompt_positions'], len(ids)
                if (ids[0] != 1 or ids[-1] != 2 or labels != [-100] * start + ids[start:]
                        or length != info['complete_context_tokens'] or length - start != info['response_EOS_targets']
                        or hashlib.sha256(struct.pack('<' + 'H' * length, *ids)).hexdigest() != info['complete_native_input_sha256']
                        or batch['input_ids'][j, :length].tolist() != ids or batch['labels'][j, :length].tolist() != labels
                        or not bool(batch['attention_mask'][j, :length].all()) or bool(batch['attention_mask'][j, length:].any())
                        or bool((batch['input_ids'][j, length:] != 0).any()) or bool((batch['labels'][j, length:] != -100).any())):
                    raise ValueError('All original IDs, full question/response/EOS targets and padding must reconstruct before gradients')
            with torch.autocast(self.device.type, enabled=False):
                logits = self.model(batch['input_ids'], attention_mask=batch['attention_mask'], pixel_values=None)['logits']
                if logits.dtype != torch.float32 or not bool(torch.isfinite(logits).all()):
                    raise ValueError('Every actual original FP32 forward must remain finite')
                ce = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), batch['labels'][:, 1:].reshape(-1),
                    reduction='none', ignore_index=-100).reshape(len(micro), -1).sum(-1)
                weights = torch.tensor([self.data.weight(self.profile, p) for p in micro], device=self.device, dtype=torch.float32)
                loss = (ce * weights).sum() / denominator
                if not bool(torch.isfinite(loss)): raise ValueError('Complete positive weighted CE must remain finite')
                loss.backward()
            sums = ce.detach().cpu().tolist()
            observations.append(dict(actual_positions=micro, complete_sources=[self.data.information(p) for p in micro],
                actual_original_CE_sums=sums, actual_weights=[self.data.weight(self.profile, p) for p in micro],
                actual_context_tokens=sum(self.data.information(p)['complete_context_tokens'] for p in micro),
                actual_response_EOS_targets=sum(self.data.information(p)['response_EOS_targets'] for p in micro),
                actual_padded_input_shape=list(batch['input_ids'].shape), actual_logits_dtype=str(logits.dtype),
                actual_global_weighted_target_denominator=denominator, whole_original_EOS_targets_preserved=True))
            del loss, logits, ce, batch
        return observations

    def validate_records(self, rows):
        if not isinstance(rows, list) or not 1 <= len(rows) <= len(self.entries): raise ValueError('Complete nonempty metric prefix required')
        for index, row in enumerate(rows): self.validate_record(row, index)

    def validate_record(self, row, index):
        expected = self.expected[index]
        if (row['step'] != index + 1 or row['actual_positions'] != self.entries[index] or row['source_metadata'] != expected
                or row['actual_world_size'] != self.world or row['normalization'] != self.initial_binding['normalization']
                or row['learning_rate'] != self.options['learning_rate'] or len(row['actual_rank_observations']) != self.world
                or type(row['gradient_norm_before_clip']) not in [int, float] or not math.isfinite(row['gradient_norm_before_clip'])
                or row['gradient_norm_before_clip'] < 0 or row['private_global_RNG_unchanged'] is not True
                or row['one_global_gradient_SUM_and_Adam_commit'] is not True):
            raise ValueError('Every actual complete source and globally normalized update must reconstruct')
        all_sums, weighted = [], []
        for rank, actual in enumerate(row['actual_rank_observations']):
            local = self.entries[index][rank::self.world]
            micros = actual['microbatches']
            if (actual['rank'] != rank or actual['actual_positions'] != local
                    or [p for m in micros for p in m['actual_positions']] != local):
                raise ValueError('Each actual replica must preserve full local sources or a genuinely empty tail')
            for m in micros:
                positions = m['actual_positions']
                infos = [self.data.information(p) for p in positions]
                weights = [self.data.weight(self.profile, p) for p in positions]
                if (not positions or len(positions) > self.options['maximum_microbatch_examples']
                        or m['complete_sources'] != infos or m['actual_weights'] != weights
                        or m['actual_context_tokens'] != sum(i['complete_context_tokens'] for i in infos)
                        or m['actual_response_EOS_targets'] != sum(i['response_EOS_targets'] for i in infos)
                        or m['actual_padded_input_shape'] != [len(positions), max(i['complete_context_tokens'] for i in infos)]
                        or math.prod(m['actual_padded_input_shape']) > self.options['maximum_padded_tokens']
                        or m['actual_logits_dtype'] != 'torch.float32' or not m['whole_original_EOS_targets_preserved']
                        or m['actual_global_weighted_target_denominator'] != expected['actual_global_weighted_targets']
                        or len(m['actual_original_CE_sums']) != len(positions)
                        or any(type(v) not in [int, float] or not math.isfinite(v) or v < 0 for v in m['actual_original_CE_sums'])):
                    raise ValueError('Every actual full native microbatch, precision, CE sum and global denominator must reconstruct')
                all_sums.extend(m['actual_original_CE_sums'])
                weighted.extend(v * w for v, w in zip(m['actual_original_CE_sums'], weights))
        if (row['original_token_mean_NLL'] != math.fsum(all_sums) / expected['response_EOS_targets']
                or row['weighted_token_mean_NLL'] != math.fsum(weighted) / expected['actual_global_weighted_targets']):
            raise ValueError('Actual complete CE sums must rederive both reported loss metrics')

    def update(self):
        error = None
        try:
            if self.failed or self.completed >= len(self.entries): raise ValueError('One healthy remaining full-source update required')
            self.intact()
            if self.expected[self.completed] != self.data.metadata(self.entries[self.completed], self.profile):
                raise ValueError('The original complete source/weight metadata changed')
        except Exception as exc: error = repr(exc)
        self.coordinated(error, 'update preflight')
        positions = self.entries[self.completed]
        before_rng = value_digest(capture_rng(self.device))
        self._active_local_observations = []
        micros, error = self._active_local_observations, None
        try:
            micros = self._local_gradients(positions[self.rank::self.world], self.expected[self.completed]['actual_global_weighted_targets'])
            if value_digest(capture_rng(self.device)) != before_rng: raise ValueError('Deterministic supervised updates cannot consume private global RNG')
            self.data.guard.check()
        except Exception as exc: error = dict(error=repr(exc), actual_completed_local_microbatches=copy.deepcopy(micros))
        try:
            self.coordinated(error, 'complete local source gradients before reduction')
            observed = self.gather(dict(rank=self.rank, actual_positions=positions[self.rank::self.world], microbatches=micros))
            error = None
            try:
                count = sum_gradients(self.parameters, self.device, self.world, self.options['gradient_bucket_values'], self.gradient_group)
                if count != len(self.parameters) or any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in self.parameters):
                    raise ValueError('All actual selected gradients must be finite after one globally normalized SUM')
                gradient = tensor_digest((n, p.grad) for n, p in self.named)
                norm = float(torch.nn.utils.clip_grad_norm_(self.parameters, self.options['gradient_clip'], error_if_nonfinite=True))
            except Exception as exc: error = repr(exc)
            self.coordinated(error, 'summed gradient validation before Adam')
            all_m = [m for r in observed for m in r['microbatches']]
            row = dict(step=self.completed + 1, actual_positions=list(positions), source_metadata=self.expected[self.completed],
                actual_world_size=self.world, normalization=self.initial_binding['normalization'], actual_rank_observations=observed,
                original_token_mean_NLL=math.fsum(v for m in all_m for v in m['actual_original_CE_sums']) / self.expected[self.completed]['response_EOS_targets'],
                weighted_token_mean_NLL=math.fsum(v * w for m in all_m for v, w in zip(m['actual_original_CE_sums'], m['actual_weights']))
                    / self.expected[self.completed]['actual_global_weighted_targets'],
                actual_unclipped_global_actor_gradient_sha256=gradient, gradient_norm_before_clip=norm,
                learning_rate=self.options['learning_rate'], private_global_RNG_unchanged=True, one_global_gradient_SUM_and_Adam_commit=True)
            error = None
            try: self.validate_record(row, self.completed)
            except Exception as exc: error = repr(exc)
            self.coordinated(error, 'complete source record before Adam')
            if any(v != value_digest(row) for v in self.gather(value_digest(row))): raise ValueError('All actual source records/gradients must agree before Adam')
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            error = None
            try:
                if (signature(self.fixed) != self.fixed_signature or any(not bool(torch.isfinite(p).all()) for p in self.parameters)
                        or any(float(self.optimizer.state[p]['step']) != self.completed + 1 for p in self.parameters)):
                    raise ValueError('Actual post-Adam values/clocks or frozen values changed incorrectly')
            except Exception as exc: error = repr(exc)
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
            schedule_identity_sha256=self.schedule_identity, all_consumed_global_source_batches=self.entries[:completed], original_source_role='train')
        if rank is not None: value.update(rank=rank, all_consumed_local_source_batches=[e[rank::self.world] for e in self.entries[:completed]])
        return value

    def snapshot(self):
        error = None
        try:
            self.intact()
            if self.failed or self.completed < 1: raise ValueError('Only a healthy completed optimizer boundary can be published')
            self.validate_records(self._metrics)
            if tensor_digest(self.model.base.named_parameters()) != self.base_digest: raise ValueError('Every frozen base value must remain unchanged')
            adam = _cpu(self.optimizer.state_dict())
            _validate_adam(adam, self.parameters, self.completed)
            rng = capture_rng(self.device)
            _validate_rng(rng, self.device)
            local = dict(rank=self.rank, local_source_cursor=self.cursor(self.completed, self.rank), rng=rng,
                replica=dict(frozen_base_sha256=self.base_digest, model_parts_sha256=tensor_digest(self.model.parts.named_parameters()),
                    optimizer_sha256=value_digest(adam), global_metrics_sha256=value_digest(self._metrics)))
        except Exception as exc: error = repr(exc)
        self.coordinated(error, 'shared snapshot preflight')
        workers = self.gather(local)
        if any(w['replica'] != workers[0]['replica'] for w in workers): raise ValueError('All actual model/Adam/source replicas must agree before shared storage')
        state = receipt = error = None
        if self.rank == 0:
            try:
                payload = self.model.export(self.sources)
                self.model.validate_export(payload, expected_sources=self.sources)
                state = dict(format=FORMAT, world_size=self.world, profile=self.profile, mode=self.mode, precision=self.precision,
                    completed=self.completed, planned=len(self.entries), schedule_identity_sha256=self.schedule_identity,
                    options_sha256=self.options_digest, groups=['actor'], optimized_names=[n for n, _ in self.named], model=payload,
                    optimizer=adam, global_cursor=self.cursor(self.completed), metrics=_cpu(self._metrics), private_workers=_cpu(workers),
                    one_shared_policy_and_Adam_copy_verified_across_all_replicas=True, all_private_RNGs_and_source_cursors_retained=True,
                    candidate_promoted=False, goal_achieved=False)
                receipt = manifest(state)
            except Exception as exc: error = repr(exc)
        self.coordinated(error, 'root shared payload serialization')
        return state, self.gather(receipt)[0]

    def resume(self, state, *, expected_manifest, allow_completed=False):
        error = None
        try:
            current = selected(self.model, self.optimizer, ('actor',))
            self.data.guard.check()
            if ([(n, id(p)) for n, p in current] != [(n, id(p)) for n, p in self.named]
                    or type(self.data) is not MixedCodingSFTData or self.data.role != 'train' or self.data.purpose != 'optimization'
                    or self.data.use_as_RL_reward_oracle_forbidden is not True
                    or self.binding() != self.initial_binding or signature(self.fixed) != self.fixed_signature
                    or not expected_manifest or manifest(state) != expected_manifest or state['format'] != FORMAT
                    or state['world_size'] != self.world or state['profile'] != self.profile or state['mode'] != self.mode
                    or state['precision'] != self.precision or state['planned'] != len(self.entries)
                    or type(state['completed']) is not int or not 0 < state['completed'] <= len(self.entries)
                    or state['completed'] == len(self.entries) and not allow_completed
                    or state['schedule_identity_sha256'] != self.schedule_identity or state['options_sha256'] != self.options_digest
                    or state['global_cursor'] != self.cursor(state['completed']) or len(state['metrics']) != state['completed']
                    or state['groups'] != ['actor'] or state['optimized_names'] != [n for n, _ in self.named]
                    or state['candidate_promoted'] is not False or state['goal_achieved'] is not False
                    or len(state['private_workers']) != self.world or not state['one_shared_policy_and_Adam_copy_verified_across_all_replicas']
                    or not state['all_private_RNGs_and_source_cursors_retained']):
                raise ValueError('Exact retained shared manifest/source/options/optimizer prefix required before mutation')
            self.validate_records(state['metrics'])
            self.model.validate_export(state['model'], expected_sources=self.sources)
            _validate_adam(state['optimizer'], self.parameters, state['completed'])
            if state['optimizer']['param_groups'] != self.optimizer.state_dict()['param_groups']:
                raise ValueError('All actual ordinary Adam options/order must remain unchanged')
            names = {n for n, _ in self.named}
            if any(not torch.equal(t.detach().cpu(), state['model']['parts'][n]) for n, t in self.model.parts.state_dict().items() if n not in names):
                raise ValueError('Every unused critic value must remain unchanged')
            replica = dict(frozen_base_sha256=self.base_digest, model_parts_sha256=tensor_digest(state['model']['parts'].items()),
                optimizer_sha256=value_digest(state['optimizer']), global_metrics_sha256=value_digest(state['metrics']))
            for rank, worker in enumerate(state['private_workers']):
                if worker['rank'] != rank or worker['local_source_cursor'] != self.cursor(state['completed'], rank) or worker['replica'] != replica:
                    raise ValueError('Every actual private worker and source cursor must reconstruct')
                _validate_rng(worker['rng'], self.device)
        except Exception as exc: error = repr(exc)
        self.coordinated(error, 'complete shared/private recovery before mutation')
        if any(m != expected_manifest for m in self.gather(expected_manifest)): raise ValueError('All replicas need the same independent external semantic manifest')
        with torch.no_grad():
            for name, p in self.named: p.copy_(state['model']['parts'][name])
        self.optimizer.load_state_dict(_cpu(state['optimizer']))
        self.optimizer.zero_grad(set_to_none=True)
        restore_rng(state['private_workers'][self.rank]['rng'], self.device)
        self.completed, self._metrics, self.failed = state['completed'], _cpu(state['metrics']), False
        self.intact()
        return self.completed
