"""Native code updates with durable pre-Adam observations and bounded history.

Original sampling, objectives, source normalizers and AdamW stay unchanged.
Snapshots retain receipt envelopes; complete observations and typed failures
remain in immutable indexed files. This stage owns no GPU lease or admission.
"""
import copy
from pathlib import Path

import torch

from .codenet_checkpoint_v1 import capture_rng, restore_rng
from .code_residual_checkpoint_v1 import _cpu, _validate_adam, _validate_rng
from .corpus import file_hash
from .frozen_own_actor_reference_v1 import signature
from .indexed_native_history_v1 import Reader, limits, sha
from .indexed_native_history_v2 import Writer, PROTOCOL as FAILURE_PROTOCOL, freeze_observation
from .native_reference_records_v1 import PROTOCOL, canonical
from .native_suffix_compact_stage_v3 import NativeSuffixCompactStageV3
from .native_suffix_train_data_v1 import NativeSuffixTrainData
from .own_dense_actor_checkpoint_v1 import selected
from .parallel_function_repair_stage_v1 import value_digest, manifest
from .parallel_native_suffix_stage_v1 import sum_gradients, private_seeds, group_gradients
from .whole_human_sft_evidence_v1 import tensor_digest

FORMAT = 'own-native-code-indexed-receipts-shared-AdamW-private-RNG-v4'
HISTORY = 'own-native-complete-pre-Adam-observation-receipt-envelopes-v4'


class NativeSuffixIndexedStageV4(NativeSuffixCompactStageV3):
    def __init__(self, *args, history_budgets, **kwargs):
        self.history_budgets = limits(history_budgets)
        self.history_writer = self.history_reader = self.history_start = None
        self._history_readers = {}
        super().__init__(*args, **kwargs)
        error = None
        try:
            directory = Path(__file__).resolve().parent
            self.indexed_source_files = [Path(__file__).resolve(), directory / 'indexed_native_history_v1.py',
                directory / 'indexed_native_history_v2.py']
            if (any(self.sources.get(str(p)) != file_hash(p) for p in self.indexed_source_files)
                    or self.history_budgets['maximum_records'] != len(self.entries)
                    or self.mode != 'mechanics' and self.history_budgets['minimum_free_bytes'] < 40 * 1024**3):
                raise ValueError('All actual indexed source bytes, complete source-prefix capacity and unchanged production reserve are required')
            self.check_history_reference()
        except Exception as exc: error = repr(exc)
        self.coordinated(error, 'indexed native history construction')

    def binding(self):
        return dict(super().binding(), format=FORMAT, durable_history_protocol=HISTORY,
            independent_failure_protocol=FAILURE_PROTOCOL, history_budgets=self.history_budgets)

    def check_history_reference(self):
        super().check_history_reference()
        if hasattr(self, 'indexed_source_files') and any(self.sources.get(str(p)) != file_hash(p) for p in self.indexed_source_files):
            raise ValueError('Every exact indexed stage and success/failure storage source byte must remain')

    def history_guard(self):
        self.check_history_reference()
        self.data.guard.check()
        self.reference.assert_bound(self.model)

    def intact(self):
        super().intact()
        if self.history_reader is not None:
            if (self.history_reader.header['starting_completed'] != self.history_start
                    or self.history_reader.budgets != self.history_budgets
                    or file_hash(self.history_reader.root / 'reference.json.gz') != self.history_reader.header_sha256
                    or self.rank == 0 and (self.history_writer.failed or self.history_writer.ticket is not None
                        or self.history_writer.pending_model_commit
                        or self.history_writer.records != self.completed - self.history_start)):
                raise ValueError('The exact active attempt and healthy clear-gradient completed history boundary must remain')

    def history_file_reader(self, directory, header_sha256):
        key = (directory, header_sha256)
        if key not in self._history_readers:
            reader = Reader(directory, expected_header_sha256=header_sha256,
                maximum_compact_record_bytes=self.history_budgets['maximum_compact_record_bytes'],
                maximum_expanded_record_bytes=self.history_budgets['maximum_expanded_record_bytes'], guard=self.history_guard)
            if (reader.budgets != self.history_budgets or reader.codec.identity != self.reference_records.identity
                    or reader.codec.original_bytes != self.reference_records.original_bytes):
                raise ValueError('Every original attempt must retain this exact reference and complete unchanged bounds')
            # Retain one full reference across attempts as well as across rows.
            reader.header['reference_binding'] = self.reference_records.reference
            reader.codec = self.reference_records
            self._history_readers[key] = reader
        return self._history_readers[key]

    def open_history(self, directory):
        error = common = None
        try:
            self.intact()
            if self.failed or self.history_reader is not None or self.completed >= len(self.entries):
                raise ValueError('One fresh immutable history attempt at a healthy remaining optimizer boundary is required')
            common = dict(directory=str(Path(directory).absolute()), starting_completed=self.completed,
                budgets=self.history_budgets, reference_sha256=self.reference_records.identity)
        except Exception as exc: error = repr(exc)
        self.coordinated(error, 'open immutable history preflight')
        if any(v != common for v in self.gather(common)):
            raise ValueError('Every replica must retain the same exact new history attempt')
        error = header = None
        if self.rank == 0:
            try:
                Path(common['directory']).parent.mkdir(parents=True, exist_ok=True)
                self.history_writer = Writer(common['directory'], reference_binding=self.reference_binding,
                    starting_completed=self.completed, budgets=self.history_budgets, guard=self.history_guard)
                header = self.history_writer.header_sha256
            except Exception as exc: error = repr(exc)
        self.coordinated(error, 'fund original history header')
        headers = self.gather(header)
        error = reader = None
        try:
            if headers[0] is None or any(h is not None for h in headers[1:]):
                raise ValueError('Only root may create the exact shared history header')
            reader = self.history_file_reader(common['directory'], headers[0])
            if reader.header['starting_completed'] != self.completed:
                raise ValueError('A new recovery attempt must begin at exactly the saved complete source cursor')
        except Exception as exc: error = repr(exc)
        self.coordinated(error, 'all replica original history header readback')
        self.history_reader, self.history_start = reader, self.completed
        return dict(directory=common['directory'], header_sha256=headers[0], starting_completed=self.completed)

    def reserve_history(self):
        error = None
        try:
            if self.history_reader is None or self.rank == 0 and type(self.history_writer) is not Writer:
                raise ValueError('A separately funded actual indexed attempt must precede neural sampling')
            if self.rank == 0: self.history_writer.reserve_next()
        except Exception as exc: error = repr(exc)
        self.coordinated(error, 'both complete success and failure allocations before native sampling')

    def publish_observation(self, row, compact):
        error = envelope = None
        if self.rank == 0:
            try:
                receipt = self.history_writer.append(row)
                envelope = dict(format=HISTORY, directory=str(self.history_reader.root),
                    header_sha256=self.history_reader.header_sha256, record=receipt)
            except Exception as exc:
                error = dict(error=repr(exc), complete_original_global_observation=row)
        self.coordinated(error, 'complete original observation durable before Adam')
        envelopes = self.gather(envelope)
        error = observed = None
        try:
            if envelopes[0] is None or any(e is not None for e in envelopes[1:]):
                raise ValueError('Exactly root must publish the complete pre-Adam observation')
            observed = envelopes[0]
            durable = self.history_reader.compact(observed['record'])
            if value_digest(durable) != value_digest(compact):
                raise ValueError('Every actual replica must independently read its exact full observation before Adam')
        except Exception as exc: error = repr(exc)
        self.coordinated(error, 'every peer exact pre-Adam observation readback')
        return observed

    def acknowledge_history(self, step):
        error = None
        if self.rank == 0:
            try: self.history_writer.acknowledge_success(step)
            except Exception as exc: error = repr(exc)
        self.coordinated(error, 'all peer actual Adam boundary and funded evidence slot')

    def preserve_failed_update(self, kernel, row, exception):
        local = dict(rank=self.rank, actual_original_source_positions=self.entries[self.completed][self.rank:self.rank + 1],
            complete_actual_local_kernel=kernel)
        observed = self.gather(local)
        error = receipt = None
        if self.rank == 0:
            try:
                captured = dict(attempted_step=self.completed + 1, actual_exception=exception,
                    all_actual_rank_observations=observed, complete_global_row_if_constructed=row,
                    complete_coordinated_failure_observations=self.failure_observations,
                    successful_optimizer_commit_certified=False)
                receipt = self.history_writer.preserve_failure(captured)
            except Exception as exc: error = repr(exc)
        self.coordinated(error, 'complete typed original failure evidence in independent funded blocks')
        self.last_failure_receipt = self.gather(receipt)[0]

    def validate_records(self, rows):
        if type(rows) is not list or not 0 < len(rows) <= len(self.entries):
            raise ValueError('Every exact original committed receipt prefix is required')
        previous, seen, sequence, expected_previous = None, set(), 0, None
        with self.checked_validation_sources():
            for index, envelope in enumerate(rows):
                if type(envelope) is not dict or set(envelope) != {'format', 'directory', 'header_sha256', 'record'} or envelope['format'] != HISTORY:
                    raise ValueError('A complete bounded indexed receipt envelope is required')
                key = (envelope['directory'], envelope['header_sha256'])
                reader = self.history_file_reader(*key)
                if key != previous:
                    if key in seen or reader.header['starting_completed'] != index:
                        raise ValueError('Every immutable attempt begins at its original exact completed prefix')
                    seen.add(key); previous = key; sequence = 0; expected_previous = reader.header_sha256
                sequence += 1
                receipt = envelope['record']
                if (receipt['sequence'] != sequence or receipt['step'] != index + 1
                        or receipt['previous_receipt_sha256'] != expected_previous):
                    raise ValueError('Every original receipt and complete source step must retain its exact chained position')
                compact = reader.compact(receipt)
                self.validate_record(self.reference_records.expand(compact, share_reference=True), index)
                expected_previous = sha(canonical(receipt))

    def expanded_metrics(self):
        for envelope in self._metrics:
            reader = self.history_file_reader(envelope['directory'], envelope['header_sha256'])
            yield self.reference_records.expand(reader.compact(envelope['record']))


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
        self.reserve_history()
        positions = self.entries[self.completed]
        before_rng = value_digest(capture_rng(self.device))
        kernel = row = None
        error = None
        try:
            if self.rank < len(positions):
                position = positions[self.rank]
                if self.data.information(position)['native_execution_eligible']:
                    with self.data.episode(position) as episode:
                        kernel = group_gradients(self.model, episode, self.reference,
                            profile=self.profile, seeds=private_seeds(self.options['seed'], self.completed + 1, position, self.options['group_size']),
                            purpose='CPU-mechanics' if self.mode == 'mechanics' else 'GPU-runtime',
                            **{k: self.options[k] for k in ['maximum_cache_bytes', 'KL_weight', 'anchor_weight',
                                'cache_absolute_bound', 'execution_workers', 'CPU_mechanics_horizon']})
                else:
                    kernel = self.ce_only_gradients(position)
                self.validate_kernel(kernel, position, self.completed + 1)
                for p in self.parameters:
                    p.grad.div_(len(positions))
            else:
                self.model.train()
                self.model.policy.gradient_checkpointing = True
            if value_digest(capture_rng(self.device)) != before_rng:
                raise ValueError('Private seeded attempts must not consume the original global RNG')
        except Exception as exc:
            captured, metadata = freeze_observation(getattr(exc, 'report', None))
            error = dict(exception_type=type(exc).__name__, message=str(exc),
                complete_actual_failure_observation=captured, original_tensor_metadata=metadata)
        try:
            self.coordinated(error, 'whole source sampling/verifier/gradient before reduction')
            observed = self.gather_observations(dict(rank=self.rank, actual_original_positions=positions[self.rank:self.rank+1], kernel=kernel))
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
                original_response_EOS_targets=sum(i['original_response_EOS_targets'] for i in self.expected[self.completed]),
                complete_original_context_tokens=sum(i['original_context_tokens'] for i in self.expected[self.completed]),
                actual_unclipped_global_actor_gradient_sha256=gradient_sha, gradient_norm_before_clip=norm,
                learning_rate=self.options['learning_rate'], private_global_RNG_unchanged_by_update=True,
                one_global_gradient_SUM_and_Adam_commit=True, genuine_empty_ranks_have_no_source_forward=True)
            error = None
            try:
                self.validate_record(row, self.completed)
            except Exception as exc:
                error = repr(exc)
            self.coordinated(error, 'global source record before Adam')
            error = compact_row = None
            try:
                compact_row = self.reference_records.pack(row)
            except Exception as exc:
                error = dict(error=repr(exc),complete_original_global_observation=row)
            self.coordinated(error, 'complete native source record before compact history')
            if any(v != value_digest(compact_row) for v in self.gather(value_digest(compact_row))):
                raise ValueError('Actual shared source records and all summed gradients must agree before Adam')
            envelope = self.publish_observation(row, compact_row)
            error = None
            try:
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
            except Exception as exc: error = repr(exc)
            self.coordinated(error, 'actual native Adam invocation')
            error = None
            try:
                if (signature(self.fixed) != self.fixed_signature or any(not bool(torch.isfinite(p).all()) for p in self.parameters)
                        or any(float(self.optimizer.state[p]['step']) != self.completed + 1 for p in self.parameters)):
                    raise ValueError('Every selected weight/clock and unchanged fixed value must preserve the completed update')
                self.reference.assert_bound(self.model)
            except Exception as exc:
                error = repr(exc)
            self.coordinated(error, 'post-Adam completed boundary')
            self.acknowledge_history(self.completed + 1)
        except BaseException as exc:
            self.failed = True
            self.optimizer.zero_grad(set_to_none=True)
            self.preserve_failed_update(kernel, row, repr(exc))
            raise
        self.completed += 1
        self._metrics.append(copy.deepcopy(envelope))
        return copy.deepcopy(row)

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
                    history_reference_protocol=PROTOCOL, reference_binding_sha256=self.reference_records.identity,
                    indexed_history_protocol=HISTORY, independent_failure_protocol=FAILURE_PROTOCOL,
                    complete_observations_are_durable_receipts=True,
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
            self.check_history_reference()
            current = selected(self.model, self.optimizer, ('actor',))
            self.data.guard.check()
            if ([(n, id(p)) for n, p in current] != [(n, id(p)) for n, p in self.named]
                    or type(self.data) is not NativeSuffixTrainData or self.data.role != 'train' or self.data.purpose != ('CPU-mechanics' if self.mode == 'mechanics' else 'capacity' if self.mode == 'capacity' else 'optimization')
                    or self.binding() != self.initial_binding or signature(self.fixed) != self.fixed_signature
                    or manifest(state) != expected_manifest or not expected_manifest
                    or state['format'] != FORMAT or state['world_size'] != self.world or state['profile'] != self.profile
                    or state['mode'] != self.mode or state['precision'] != self.precision or state['planned'] != len(self.entries)
                    or type(state['completed']) is not int or not 0 < state['completed'] <= len(self.entries)
                    or state['completed'] == len(self.entries) and not allow_completed
                    or state['schedule_identity_sha256'] != self.schedule_identity or state['options_sha256'] != self.options_digest
                    or state['global_cursor'] != self.cursor(state['completed']) or len(state['metrics']) != state['completed']
                    or state['groups'] != ['actor'] or state['optimized_names'] != [n for n, _ in self.named]
                    or state['reference_binding'] != self.reference_binding
                    or state.get('history_reference_protocol') != PROTOCOL
                    or state.get('indexed_history_protocol') != HISTORY
                    or state.get('independent_failure_protocol') != FAILURE_PROTOCOL
                    or state.get('complete_observations_are_durable_receipts') is not True
                    or state.get('reference_binding_sha256') != self.reference_records.identity
                    or value_digest(state['reference_binding']) != self.reference_records.identity
                    or state['candidate_promoted'] is not False
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
        self.history_writer = self.history_reader = self.history_start = None
        self.intact()
        return self.completed
