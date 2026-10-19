"""Complete native code stage with compact collectives/history and private recovery.

Neural sampling, code execution, objectives, source normalization and AdamW
use the unchanged original kernels. Full reference data is stored once; all
original dynamic observations, private RNGs and source cursors remain exact.
This owns no resources and cannot admit its own full-size GPU training.
"""
import copy
from pathlib import Path

import torch

from .codenet_checkpoint_v1 import capture_rng,restore_rng
from .code_residual_checkpoint_v1 import _cpu,_validate_adam,_validate_rng
from .corpus import file_hash
from .frozen_own_actor_reference_v1 import signature
from .native_reference_records_v1 import NativeReferenceRecords,PROTOCOL
from .native_suffix_train_data_v1 import NativeSuffixTrainData
from .native_suffix_validation_stage_v2 import NativeSuffixValidationStageV2
from .own_dense_actor_checkpoint_v1 import selected
from .parallel_function_repair_stage_v1 import value_digest,manifest
from .parallel_native_suffix_stage_v1 import sum_gradients,private_seeds,group_gradients
from .whole_human_sft_evidence_v1 import tensor_digest

FORMAT='own-native-code-compact-collectives-history-shared-AdamW-private-RNG-v3'


class NativeSuffixCompactStageV3(NativeSuffixValidationStageV2):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        error=None
        try:
            module=Path(__file__).resolve()
            record_module=module.with_name('native_reference_records_v1.py')
            if any(self.sources.get(str(p))!=file_hash(p) for p in [module,record_module]):
                raise ValueError('The actual compact stage and reference-record protocol must be explicitly source-bound')
            self.compact_source_files=[module,record_module]
            self.reference_records=NativeReferenceRecords(self.reference_binding)
        except Exception as exc:error=repr(exc)
        self.coordinated(error,'compact native history construction')

    def binding(self):
        return dict(super().binding(),format=FORMAT,history_reference_protocol=PROTOCOL)

    def intact(self):
        super().intact()
        if hasattr(self,'reference_records'):self.check_history_reference()

    def check_history_reference(self):
        self.reference_records.check(self.reference_binding)
        if any(self.sources.get(str(p))!=file_hash(p) for p in self.compact_source_files):
            raise ValueError('The exact compact stage and reference protocol bytes changed')

    def gather_observations(self,value):
        error=packed=None
        try:packed=self.reference_records.pack_observation(value)
        except Exception as exc:error=dict(error=repr(exc),complete_original_local_observation=copy.deepcopy(value))
        self.coordinated(error,'native source observation before compact collective')
        values=self.gather(packed)
        error=result=None
        try:
            result=[self.reference_records.expand_observation(r,share_reference=True) for r in values]
            if [r['rank'] for r in result]!=list(range(self.world)):
                raise ValueError('Every original actual rank must retain its exact compact collective position')
        except Exception as exc:error=dict(error=repr(exc),all_actual_compact_rank_observations=values)
        self.coordinated(error,'native source observation after compact collective')
        return result

    def expanded_metrics(self):
        """Yield one original complete record at a time; never expand a whole run."""
        for row in self._metrics:yield self.reference_records.expand(row)

    def validate_records(self,rows):
        if not isinstance(rows,list) or not 0<len(rows)<=len(self.entries):
            raise ValueError('Every actual complete compact source prefix must remain')
        with self.checked_validation_sources():
            for index,row in enumerate(rows):
                super().validate_record(self.reference_records.expand(row,share_reference=True),index)

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
            error = dict(exception_type=type(exc).__name__, message=str(exc),
                complete_actual_failure_observation=copy.deepcopy(getattr(exc, 'report', None)))
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
        self._metrics.append(copy.deepcopy(compact_row))
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
        self.intact()
        return self.completed

