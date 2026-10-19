"""Joint source-prefix training with allocated pre-Adam records and recovery.

The caller supplies the frozen original source computation. Every actual
replica independently reads its complete metric before Adam. Success is not
committed by a journal alone: complete selected state, sparse Adam and private
RNG require the separate two-generation durable checkpoint. No job admission.
"""
import copy
import fcntl
from pathlib import Path

from . import joint_indexed_durable_state_v1 as durable
from .codenet_checkpoint_v1 import capture_rng
from .code_residual_checkpoint_v1 import _cpu
from .corpus import file_hash
from .frozen_own_text_reference_v1 import FrozenOwnTextReferenceV1
from .indexed_native_history_v1 import canonical,exclusive,read_regular,limits
from .joint_indexed_history_v1 import Writer,Reader
from .joint_indexed_checkpoint_v1 import ENVELOPE,configuration,scan,validate_metric
from .joint_objective_step_v1 import JointReplicaGuardV1
from .parallel_function_repair_stage_v1 import value_digest
from .shared_joint_checkpoint_v2 import context,agree,control,gather
from .whole_human_sft_evidence_v1 import tensor_digest


class JointIndexedStageV1:
    def __init__(self,view,reference,optimizer,entries,sources,options,*,history_root,history_budgets,source_guard,
                 maximum_attempts=8,control_group=None,gradient_group=None):
        self.view,self.reference,self.optimizer=view,reference,optimizer
        self.rank,self.world=control(control_group);self.control_group,self.gradient_group=control_group,gradient_group
        self.device=next(view.parameters()).device;self.source_guard=source_guard
        self.entries,self.sources,self.options=copy.deepcopy(entries),copy.deepcopy(sources),copy.deepcopy(options)
        self.guard=JointReplicaGuardV1(view,optimizer,control_group=control_group)
        error=identity=None
        try:
            if (not callable(source_guard) or type(reference) is not FrozenOwnTextReferenceV1 or reference.base is not view.actor.base
                    or reference.reference_binding['tokenizer_sha256']!=sources['tokenizer_sha256']):raise ValueError('The exact own shared text reference, own tokenizer and original source guard are required')
            source_guard();reference.assert_bound(full=True)
            context(view,optimizer,self.guard.fixed,sources,self.entries,self.rank,self.world,self.device,control_group)
            self.history=dict(binding=dict(sources=self.sources,model_metadata=self.guard.fixed.metadata,own_text_reference=reference.reference_binding,
                options=self.options,world_size=self.world,optimizer_settings_sha256=value_digest(optimizer.state_dict()['param_groups'])),
                budgets=limits(history_budgets),maximum_attempts=maximum_attempts,history_root=str(Path(history_root).absolute()))
            configuration(self.history,sources=sources,metadata=self.guard.fixed.metadata,world=self.world,optimizer=optimizer,planned=len(entries))
            identity=value_digest(self.history)
        except Exception as exc:error=repr(exc)
        agree(error,identity,control_group,self.world);self.identity=identity
        self.expected_entry_hashes=tuple(value_digest(row) for row in self.entries)
        self.completed=0;self.receipts=[];self.clocks={n:0 for n,_ in view.named_selected()}
        self.writer=self.reader=self.pending=self.latest_metric=self.last_failure=None;self.failed=False

    def coordinated(self,error,identity=None):agree(error,identity,self.control_group,self.world)
    def gather(self,value):return gather(value,self.control_group,self.world)

    def history_guard(self):
        self.source_guard();self.reference.assert_bound();self.guard.fixed.check(self.view,self.optimizer)
        if value_digest(self.history)!=self.identity:raise ValueError('Every original joint source/model/reference/optimizer/history bound must remain')

    def intact(self):
        self.history_guard();self.guard.check()
        if (self.failed or self.completed!=len(self.receipts) or len(self.entries)!=len(self.expected_entry_hashes) or not 0<=self.completed<=len(self.entries)
                or self.completed<len(self.entries) and value_digest(self.entries[self.completed])!=self.expected_entry_hashes[self.completed]):
            raise ValueError('A healthy complete joint optimizer boundary and unchanged original next source are required')

    def open_history(self):
        error=specification=None
        try:
            self.intact()
            if self.reader is not None or self.completed==len(self.entries):raise ValueError('A new immutable attempt requires a remaining restored optimizer boundary')
            specification=value_digest(dict(configuration=self.history,completed=self.completed))
        except Exception as exc:error=repr(exc)
        self.coordinated(error,specification);error=location=None
        if self.rank==0:
            try:
                root=Path(self.history['history_root']);root.mkdir(parents=True,exist_ok=True)
                with (root/'root.lock').open('a+') as lock:
                    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                    binding=root/'binding.json';raw=canonical(self.history)
                    if binding.exists():
                        if read_regular(binding,16*1024**2)!=raw:raise ValueError('An existing journal root cannot change its original experiment or bounds')
                    else:
                        if any(p.name!='root.lock' for p in root.iterdir()):raise ValueError('Preserve every unknown original directory before creating a joint journal')
                        exclusive(binding,raw)
                    directories=sorted(p for p in root.iterdir() if p.name not in ['root.lock','binding.json'])
                    if len(directories)>=self.history['maximum_attempts']:raise ValueError('All attempts, including failed and uncommitted ones, consume the fixed original attempt budget')
                    for index,directory in enumerate(directories,1):
                        if directory.name!='attempt-'+str(index).zfill(6) or not directory.is_dir() or directory.is_symlink():raise ValueError('Every original immutable attempt and allocated failure must remain')
                        old=self.read_header(directory,file_hash(directory/'reference.json.gz'))
                        if old.codec.identity!=value_digest(self.history['binding']) or old.budgets!=self.history['budgets']:raise ValueError('Every earlier failed or successful attempt must retain the exact original binding')
                    directory=root/('attempt-'+str(len(directories)+1).zfill(6))
                    self.writer=Writer(directory,reference_binding=self.history['binding'],starting_completed=self.completed,
                        budgets=self.history['budgets'],guard=self.history_guard)
                    location=dict(directory=str(directory),header_sha256=self.writer.header_sha256)
            except Exception as exc:error=repr(exc)
        self.coordinated(error);location=self.gather(location)[0];error=reader=None
        try:reader=self.read_header(location['directory'],location['header_sha256'])
        except Exception as exc:error=repr(exc)
        self.coordinated(error,location['header_sha256']);self.reader=reader
        return location

    def read_header(self,directory,sha):
        return Reader(directory,expected_header_sha256=sha,maximum_compact_record_bytes=self.history['budgets']['maximum_compact_record_bytes'],
            maximum_expanded_record_bytes=self.history['budgets']['maximum_expanded_record_bytes'],guard=self.history_guard)

    def before_adam(self,metric):
        error=identity=None
        try:
            if self.pending is not None:raise ValueError('Exactly one complete observation and one Adam boundary per original source step are required')
            if value_digest(self.entries[self.completed])!=self.expected_entry_hashes[self.completed]:raise ValueError('The original source entry cannot change during its computation')
            validate_metric(self.view,self.entries[self.completed],metric);identity=value_digest(metric);self.latest_metric=copy.deepcopy(metric)
        except Exception as exc:error=repr(exc)
        self.coordinated(error,identity);error=envelope=None
        if self.rank==0:
            try:envelope=dict(format=ENVELOPE,directory=str(self.reader.root),header_sha256=self.reader.header_sha256,record=self.writer.append(metric))
            except Exception as exc:error=repr(exc)
        self.coordinated(error);envelope=self.gather(envelope)[0];error=None
        try:
            if value_digest(self.reader.compact(envelope['record']))!=identity:raise ValueError('Each actual peer must read the exact complete original observation before Adam')
        except Exception as exc:error=repr(exc)
        self.coordinated(error,identity);self.pending=envelope

    def preserve_failure(self,exception):
        self.failed=True
        local=dict(rank=self.rank,exception=repr(exception),complete_actual_pre_Adam_metric=self.latest_metric,
            selected_weights_sha256=tensor_digest(self.view.named_selected()),actual_Adam_sha256=value_digest(self.optimizer.state_dict()),
            actual_private_RNG=_cpu(capture_rng(self.device)),successful_optimizer_commit_certified=False)
        observations=self.gather(local);error=receipt=None
        if self.rank==0:
            try:
                if self.writer is None or self.writer.failure_slot is None:raise ValueError('No complete pre-update failure slot was funded; original attempt artifacts remain')
                receipt=self.writer.preserve_failure(dict(attempted_step=self.completed+1,all_actual_rank_observations=observations,
                    original_pending_pre_Adam_receipt=self.pending,successful_optimizer_commit_certified=False))
            except Exception as exc:error=repr(exc)
        self.coordinated(error);self.last_failure=self.gather(receipt)[0]

    def step(self,compute):
        error=None
        try:
            self.intact()
            if not callable(compute) or self.reader is None or self.completed==len(self.entries) or self.pending is not None:
                raise ValueError('One original remaining source computation and fresh allocated attempt are required')
        except Exception as exc:error=repr(exc)
        self.coordinated(error)
        try:
            error=None
            if self.rank==0:
                try:self.writer.reserve_next()
                except Exception as exc:error=repr(exc)
            self.coordinated(error);error=metric=None
            try:metric=compute(self.completed,self.before_adam)
            except Exception as exc:error=repr(exc)
            self.coordinated(error);error=None
            try:
                self.guard.check();names=validate_metric(self.view,self.entries[self.completed],metric)
                if self.pending is None or value_digest(metric)!=value_digest(self.latest_metric):raise ValueError('The actual successful Adam boundary must match its complete independently read pre-Adam metric')
                expected={n:count+int(n in names) for n,count in self.clocks.items()}
                for name,parameter in self.view.named_selected():
                    actual=self.optimizer.state.get(parameter,{})
                    if (not expected[name] and actual or expected[name] and (not actual or float(actual['step'])!=expected[name])):
                        raise ValueError('Every actual sparse Adam clock must advance exactly once for the original active tasks')
            except Exception as exc:error=repr(exc)
            self.coordinated(error,value_digest(metric));error=None
            if self.rank==0:
                try:self.writer.acknowledge_success(self.completed+1)
                except Exception as exc:error=repr(exc)
            self.coordinated(error);self.receipts.append(self.pending);self.completed+=1;self.clocks=expected
            self.pending=self.latest_metric=None
            return metric
        except Exception as exc:
            self.preserve_failure(exc)
            raise

    def commit(self,directory,*,maximum_bytes,minimum_free_bytes):
        error=None
        try:self.intact()
        except Exception as exc:error=repr(exc)
        self.coordinated(error)
        return durable.commit(self.view,self.optimizer,self.guard.fixed,directory,sources=self.sources,entries=self.entries,completed=self.completed,
            history_receipts=self.receipts,history_configuration=self.history,history_guard=self.history_guard,
            maximum_bytes=maximum_bytes,minimum_free_bytes=minimum_free_bytes,rank=self.rank,world=self.world,device=self.device,control_group=self.control_group)

    def restore(self,directory,*,allow_completed=False,expected_pointer_sha256=None):
        completed,receipts,receipt=durable.restore_latest(self.view,self.optimizer,self.guard.fixed,directory,sources=self.sources,entries=self.entries,
            history_configuration=self.history,history_guard=self.history_guard,rank=self.rank,world=self.world,device=self.device,
            control_group=self.control_group,allow_completed=allow_completed,expected_pointer_sha256=expected_pointer_sha256)
        self.guard=JointReplicaGuardV1(self.view,self.optimizer,control_group=self.control_group)
        self.completed,self.receipts=completed,receipts
        self.clocks=scan(self.view,self.entries,completed,receipts,self.history,self.history_guard)['actual_per_parameter_Adam_steps']
        self.writer=self.reader=self.pending=self.latest_metric=None;self.failed=False
        return receipt
