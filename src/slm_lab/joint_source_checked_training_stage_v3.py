"""Same complete joint controller with actual-length history reads."""
import fcntl
from pathlib import Path

from . import joint_indexed_durable_state_v3 as durable
from .joint_source_checked_training_stage_v2 import JointSourceCheckedStageV2
from .joint_indexed_history_v2 import Writer,Reader,read_regular
from .indexed_native_history_v1 import canonical,exclusive
from .corpus import file_hash
from .joint_objective_step_v1 import JointReplicaGuardV1
from .parallel_function_repair_stage_v1 import value_digest


class JointSourceCheckedStageV3(JointSourceCheckedStageV2):
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

    def commit(self,directory,*,maximum_bytes,minimum_free_bytes):
        error=None
        try:self.intact()
        except Exception as exc:error=repr(exc)
        self.coordinated(error)
        return durable.commit(self.view,self.optimizer,self.guard.fixed,directory,sources=self.sources,entries=self.entries,completed=self.completed,
            history_receipts=self.receipts,history_configuration=self.history,source_files=self.source_files,resource_guard=self.resource_guard,
            maximum_bytes=maximum_bytes,minimum_free_bytes=minimum_free_bytes,rank=self.rank,world=self.world,device=self.device,control_group=self.control_group)

    def restore(self,directory,*,allow_completed=False,expected_pointer_sha256=None):
        completed,receipts,receipt=durable.restore_latest(self.view,self.optimizer,self.guard.fixed,directory,sources=self.sources,entries=self.entries,
            history_configuration=self.history,source_files=self.source_files,resource_guard=self.resource_guard,rank=self.rank,world=self.world,device=self.device,
            control_group=self.control_group,allow_completed=allow_completed,expected_pointer_sha256=expected_pointer_sha256)
        self.guard=JointReplicaGuardV1(self.view,self.optimizer,control_group=self.control_group)
        self.completed,self.receipts=completed,receipts;self.clocks={n:0 for n,_ in self.view.named_selected()}
        # Complete metric/source validation already finished before restore.
        # Rebuild just the declared sparse clocks, without rereading records.
        for entry in self.entries[:completed]:
            names={n for objective in entry['objectives'] for n in self.view.task_names(objective['task'])}
            for name in names:self.clocks[name]+=1
        self.writer=self.reader=self.pending=self.latest_metric=None;self.failed=False
        return receipt
