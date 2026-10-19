"""Same joint updates, with complete source checks around checkpoint reads.

Only source-only journal reconstruction is batched: every source byte is
hashed before and after; every original record is still read/reconstructed.
Final source verification precedes any restore mutation or checkpoint write.
Neural updates retain the existing unbatched source guards unchanged.
"""
import copy

from . import joint_indexed_durable_state_v2 as durable
from .joint_indexed_training_stage_v1 import JointIndexedStageV1
from .joint_objective_step_v1 import JointReplicaGuardV1
from .parallel_function_repair_stage_v1 import value_digest


class JointSourceCheckedStageV2(JointIndexedStageV1):
    def __init__(self,*args,source_files,**kwargs):
        self.source_files=copy.deepcopy(source_files);self.source_files_identity=value_digest(source_files)
        super().__init__(*args,**kwargs)

    def resource_guard(self):
        self.reference.assert_bound();self.guard.fixed.check(self.view,self.optimizer)
        if value_digest(self.history)!=self.identity or value_digest(self.source_files)!=self.source_files_identity:
            raise ValueError('The complete original joint source/model/reference/history binding cannot change during record validation')

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
