"""All original admitted sources through the indexed joint training stage.

This connector retains the entire5362step schedule,4096token native
context and224/448px,whole-word140frame geometry. GPU ownership, completed
initializer provenance, funded full-size capacity/recovery and final quality
still belong to an owned production driver; this class grants no admission.
"""
from dataclasses import asdict

from .complete_joint_training_data_v1 import CompleteJointTrainingDataV1
from .complete_joint_objective_bridge_v2 import complete_entries,train_step
from .joint_indexed_training_stage_v1 import JointIndexedStageV1
from .shared_joint_checkpoint_v2 import agree,control


class CompleteJointIndexedTrainingV1(JointIndexedStageV1):
    def __init__(self,source,view,reference,optimizer,sources,options,*,history_root,history_budgets,
                 initializer_guard,maximum_attempts=8,control_group=None,gradient_group=None):
        _,world=control(control_group);error=entries=None
        try:
            if (type(source) is not CompleteJointTrainingDataV1 or len(source)!=5362 or world!=4
                    or asdict(source.config)!=asdict(view.actor.config) or view.actor.base.heads['word'].expansion!=5
                    or sources['source_admission_sha256']!=source.identity_sha256 or sources['tokenizer_sha256']!=source.tokenizer_sha256
                    or history_budgets['minimum_free_bytes']<40*1024**3 or not callable(initializer_guard)):
                raise ValueError('All actual original joint sources,exact model/vocabulary/geometry,initializer guard,four replicas and40GiBreserve are required')
            initializer_guard();entries=complete_entries(source)
        except Exception as exc:error=repr(exc)
        agree(error,None,control_group,world)
        self.source,self.initializer_guard=source,initializer_guard
        def source_guard():
            initializer_guard();source.guard.check()
        super().__init__(view,reference,optimizer,entries,sources,options,history_root=history_root,history_budgets=history_budgets,
            source_guard=source_guard,maximum_attempts=maximum_attempts,control_group=control_group,gradient_group=gradient_group)

    def train_next(self):
        def compute(index,before_optimizer):
            _,metric=train_step(self.source,index,self.view,self.reference,self.optimizer,self.guard,options=self.options,
                control_group=self.control_group,gradient_group=self.gradient_group,before_optimizer=before_optimizer)
            return metric
        return self.step(compute)
