"""Complete original joint sources with source-checked recovery and complete corrective target accounting.

All neural/source updates remain the indexedV1path. Only source-only record
validation is batched; complete original source bytes are checked before and
after, with final validation before any restore mutation. No GPU admission.
"""
from dataclasses import asdict

from .codenet_checkpoint_v1 import FileBindings
from .complete_joint_training_data_v1 import CompleteJointTrainingDataV1
from .complete_joint_objective_bridge_v3 import complete_entries,train_step
from .joint_source_checked_training_stage_v3 import JointSourceCheckedStageV3
from .shared_joint_checkpoint_v2 import agree,control


class CompleteJointIndexedTrainingV4(JointSourceCheckedStageV3):
    def __init__(self,source,view,reference,optimizer,sources,options,*,history_root,history_budgets,
                 initializer_guard,maximum_attempts=8,control_group=None,gradient_group=None):
        _,world=control(control_group);error=entries=files=None
        try:
            if (type(source) is not CompleteJointTrainingDataV1 or len(source)!=5362 or world!=4
                    or asdict(source.config)!=asdict(view.actor.config) or view.actor.base.heads['word'].expansion!=5
                    or sources['source_admission_sha256']!=source.identity_sha256 or sources['tokenizer_sha256']!=source.tokenizer_sha256
                    or history_budgets['minimum_free_bytes']<40*1024**3 or type(initializer_guard) is not FileBindings):
                raise ValueError('All actual complete joint sources/model/geometry,four replicas,exact initializer files and40GiBreserve are required')
            initializer_guard.check();entries=complete_entries(source);files=dict(source.files)
            for path,digest in initializer_guard.files.items():
                if str(path) in files and files[str(path)]!=digest:raise ValueError('Original data and initializer bindings cannot disagree')
                files[str(path)]=digest
        except Exception as exc:error=repr(exc)
        agree(error,None,control_group,world);self.source,self.initializer_guard=source,initializer_guard
        def source_guard():
            initializer_guard.check();source.guard.check()
        super().__init__(view,reference,optimizer,entries,sources,options,history_root=history_root,history_budgets=history_budgets,
            source_guard=source_guard,source_files=files,maximum_attempts=maximum_attempts,control_group=control_group,gradient_group=gradient_group)

    def train_next(self):
        def compute(index,before_optimizer):
            _,metric=train_step(self.source,index,self.view,self.reference,self.optimizer,self.guard,options=self.options,
                control_group=self.control_group,gradient_group=self.gradient_group,before_optimizer=before_optimizer)
            return metric
        return self.step(compute)
