"""Execute whole original joint capacity cases with indexed private recovery.

The complete source workload is independently reconstructed at admission.
Only its local capacity cursor is renumbered; raw task/source observations
and original joint steps remain explicit. This stage owns no GPU resources
and does not admit production training.
"""
import copy
from dataclasses import asdict

from .codenet_checkpoint_v1 import FileBindings
from .complete_joint_capacity_plan_v3 import choose
from .complete_joint_objective_bridge_v3 import train_step
from .complete_joint_training_data_v1 import CompleteJointTrainingDataV1
from .joint_source_checked_training_stage_v3 import JointSourceCheckedStageV3
from .parallel_function_repair_stage_v1 import value_digest
from .shared_joint_checkpoint_v2 import agree, control


class CompleteJointCapacityStageV2(JointSourceCheckedStageV3):
    def __init__(self, source, view, reference, optimizer, sources, options, *, capacity_plan, original_visual_capacity,
                 history_root, history_budgets, initializer_guard, maximum_attempts=3, control_group=None, gradient_group=None):
        _, world = control(control_group); error = identity = None
        try:
            if (type(source) is not CompleteJointTrainingDataV1 or world != 4
                    or asdict(source.config) != asdict(view.actor.config) or view.actor.base.heads['word'].expansion != 5
                    or sources['source_admission_sha256'] != source.identity_sha256
                    or sources['tokenizer_sha256'] != source.tokenizer_sha256
                    or type(initializer_guard) is not FileBindings or history_budgets['minimum_free_bytes'] < 40 * 1024**3
                    or choose(source, original_visual_capacity) != capacity_plan):
                raise ValueError('The actual complete original joint capacity workload,four replicas,own initializer and40GiBreserve must match')
            entries = [copy.deepcopy(r['checkpoint_entry']) for r in capacity_plan['cases']]
            if sources['complete_schedule_sha256'] != value_digest(entries):
                raise ValueError('The capacity checkpoint must bind every original selected complete joint batch')
            files = dict(source.files)
            for path, digest in initializer_guard.files.items():
                path = str(path)
                if path in files and files[path] != digest:
                    raise ValueError('Actual initializer and complete original source bindings disagree')
                files[path] = digest
            initializer_guard.check(full=True)
            identity = value_digest(capacity_plan)
        except Exception as exc:
            error = repr(exc)
        agree(error, identity, control_group, world)
        self.source = source; self.initializer_guard = initializer_guard
        self.capacity_plan = copy.deepcopy(capacity_plan); self.capacity_plan_sha256 = identity

        def source_guard():
            initializer_guard.check(); source.guard.check()
            if value_digest(self.capacity_plan) != self.capacity_plan_sha256:
                raise ValueError('The actual original joint capacity workload cannot change during execution')

        super().__init__(view, reference, optimizer, entries, sources, options, history_root=history_root,
            history_budgets=history_budgets, source_guard=source_guard, source_files=files,
            maximum_attempts=maximum_attempts, control_group=control_group, gradient_group=gradient_group)

    def train_next(self):
        def compute(index, before_optimizer):
            case = self.capacity_plan['cases'][index]

            def local_metric(original):
                return capacity_metric(original,case,self.entries[index],self.capacity_plan_sha256)

            _, original = train_step(self.source, case['original_joint_step'] - 1, self.view, self.reference, self.optimizer,
                self.guard, options=self.options, control_group=self.control_group, gradient_group=self.gradient_group,
                before_optimizer=lambda metric: before_optimizer(local_metric(metric)))
            return local_metric(original)

        return self.step(compute)


def capacity_metric(original,case,entry,capacity_plan_sha256):
    """Keep the bridge's typed digest; JSON file digests are a different protocol."""
    declared_original=dict(copy.deepcopy(entry),step=case['original_joint_step'])
    if (case['original_checkpoint_entry_sha256']!=value_digest(declared_original)
            or original['step']!=case['original_joint_step']
            or original['entry_sha256']!=case['original_checkpoint_entry_sha256']):
        raise ValueError('Actual complete source computation must retain its original joint step and typed entry digest')
    metric=copy.deepcopy(original)
    metric.update(step=entry['step'],entry_sha256=value_digest(entry))
    metric['gradient_interaction']['actual_original_joint_step']=case['original_joint_step']
    metric['gradient_interaction']['actual_original_checkpoint_entry_sha256']=case['original_checkpoint_entry_sha256']
    metric['gradient_interaction']['actual_capacity_plan_sha256']=capacity_plan_sha256
    return metric
