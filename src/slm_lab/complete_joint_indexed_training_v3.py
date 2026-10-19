"""All original joint sources with checked, actual-length history recovery.

Keep the fullV2source/model/initializer/5362step gates and unchanged native
losses. Select the equivalent V3controller's bounded header/frame reads.
Full-size owned GPU capacity, training/export and final quality are required.
"""
from .complete_joint_indexed_training_v2 import CompleteJointIndexedTrainingV2
from .joint_source_checked_training_stage_v3 import JointSourceCheckedStageV3


class CompleteJointIndexedTrainingV3(CompleteJointIndexedTrainingV2,JointSourceCheckedStageV3):
    pass
