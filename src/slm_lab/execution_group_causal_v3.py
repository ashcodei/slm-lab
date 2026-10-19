"""The controlled learner with output-dependent, whole-stream execution rewards."""
from .causal_code_reward_catalog_v1 import CausalPublicCodeReward
from .execution_group_controlled_v2 import ControlledExecutionGroup


class CausalExecutionGroup(ControlledExecutionGroup):
    def __init__(self,model,optimizer,root,**kwargs):
        super().__init__(model,optimizer,root,**kwargs)
        self.reward=CausalPublicCodeReward(root)

    def settings(self):
        return dict(super().settings(),reward_version='causal-complete-published-stream-v2',
            exact_reward_provenance=self.reward.provenance)
