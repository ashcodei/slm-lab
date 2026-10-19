"""Controlled original-target/RL methods with admitted bounded train probes."""
from .bounded_code_reward_catalog_v1 import BoundedPublicCodeReward
from .execution_group_causal_v3 import CausalExecutionGroup


class BoundedExecutionGroup(CausalExecutionGroup):
    def __init__(self,model,optimizer,root,**kwargs):
        super().__init__(model,optimizer,root,**kwargs);self.reward=BoundedPublicCodeReward(root)

    def settings(self):
        return dict(super().settings(),reward_version='bounded-train-reference-observations-v1',exact_reward_provenance=self.reward.provenance)
