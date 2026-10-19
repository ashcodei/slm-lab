"""Experimental one-prefix forward with separate loss-branch backwards.

The two vector/Jacobian products keep loss gradients separate until parameter
accumulation. Floating-point accumulation can still differ from a two-forward
joint backward, especially under autocast. No production adoption is implied.
"""
import math

import torch

from .own_actor_vision_prefix_reuse_v1 import OwnActorVisionPrefixReuse


class OwnActorVisionPrefixSplitBackward(OwnActorVisionPrefixReuse):
    def objective(self,*args,matching_weight=.1,**kwargs):
        value = super().objective(*args,matching_weight=matching_weight,**kwargs)
        return dict(**value,matching_weight=matching_weight)

    def backward(self,objective):
        weight = objective.get('matching_weight')
        if type(weight) not in (int,float) or not math.isfinite(weight) or weight <= 0:
            raise ValueError('The exact positive matching coefficient must accompany both loss branches')
        selected = [p for _,p in self.named_selected()]; chosen = {id(p) for p in selected}
        if any(p.grad is not None for p in self.parameters()):
            raise ValueError('Every gradient must start clear before the two explicit branch backwards')
        (weight*objective['matching_loss']).backward(inputs=selected,retain_graph=True)
        objective['caption_CE'].backward(inputs=selected)
        if (any(p.grad is not None for p in self.parameters() if id(p) not in chosen)
                or any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in selected)):
            raise ValueError('Only the original vision/pointing leaves may have finite accumulated gradients')
