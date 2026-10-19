"""Matched own-response human PPO and KL-only control, same frozen scorer.

Both profiles compute and retain the actual human-head score. Only the
predeclared scale changes its terminal reward contribution. This separates
human reward optimization from drift caused by the shared KL/value updates.
"""
import math
import torch

from .dense_ppo_cached_v2 import DenseCachedPPOStage,FORMAT as CACHED_FORMAT,validate_cached
from .dense_ppo_v1 import validate_rollouts
from .objectives import generalized_advantages,normalize_rollout_advantages

FORMAT='own-dense-cached-human-reward-PPO-matched-KL-control-v1'


def validate_controlled(rows,scale):
    for row in rows:
        if (row.get('terminal_human_reward_scale')!=scale
                or not math.isfinite(row.get('raw_human_reward_score',float('nan')))
                or row['human_reward_score']!=scale*row['raw_human_reward_score']
                or row.get('reference_route')!='same_counted_frozen_fused_dense_base'):
            raise ValueError('Actual raw human score and the exact declared terminal reward control required')
    validate_cached(rows)


class DenseControlledPPOStage(DenseCachedPPOStage):
    def __init__(self,*args,human_reward_scale,**kwargs):
        if type(human_reward_scale) is not int or human_reward_scale not in (0,1):
            raise ValueError('Explicit matched human reward scale0or1required')
        self.human_reward_scale=human_reward_scale
        super().__init__(*args,**kwargs)

    def collect(self,positions):
        rows=super().collect(positions)
        for row in rows:
            raw=row['human_reward_score'];row['raw_human_reward_score']=raw
            row['terminal_human_reward_scale']=self.human_reward_scale
            row['human_reward_score']=self.human_reward_scale*raw
            row['reference_route']='same_counted_frozen_fused_dense_base'
            rewards=-self.options['KL_coefficient']*(torch.tensor(row['old_logps'])-torch.tensor(row['reference_logps']))
            rewards[-1]+=row['human_reward_score']
            advantages,returns=generalized_advantages(rewards,torch.tensor(row['old_values']),self.options['gamma'],self.options['GAE_lambda'])
            row.update(token_rewards=rewards.tolist(),raw_advantages=advantages.tolist(),returns=returns.tolist(),advantages=advantages)
        normalize_rollout_advantages(rows)
        for row in rows:row['advantages']=row['advantages'].tolist()
        validate_rollouts(self.data,positions,self.options,rows,self.reward.provenance,self.model.config.vocab_size,
            min(self.model.config.max_text_length,self.reward.encoder.config.max_text_length))
        validate_controlled(rows,self.human_reward_scale);return rows

    def state(self):return {**super().state(),'format':FORMAT,'terminal_human_reward_scale':self.human_reward_scale}

    def resume(self,state,*,allow_completed=False):
        if state.get('format')!=FORMAT or state.get('terminal_human_reward_scale')!=self.human_reward_scale:
            raise ValueError('Exact human PPO or matched KL-only control state required before mutation')
        for record in state.get('core',{}).get('metrics',[]):validate_controlled(record['rollouts'],self.human_reward_scale)
        return super().resume({**state,'format':CACHED_FORMAT},allow_completed=allow_completed)
