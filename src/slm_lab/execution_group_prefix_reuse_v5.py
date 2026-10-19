"""Same scalar action draws with one immutable prompt prefill per local group.

The existing decoder appends new K/V tensors without modifying its input cache.
Reuse lasts only for one unchanged-policy group. No new network values or
approximate probabilities are introduced. Hardware speedup must be measured.
"""
import torch

from .execution_group_bounded_v4 import BoundedExecutionGroup


def sample_group(stage,prefix,source_id):
    if (not prefix or type(stage.group_size) is not int or not 2<=stage.group_size<=16
            or type(stage.max_new_tokens) is not int or stage.max_new_tokens<1
            or len(prefix)+stage.max_new_tokens>stage.model.config.max_text_length):raise ValueError('Whole source prefix and complete own-response group must fit')
    device=next(stage.model.parameters()).device;stage.model.eval();rows=[]
    with torch.no_grad(),torch.autocast(device_type=device.type,enabled=False):
        initial=stage.model.policy.prefill(torch.tensor([prefix],device=device));cache=initial['cache']
        def fingerprint():
            tensors=[initial['logits'],*[t for pair in cache.layers for t in pair]]
            return [(t._version,t.data_ptr(),tuple(t.shape),t.dtype,t.device) for t in tensors]
        before=fingerprint()
        for _ in range(stage.group_size):
            state=initial;actions=[];old=[]
            for i in range(stage.max_new_tokens):
                logits=state['logits'][0,-1].float();token=int(torch.multinomial(logits.softmax(-1),1))
                actions.append(token);old.append(float(logits.log_softmax(-1)[token]))
                if token==2 or i+1==stage.max_new_tokens:break
                state=stage.model.policy.decode(torch.tensor([[token]],device=device),state['cache'])
            del state
            response=stage.tokenizer.decode(actions);reward=stage.reward.score(source_id,response)
            if fingerprint()!=before or stage.model.policy.signature()!=cache.parameter_signature:
                raise ValueError('Reused original cache/logits and unchanged policy must remain exact within the entire group')
            rows.append(dict(response_ids=actions,actual_cached_behavior_logps=old,own_response=response,
                actual_execution_reward=reward,temperature=1.,full_vocabulary_sampling=True,
                shared_prefix_group_prefills=1,shared_prefix_group_samples=stage.group_size,
                original_complete_prefix_tokens=len(prefix),original_prefix_KV_bytes=cache.bytes))
    return rows


class PrefixReuseExecutionGroup(BoundedExecutionGroup):
    def sample(self,prefix,source_id):return sample_group(self,prefix,source_id)

    def settings(self):return dict(super().settings(),sampler='one_exact_immutable_prefix_prefill_per_local_scalar_group_v1')
