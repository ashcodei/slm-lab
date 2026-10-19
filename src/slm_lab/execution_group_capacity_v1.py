"""Full original-prompt/fixed-horizon capacity with no optimizer commit.

The actual controlled kernel must reach its finite-gradient/clipping boundary.
An intercepted optimizer commit ends the disposable stress check. Every actor,
base, Adam moment, RNG and original source cursor is checked unchanged.
"""
import copy
import types

import torch

from .codenet_checkpoint_v1 import capture_rng
from .execution_group_controlled_v2 import ControlledExecutionGroup
from .whole_human_sft_evidence_v1 import tensor_digest


class GradientBoundaryReached(Exception):pass


def equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal(v,b[k]) for k,v in a.items())
    if isinstance(a,(list,tuple)):return type(a) is type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def maximum_prompt(stage):
    info=[]
    for i in range(len(stage.pack)):
        seq,labels=stage.pack.sequence(i);start=next(j for j,y in enumerate(labels) if y!=-100)
        info.append((start,-i))
    length,negative=max(info)
    return dict(original_train_position=-negative,complete_prompt_tokens=length,source_population=len(info),
                maximum_prompt_and_full_horizon_fit=length+stage.max_new_tokens<=stage.model.config.max_text_length)


def stress(stage,position,*,fixed_action_token=261):
    if (type(stage) is not ControlledExecutionGroup or type(position) is not int or not 0<=position<len(stage.pack)
            or type(fixed_action_token) is not int or fixed_action_token==2
            or not 0<=fixed_action_token<stage.model.config.vocab_size or stage.completed>=len(stage.order)):
        raise ValueError('Exact controlled kernel, original train prompt and fixed non-EOS action required')
    device=next(stage.model.parameters()).device
    before=dict(weights=tensor_digest(stage.model.named_parameters()),metadata=stage.signature(stage.model.parameters()),
                optimizer=copy.deepcopy(stage.optimizer.state_dict()),RNG=capture_rng(device),
                order=list(stage.order),completed=stage.completed,metrics=copy.deepcopy(stage.metrics))
    original_sample,original_step=stage.sample,stage.optimizer.step;trace=[];boundary={}
    def fixed_sample(self,prefix,source_id):
        rows=[];self.model.eval()
        with torch.no_grad(),torch.autocast(device_type=device.type,enabled=False):
            for _ in range(self.group_size):
                state=self.model.policy.prefill(torch.tensor([prefix],device=device));old=[];decodes=0
                for i in range(self.max_new_tokens):
                    old.append(float(state['logits'][0,-1].float().log_softmax(-1)[fixed_action_token]))
                    if i+1<self.max_new_tokens:
                        state=self.model.policy.decode(torch.tensor([[fixed_action_token]],device=device),state['cache']);decodes+=1
                del state
                actions=[fixed_action_token]*self.max_new_tokens;response=self.tokenizer.decode(actions)
                reward=self.reward.score(source_id,response)
                rows.append(dict(response_ids=actions,actual_cached_behavior_logps=old,own_response=response,
                    actual_execution_reward=reward,temperature=None,full_vocabulary_sampling=False,
                    synthetic_fixed_non_EOS_capacity_actions=True))
                trace.append(dict(complete_original_prompt_tokens=len(prefix),actual_prefills=1,actual_decodes=decodes,
                    actual_fixed_actions=len(actions),actual_execution_reward=reward['reward'],source_id=source_id))
        return rows
    def no_commit(*args,**kwargs):
        boundary.update(finite_actor_gradients=all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in stage.parameters),
            fixed_parameters_have_no_gradients=all(p.grad is None for p in stage.fixed),
            actual_actor_gradient_tensors=len(stage.parameters))
        raise GradientBoundaryReached('Actual configured backward and clipping complete; optimizer commit prohibited')
    try:
        stage.sample=types.MethodType(fixed_sample,stage);stage.optimizer.step=no_commit
        stage.order[stage.completed]=position
        try:stage.update()
        except GradientBoundaryReached:pass
        else:raise ValueError('Stress must stop exactly before its first optimizer commit')
    finally:
        stage.sample=original_sample;stage.optimizer.step=original_step;stage.order=before['order']
    if (not boundary.get('finite_actor_gradients') or not boundary.get('fixed_parameters_have_no_gradients')
            or tensor_digest(stage.model.named_parameters())!=before['weights']
            or stage.signature(stage.model.parameters())!=before['metadata']
            or not equal(stage.optimizer.state_dict(),before['optimizer']) or not equal(capture_rng(device),before['RNG'])
            or stage.completed!=before['completed'] or stage.metrics!=before['metrics']
            or len(trace)!=stage.group_size or any(t['actual_fixed_actions']!=stage.max_new_tokens
                or t['actual_decodes']!=stage.max_new_tokens-1 for t in trace)):
        raise ValueError('Actual full stress must leave all weights/metadata/Adam/private RNG/source state unchanged')
    return dict(original_train_position=position,local_trace=trace,whole_global_group_size=stage.group_size*stage.world,
        complete_global_fixed_actions=stage.group_size*stage.world*stage.max_new_tokens,
        fixed_action_token=fixed_action_token,configured_full_horizon=stage.max_new_tokens,**boundary,
        unchanged_weights_metadata_Adam_private_RNG_and_source_cursor=True,no_optimizer_commit=True,
        synthetic_capacity_not_sampled_behavior_or_quality=True,production_updates=0,goal_achieved=False)
