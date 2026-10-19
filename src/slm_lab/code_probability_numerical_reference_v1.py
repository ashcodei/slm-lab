"""Compare original FP32 routes with FP64 recomputation of the same equations.

This disposable copy uses the student's own unchanged weights. FP64 RMS norm
and rotary replace their deliberate FP32 accumulation for the reference only.
No reference weights or predictions enter training or an exported student.
"""
import copy
from types import MethodType

import torch

from . import model as model_module
from .own_dense_actor_v1 import OwnDenseActor


def norm64(self, hidden):
    if hidden.dtype != torch.float64:
        raise ValueError('Independent norm reference must stay FP64')
    return hidden*torch.rsqrt(hidden.square().mean(-1,keepdim=True)+self.eps)*self.weight


def rotary64(hidden, theta, offset=0):
    if hidden.dtype != torch.float64:
        raise ValueError('Independent rotary reference must stay FP64')
    dim = hidden.shape[-1]
    frequency = theta**(-torch.arange(0,dim,2,device=hidden.device,dtype=torch.float64)/dim)
    angles = torch.arange(offset,offset+hidden.shape[-2],device=hidden.device,dtype=torch.float64)[:,None]*frequency[None,:]
    cosine,sine = angles.cos(),angles.sin()
    a,b = hidden[...,0::2],hidden[...,1::2]
    return torch.stack((a*cosine-b*sine,a*sine+b*cosine),dim=-1).flatten(-2)


@torch.no_grad()
def cached_actions(model, prefix, actions):
    device = next(model.parameters()).device
    state = model.policy.prefill(torch.tensor([prefix],device=device))
    values = []
    for i,action in enumerate(actions):
        values.append(state['logits'][0,-1].log_softmax(-1)[action].cpu())
        if i+1 < len(actions):
            state = model.policy.decode(torch.tensor([[action]],device=device),state['cache'])
    return torch.stack(values)


@torch.no_grad()
def compare(model, observation):
    if (type(model) is not OwnDenseActor
            or any(p.dtype != torch.float32 for p in model.parameters())):
        raise ValueError('Exact original FP32 counted own policy required')
    prefix,actions = observation['complete_prompt_ids'],observation['response_ids']
    if not prefix or not actions or len(prefix)+len(actions)>model.config.max_text_length:
        raise ValueError('Complete original prompt and actual bounded action sequence required')
    device = next(model.parameters()).device
    original_signature = model.policy.signature()
    was_training = model.training
    model.eval()
    cached32 = cached_actions(model,prefix,actions)
    recorded32 = torch.tensor(observation['cached_behavior_logps'])
    full32 = torch.tensor(observation['full_behavior_logps'])
    if cached32.shape != recorded32.shape or full32.shape != recorded32.shape:
        raise ValueError('Every original action needs both recorded FP32 probabilities')
    oracle = copy.deepcopy(model).double().eval().requires_grad_(False)
    if oracle.counts() != model.counts():
        raise ValueError('FP64 reference changes precision, not unique model values')
    for module in oracle.modules():
        if type(module) is model_module.RMSNorm:
            module.forward = MethodType(norm64,module)
    original_rotary = model_module.rotary
    try:
        model_module.rotary = rotary64
        ids = torch.tensor([prefix+actions],device=device)
        logits = oracle(ids)['logits'][0,len(prefix)-1:-1]
        if logits.dtype != torch.float64:
            raise ValueError('The reference policy logits must actually be FP64')
        target = torch.tensor(actions,device=device)
        full64 = logits.log_softmax(-1).gather(1,target[:,None]).squeeze(1).cpu()
        del logits
        cached64 = cached_actions(oracle,prefix,actions)
    finally:
        model_module.rotary = original_rotary
        del oracle
        model.train(was_training)
    if original_signature != model.policy.signature():
        raise ValueError('Original FP32 student parameters must remain unchanged')
    return dict(actual_counts=model.counts(),complete_prompt_tokens=len(prefix),actual_action_tokens=len(actions),
        actual_recorded_FP32_cached_action_logps=recorded32.tolist(),actual_replayed_FP32_cached_action_logps=cached32.tolist(),
        actual_recorded_FP32_full_action_logps=full32.tolist(),FP64_full_action_logps=full64.tolist(),
        FP64_cached_action_logps=cached64.tolist(),
        recorded_cached_FP32_replay_exact=torch.equal(cached32,recorded32),
        recorded_cached_FP32_replay_max_absolute_difference=float((cached32-recorded32).abs().max()),
        FP32_full_vs_cached_max_absolute_difference=float((full32-recorded32).abs().max()),
        FP64_full_vs_cached_max_absolute_difference=float((full64-cached64).abs().max()),
        FP32_full_vs_FP64_full_max_absolute_difference=float((full32.double()-full64).abs().max()),
        FP32_cached_vs_FP64_cached_max_absolute_difference=float((recorded32.double()-cached64).abs().max()),
        FP64_reference_uses_student_own_unchanged_weight_values=True,
        FP64_norm_and_rotary_equations_not_original_FP32_rounding=True,
        disposable_extra_precision_copy_not_in_export=True,external_teacher=False,training_updates=0,goal_achieved=False)
