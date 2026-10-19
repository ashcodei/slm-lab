"""Preference learning and PPO primitives, shared by text and image policies."""
import torch
from torch.nn import functional as F


def response_logps(logits, ids, labels):
    target=ids[:,1:]
    mask=labels[:,1:]!=-100
    if not torch.all(mask.sum(-1)>0):
        raise ValueError('Every preference sequence needs response tokens.')
    # Normalize only response positions. Long shared prompts do not need large
    # FP32 vocabulary softmax tensors for DPO or reference scoring.
    active=-F.cross_entropy(logits[:,:-1][mask].float(),target[mask],reduction='none')
    return torch.zeros_like(mask,dtype=active.dtype).masked_scatter(mask,active).sum(-1)


def dpo_loss(chosen,rejected,ref_chosen,ref_rejected,beta=0.1):
    if beta<=0:raise ValueError('DPO beta must be positive.')
    margin=(chosen-rejected)-(ref_chosen-ref_rejected)
    return -F.logsigmoid(beta*margin).mean()


def reward_loss(chosen,rejected):
    return -F.logsigmoid(chosen-rejected).mean()


def generalized_advantages(rewards,values,gamma=1.0,lam=0.95):
    if rewards.ndim!=1 or rewards.shape!=values.shape or not rewards.numel():
        raise ValueError('GAE expects one nonempty, completed trajectory.')
    if not 0<=gamma<=1 or not 0<=lam<=1:raise ValueError('Invalid GAE discounts.')
    advantages=torch.zeros_like(rewards)
    following_value=torch.zeros((),device=values.device,dtype=values.dtype)
    following_advantage=torch.zeros_like(following_value)
    for t in range(len(rewards)-1,-1,-1):
        delta=rewards[t]+gamma*following_value-values[t]
        following_advantage=delta+gamma*lam*following_advantage
        advantages[t]=following_advantage
        following_value=values[t]
    return advantages,advantages+values


def ppo_loss(logps,old_logps,advantages,values,old_values,returns,clip=0.2,value_coef=0.5):
    if not 0<clip<1 or value_coef<0:raise ValueError('Invalid PPO clipping or value coefficient.')
    if not all(x.shape==logps.shape for x in [old_logps,advantages,values,old_values,returns]):
        raise ValueError('PPO trajectory tensors must have equal shapes.')
    ratio=(logps-old_logps).exp()
    surrogate=torch.minimum(ratio*advantages,ratio.clamp(1-clip,1+clip)*advantages)
    clipped_values=old_values+(values-old_values).clamp(-clip,clip)
    critic=torch.maximum((values-returns).square(),(clipped_values-returns).square()).mean()/2
    return -surrogate.mean()+value_coef*critic


def normalize_rollout_advantages(rollouts):
    """Whiten over the rollout batch, retaining differences between response rewards."""
    values=torch.cat([r['advantages'] for r in rollouts]);mean=values.mean();std=values.std(unbiased=False)
    for rollout in rollouts:rollout['advantages']=(rollout['advantages']-mean)/(std+1e-8)


def contrastive_image_text_loss(images,texts,caption_ids,temperature=.07):
    """Symmetric multi-positive InfoNCE; repeated captions are positives, not negatives."""
    if images.ndim!=2 or images.shape!=texts.shape or caption_ids.shape!=(images.shape[0],):
        raise ValueError('Contrastive features and caption groups must have matching batches.')
    if images.shape[0]<2 or temperature<=0:raise ValueError('Contrastive training needs a batch and positive temperature.')
    # Similarities/softmax stay in FP32 even inside a BF16 training scope.
    with torch.autocast(device_type=images.device.type,enabled=False):
        images=F.normalize(images.float(),dim=-1);texts=F.normalize(texts.float(),dim=-1)
        scores=images@texts.T/temperature;positive=caption_ids[:,None]==caption_ids[None,:]
        i2t=-torch.logsumexp(scores.log_softmax(-1).masked_fill(~positive,float('-inf')),dim=-1).mean()
        t2i=-torch.logsumexp(scores.T.log_softmax(-1).masked_fill(~positive.T,float('-inf')),dim=-1).mean()
        return (i2t+t2i)/2
