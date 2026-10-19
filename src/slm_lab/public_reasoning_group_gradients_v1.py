"""Whole-target CE warmup and a fresh strict-verifier group policy gradient.

The optional policy term uses actual fresh full-vocabulary own-model samples,
one update only, globally normalized actual action counts and exact own-snapshot
KL. All groups retain complete public-answer CE. This is an experimental
REINFORCE/group-relative variant, not a claim to reproduce published GRPO or
establish any full-size GPU capacity, throughput or model-quality improvement.
"""
import math
from functools import wraps

import torch

from .dense_human_preference_v1 import sequence_logp
from .execution_group_controlled_v2 import advantages
from .frozen_own_actor_reference_v1 import FrozenOwnActorReference,signature
from .own_dense_actor_v1 import OwnDenseActor
from .public_reasoning_train_data_v1 import PublicReasoningTrainData
from .single_update_code_policy_v8 import objective

FORMAT='own-public-reasoning-complete-CE-fresh-verifier-group-gradient-v1'


def clean_failed_gradients(function):
    @wraps(function)
    def run(model,*args,**kwargs):
        try:return function(model,*args,**kwargs)
        except BaseException:
            # A rejected partial kernel can never leave a commit-ready partial
            # gradient. The caller still owns Adam and all update boundaries.
            if type(model) is OwnDenseActor:
                for _,parameter in model.named_selected(('actor',)):parameter.grad=None
            raise
    return run


def selected(model,data):
    if type(model) is not OwnDenseActor or type(data) is not PublicReasoningTrainData or data.role!='train':
        raise ValueError('The exact own actor and independently admitted train-only reasoning data are required')
    data.guard.check()
    parameters=[p for _,p in model.named_selected(('actor',))]
    identities={id(p) for p in parameters}
    fixed=[p for p in model.parameters() if id(p) not in identities]
    if any(p.grad is not None for p in fixed):raise ValueError('Every fixed original parameter must start without gradients')
    return parameters,fixed


@clean_failed_gradients
def warmup_gradients(model,data,positions,*,amp,weight=1.):
    parameters,fixed=selected(model,data)
    if (not isinstance(positions,list) or not positions or len(positions)!=len(set(positions))
            or not math.isfinite(weight) or weight<=0):raise ValueError('Distinct whole original source positions and finite positive CE weight are required')
    info=[data.information(position) for position in positions]
    denominator=sum(row['response_EOS_targets'] for row in info);initial=signature(fixed)
    model.train();model.policy.gradient_checkpointing=True;values=[];device=next(model.parameters()).device
    for row in info:
        batch=data.batch(row['position'],device);labels=batch.pop('labels')
        with amp():logits=model(**batch)['logits']
        logp,count=sequence_logp(logits,labels)
        if count!=row['response_EOS_targets'] or not bool(torch.isfinite(logp)):
            raise ValueError('Every original complete answer and EOS must receive a finite supervised likelihood')
        (-weight*logp/denominator).backward(inputs=parameters)
        values.append(float(-logp.detach()))
    if (signature(fixed)!=initial or any(p.grad is not None for p in fixed)
            or any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in parameters)):
        raise ValueError('Every selected gradient and no original fixed parameter must preserve its contract')
    return dict(format=FORMAT,profile='whole_answer_warmup',complete_sources=info,response_EOS_targets=denominator,
        original_CE_sum=math.fsum(values),original_token_mean_NLL=math.fsum(values)/denominator,
        all_whole_question_answer_and_true_EOS_masks_preserved=True,neural_rollouts=0,
        externally_generated_teacher_or_human_preference=False)


@clean_failed_gradients
def group_gradients(model,data,position,reference,*,profile,group_size=4,KL_weight=.01,anchor_weight=1.,
                    cache_absolute_bound=.001,CPU_mechanics_horizon=None):
    parameters,fixed=selected(model,data)
    if (type(reference) is not FrozenOwnActorReference or profile not in ('anchor_CE','anchor_RL')
            or type(group_size) is not int or not 2<=group_size<=8
            or not math.isfinite(KL_weight) or KL_weight<0 or not math.isfinite(anchor_weight) or anchor_weight<=0
            or not math.isfinite(cache_absolute_bound) or not 0<cache_absolute_bound<=.001):
        raise ValueError('An exact own frozen reference, fresh group and finite controlled objective are required')
    reference.assert_bound(model);info=data.information(position);ids,labels=data.sequence(position)
    prefix=ids[:info['original_prompt_positions']];horizon=info['full_training_action_budget']
    device=next(model.parameters()).device
    if any(p.dtype!=torch.float32 for p in model.parameters()) or device.type=='cuda' and torch.backends.cuda.matmul.allow_tf32:
        raise ValueError('Native sampling and rescore currently require original FP32 with TF32 disabled')
    if CPU_mechanics_horizon is not None:
        if device.type!='cpu' or type(CPU_mechanics_horizon) is not int or not 1<=CPU_mechanics_horizon<=horizon:
            raise ValueError('Explicit short mechanics horizons are CPU-only and are never production capacity evidence')
        horizon=CPU_mechanics_horizon
    if len(prefix)+horizon>model.config.max_text_length:raise ValueError('The entire prompt and declared native action budget must fit')
    initial_fixed=signature(fixed);initial_policy=model.policy.signature();trajectories=[]
    model.eval()
    with torch.no_grad(),torch.autocast(device.type,enabled=False):
        initial=model.policy.prefill(torch.tensor([prefix],device=device))
        for index in range(group_size):
            state=initial;actions=[];behavior=[]
            for step in range(horizon):
                logits=state['logits'][0,-1].float();token=int(torch.multinomial(logits.softmax(-1),1))
                actions.append(token);behavior.append(float(logits.log_softmax(-1)[token]))
                if token==2 or step+1==horizon:break
                state=model.policy.decode(torch.tensor([[token]],device=device),state['cache'])
            reward=data.score(position,actions,horizon=horizon)
            trajectories.append(dict(local_sample=index,response_ids=actions,actual_cached_behavior_logps=behavior,
                actual_verifier=reward,temperature=1.,full_vocabulary_sampling=True))
        del state,initial
    if model.policy.signature()!=initial_policy:raise ValueError('The actual sampling policy changed before its single update')
    rewards=[item['actual_verifier']['reward'] for item in trajectories];relative=advantages(rewards)
    action_count=sum(len(item['response_ids']) for item in trajectories);totals=dict(policy_surrogate=0.,reference_KL=0.)
    model.train();model.policy.gradient_checkpointing=True
    with torch.autocast(device.type,enabled=False):
        for index,item in enumerate(trajectories):
            native=torch.tensor([prefix+item['response_ids']],device=device)
            logits=model(native)['logits'][0,len(prefix)-1:-1]
            with torch.no_grad():original=reference(native)['logits'][0,len(prefix)-1:-1]
            full=logits.detach().float().log_softmax(-1).gather(1,torch.tensor(item['response_ids'],device=device)[:,None]).flatten()
            difference=float((full-torch.tensor(item['actual_cached_behavior_logps'],device=device)).abs().max())
            if difference>cache_absolute_bound:raise ValueError('Actual cached/full behavior differs beyond the explicit probability bound: '+str(difference))
            item.update(actual_full_behavior_logps=full.tolist(),cached_full_max_absolute_difference=difference)
            loss,metric=objective(logits,original,item['response_ids'],item['actual_cached_behavior_logps'],relative[index],action_count,
                policy_scale=int(profile=='anchor_RL'),KL_weight=KL_weight,freshly_sampled_single_update=True)
            loss.backward(inputs=parameters)
            for key,value in metric.items():totals[key]+=value
            del loss,logits,original
        anchor=warmup_gradients(model,data,[position],amp=lambda:torch.autocast(device.type,enabled=False),weight=anchor_weight)
    if (signature(fixed)!=initial_fixed or any(p.grad is not None for p in fixed)
            or any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in parameters)):
        raise ValueError('Every selected gradient and no original fixed value must preserve its contract')
    reference.assert_bound(model)
    return dict(format=FORMAT,profile=profile,complete_source=info,complete_prompt_ids=prefix,trajectories=trajectories,
        group_size=group_size,global_rewards=rewards,global_advantages=relative,actual_sampled_action_tokens=action_count,
        identical_reward_group_has_zero_policy_advantages=len(set(rewards))==1,
        original_reference_used_regardless_of_rewards=True,anchor=anchor,KL_weight=KL_weight,anchor_weight=anchor_weight,
        cache_absolute_bound=cache_absolute_bound,CPU_mechanics_horizon=CPU_mechanics_horizon,
        own_reference_binding=reference.binding(),resident_inventory=reference.resident_inventory(model),**totals,
        native_model_outputs_and_sampled_tokens_unchanged=True,full_size_capacity_and_model_quality_not_established=True,
        human_feedback=False,external_neural_teacher=False)
