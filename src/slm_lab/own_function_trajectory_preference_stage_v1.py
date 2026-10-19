"""Whole original function CE plus own-failure trajectory preference gradients.

Chosen original implementations retain their true EOS. Failed model actions
retain the real stopping/censoring boundary, with no manufactured rejection
EOS. This experimental finite-horizon preference objective uses our frozen
starting actor/reference and no external teacher or human preference labels.
"""
from contextlib import nullcontext
import hashlib
import json
import math

import torch
from torch.nn import functional as F

from .dense_human_preference_v1 import sequence_logp
from .frozen_own_actor_reference_v1 import FrozenOwnActorReference
from .own_dense_actor_v1 import OwnDenseActor
from .own_function_trajectory_preferences_v1 import OwnFunctionTrajectoryPreferences,validate

FORMAT='own-complete-original-code-CE-and-real-failed-trajectory-preference-v1'
METHODS=('chosen_CE','trajectory_DPO')


def information(pair,position):
    validate(pair)
    return dict(original_train_trajectory_position=position,id=pair['id'],original_source_group=pair['original_source_group'],
        native_pair_sha256=hashlib.sha256(json.dumps(pair,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        original_prompt_tokens=len(pair['original_native_prompt_ids']),chosen_response_EOS_targets=len(pair['chosen_original_complete_response_EOS_ids']),
        rejected_actual_action_targets=len(pair['rejected_actual_sampled_action_ids']),
        rejected_is_budget_censored=pair['rejected_is_budget_censored'],rejected_actual_stopped_on_EOS=pair['rejected_actual_stopped_on_EOS'])


def gradients(model,data,positions,reference,*,method,beta=.1,chosen_CE_weight=1.,device='cpu',amp=None):
    if (type(model) is not OwnDenseActor or type(data) is not OwnFunctionTrajectoryPreferences
            or data.role!='train' or data.purpose!='optimization' or type(reference) is not FrozenOwnActorReference
            or method not in METHODS or not positions or len(set(positions))!=len(positions)
            or any(type(p) is not int or p not in data.optimization_positions for p in positions)
            or not math.isfinite(beta) or not 0<beta<=1 or not math.isfinite(chosen_CE_weight) or not 0<chosen_CE_weight<=1):
        raise ValueError('Only exact admitted optimization-source-train trajectories and own actor/reference may provide gradients')
    reference.assert_bound(model);info=[information(data.rows[p],p) for p in positions]
    denominator=sum(r['chosen_response_EOS_targets'] for r in info);parameters=[p for _,p in model.named_selected(('actor',))]
    amp=amp or nullcontext;records=[];model.train();model.policy.gradient_checkpointing=True
    for row in info:
        scores={};originals={}
        for field in ('chosen','rejected') if method=='trajectory_DPO' else ('chosen',):
            batch=data.batch(row['original_train_trajectory_position'],field,device,for_optimization=True)
            inputs={k:v for k,v in batch.items() if k!='labels'}
            with amp():
                logits=model(**inputs)['logits']
                before=reference(**inputs)['logits'] if method=='trajectory_DPO' else None
            scores[field],count=sequence_logp(logits,batch['labels'])
            expected=row['chosen_response_EOS_targets' if field=='chosen' else 'rejected_actual_action_targets']
            if count!=expected:raise ValueError('Every complete positive target and only every actual sampled negative action must reach the objective')
            if before is not None:
                originals[field],actual=sequence_logp(before,batch['labels'])
                if actual!=count:raise ValueError('The frozen own reference must score the exact same positive/censored action masks')
        margin=preference=None;loss=-chosen_CE_weight*scores['chosen']/denominator
        if method=='trajectory_DPO':
            margin=(scores['chosen']-scores['rejected'])-(originals['chosen']-originals['rejected'])
            preference=F.softplus(-beta*margin);loss=loss+preference/len(positions)
        if not bool(torch.isfinite(loss)):raise ValueError('Every complete-source actual-trajectory objective must be finite')
        loss.backward(inputs=parameters)
        records.append(dict(**row,policy_chosen_logp=float(scores['chosen'].detach()),
            policy_rejected_logp=None if method=='chosen_CE' else float(scores['rejected'].detach()),
            reference_chosen_logp=None if method=='chosen_CE' else float(originals['chosen']),
            reference_rejected_logp=None if method=='chosen_CE' else float(originals['rejected']),
            preference_margin=None if margin is None else float(margin.detach()),
            pair_preference=0. if preference is None else float(preference.detach())))
    reference.assert_bound(model)
    chosen=-math.fsum(r['policy_chosen_logp'] for r in records)/denominator
    preference=math.fsum(r['pair_preference'] for r in records)/len(records)
    return dict(format=FORMAT,method=method,original_source_positions=list(positions),original_trajectory_pairs=records,
        chosen_response_EOS_targets=denominator,rejected_actual_action_targets=sum(r['rejected_actual_action_targets'] for r in info),
        chosen_token_mean_NLL=chosen,pair_mean_preference=preference,summed_objective=chosen_CE_weight*chosen+preference,
        beta=beta,chosen_CE_weight=chosen_CE_weight,normalization_pairs=len(positions),normalization_chosen_targets=denominator,
        own_starting_actor_reference_queried=method=='trajectory_DPO',no_unobserved_rejected_EOS_or_other_actions=True,
        failed_program_actions_never_positive_CE_targets=True,human_preferences=False,unoptimized_train_controls_not_used=True)
