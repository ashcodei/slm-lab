"""Complete original human-ranked DPO with the same frozen own dense base.

These responses were produced by Anthropic models and ranked by people. This
is human preference learning, not fresh human judgments on our own outputs.
The frozen reference shares the one dense backbone already counted in policy
inference; it is not another retained model or an external teacher.
"""
from contextlib import nullcontext
import math

import torch
from torch.nn import functional as F

from .own_dense_actor_v1 import OwnDenseActor
from .whole_human_data_v2 import WholeHumanData
from .whole_human_reward_v1 import source_fingerprint


def sequence_logp(logits,labels):
    target=labels[:,1:];mask=target!=-100
    if not bool(mask.any()):raise ValueError('A complete original response and EOS mask is required')
    return -F.cross_entropy(logits[:,:-1].float()[mask],target[mask],reduction='sum'),int(mask.sum())


def pair_information(data,position):
    row=data.rows[position];chosen,cl=data.sequence(position,'chosen');rejected,rl=data.sequence(position,'rejected')
    starts=[row['native'][f]['target_start'] for f in ('chosen','rejected')]
    if (row['feedback_source']!='human' or row['split']!='train' or row['original_split']!='train'
            or starts[0]!=starts[1] or chosen[:starts[0]]!=rejected[:starts[1]]
            or chosen[-1]!=2 or rejected[-1]!=2 or not row['whole_original_response_whitespace_preserved']):
        raise ValueError('Original human-ranked train pair, identical whole prompt and complete unchanged responses required')
    return dict(original_train_position=position,id=row['id'],source_record_sha256=row['source_record_sha256'],
        whole_token_source_fingerprint=source_fingerprint(data,position),feedback_source='human',
        response_origin=row['response_origin'],complete_context_tokens=len(chosen)+len(rejected),
        chosen_response_EOS_targets=sum(v!=-100 for v in cl[1:]),rejected_response_EOS_targets=sum(v!=-100 for v in rl[1:]))


def preference_gradients(model,data,positions,*,beta=.1,chosen_CE_weight=.1,device='cpu',amp=None):
    if (type(model) is not OwnDenseActor or type(data) is not WholeHumanData or data.family!='preferences'
            or (data.role,data.purpose)!=('train','optimization') or not positions
            or any(type(p) is not int or not 0<=p<len(data) for p in positions)
            or not math.isfinite(beta) or not 0<beta<=1 or not math.isfinite(chosen_CE_weight) or not 0<=chosen_CE_weight<=1):
        raise ValueError('Exact dense actor and admitted original train-only human rankings required')
    information=[pair_information(data,p) for p in positions]
    chosen_targets=sum(r['chosen_response_EOS_targets'] for r in information)
    parameters=[p for _,p in model.named_selected(('actor',))];amp=amp or nullcontext;records=[]
    model.train();model.policy.gradient_checkpointing=True
    for info in information:
        values={};reference={};counts={}
        for field in ('chosen','rejected'):
            batch=data.batch([info['original_train_position']],field,device)
            inputs={k:v for k,v in batch.items() if k!='labels'}
            with amp():
                policy=model(**inputs)['logits']
                with torch.no_grad():original=model.base(**inputs)['logits']
            values[field],counts[field]=sequence_logp(policy,batch['labels'])
            reference[field],actual=sequence_logp(original,batch['labels'])
            if actual!=counts[field] or actual!=info[field+'_response_EOS_targets']:
                raise ValueError('The same whole response/EOS mask must reach policy and shared own reference')
        margin=(values['chosen']-values['rejected'])-(reference['chosen']-reference['rejected'])
        loss=F.softplus(-beta*margin);weighted=loss/len(positions)-chosen_CE_weight*values['chosen']/chosen_targets
        if not bool(torch.isfinite(weighted)):raise ValueError('Every original full-pair DPO/CE objective must be finite')
        weighted.backward(inputs=parameters)
        records.append(dict(**info,policy_chosen_logp=float(values['chosen'].detach()),policy_rejected_logp=float(values['rejected'].detach()),
            reference_chosen_logp=float(reference['chosen']),reference_rejected_logp=float(reference['rejected']),
            implicit_preference_margin=float(margin.detach()),pair_DPO=float(loss.detach())))
    return dict(original_source_positions=positions,original_human_pairs=records,DPO_beta=beta,chosen_CE_weight=chosen_CE_weight,
        chosen_response_EOS_targets=chosen_targets,rejected_response_EOS_targets=sum(r['rejected_response_EOS_targets'] for r in information),
        complete_context_tokens=sum(r['complete_context_tokens'] for r in information),
        pair_mean_DPO=math.fsum(r['pair_DPO'] for r in records)/len(records),
        chosen_token_mean_NLL=-math.fsum(r['policy_chosen_logp'] for r in records)/chosen_targets,
        summed_objective=math.fsum(r['pair_DPO'] for r in records)/len(records)-chosen_CE_weight*math.fsum(r['policy_chosen_logp'] for r in records)/chosen_targets,
        original_frozen_own_reference_shares_the_one_counted_dense_backbone=True,
        human_ranked_other_model_responses_not_fresh_own_output_judgments=True,
        policy_critic_not_trained_by_this_DPO_objective=True)
