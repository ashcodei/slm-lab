"""Complete-program reference-anchored DPO for finite machine execution labels.

Caller adds these gradients to ordinary whole-code/rehearsal objectives. This
does not create human feedback, admit data, select a policy or take an update.
"""
import hashlib
import json
import math

import torch


FORMAT='own-complete-program-execution-reference-v1'


def fingerprint(sequence):
    return hashlib.sha256(json.dumps(sequence,separators=(',',':')).encode()).hexdigest()


def validate(examples,training):
    if not examples or len({e['id'] for e in examples})!=len(examples):
        raise ValueError('Distinct nonempty complete execution-feedback pairs required')
    counts=[]
    for e in examples:
        if e['split'] not in ({'train'} if training else {'train','validation'}):
            raise ValueError('Held source roles cannot enter training gradients')
        if (e.get('feedback_source')!='machine_execution_of_original_publisher_tests'
                or e.get('finite_execution_verified') is not True or e.get('human_preferences') is not False):
            raise ValueError('Explicit independently admitted finite machine-execution labels required')
        pair_counts=[];prefix=[];bodies=[]
        for name in ['chosen','rejected']:
            ids,labels=e[name]
            if (not isinstance(ids,list) or not isinstance(labels,list) or len(ids)!=len(labels)
                    or not 2<=len(ids)<=4096 or ids[0]!=1 or ids[-1]!=2):
                raise ValueError('Complete own-BPE question/program/EOS framing required')
            start=next((i for i,v in enumerate(labels) if v!=-100),len(ids))
            if not 1<=start<len(ids) or labels!=[-100]*start+ids[start:]:
                raise ValueError('Complete contiguous original response/EOS mask required')
            pair_counts.append(len(ids)-start);prefix.append(ids[:start]);bodies.append(ids[start:])
        if prefix[0]!=prefix[1] or bodies[0]==bodies[1]:
            raise ValueError('The same complete question and two distinct complete programs required')
        counts.append(pair_counts)
    return counts


def loss(model,sequence,device,amp):
    ids,labels=sequence
    with amp():value=model(input_ids=torch.tensor([ids],device=device),labels=torch.tensor([labels],device=device))['loss']
    if value.ndim!=0 or not bool(torch.isfinite(value)):
        raise ValueError('Finite complete-program mean NLL required')
    return value


def logps(model,examples,device,amp):
    counts=validate(examples,False)
    with torch.no_grad():
        return [dict(chosen=-n[0]*float(loss(model,e['chosen'],device,amp)),
            rejected=-n[1]*float(loss(model,e['rejected'],device,amp))) for e,n in zip(examples,counts)]


def reference_scores(model,examples,*,policy_sha256,device,amp):
    if (not isinstance(policy_sha256,str) or len(policy_sha256)!=64
            or any(c not in '0123456789abcdef' for c in policy_sha256)):
        raise ValueError('Exact frozen own reference policy digest required')
    counts=validate(examples,False);values=logps(model,examples,device,amp)
    records={e['id']:dict(chosen_logp=v['chosen'],rejected_logp=v['rejected'],
        chosen_targets=n[0],rejected_targets=n[1],chosen_fingerprint=fingerprint(e['chosen']),
        rejected_fingerprint=fingerprint(e['rejected'])) for e,n,v in zip(examples,counts,values)}
    return dict(format=FORMAT,policy_sha256=policy_sha256,records=records,
        response_probability='Sum over every original response/EOS token, no prompt positions.',human_preferences=False)


def preference_gradients(model,examples,reference,*,expected_policy_sha256,alpha,beta,device,amp):
    """Exact streamed gradient of alpha*mean(-logsigmoid(beta*DPOmargin)).

    DPOmargin=(sumLogPchosen-sumLogPrejected)-(referenceChosen-referenceRejected).
    Detached analytic coefficients avoid retaining two full4096-token graphs.
    All masks, source roles and reference inputs validate before any gradient.
    """
    counts=validate(examples,True)
    if (any(type(v) not in {int,float} or not math.isfinite(v) for v in [alpha,beta])
            or not 0<=alpha<=1 or not 0<beta<=1
            or not isinstance(expected_policy_sha256,str) or len(expected_policy_sha256)!=64
            or any(c not in '0123456789abcdef' for c in expected_policy_sha256)
            or any(isinstance(m,torch.nn.Dropout) and m.p for m in model.modules())
            or reference.get('format')!=FORMAT or reference.get('policy_sha256')!=expected_policy_sha256
            or reference.get('human_preferences') is not False):
        raise ValueError('Deterministic policy, exact frozen own reference and finite explicit DPO coefficients required')
    refs=[]
    for e,n in zip(examples,counts):
        ref=reference['records'].get(e['id'])
        if (not isinstance(ref,dict) or ref.get('chosen_fingerprint')!=fingerprint(e['chosen'])
                or ref.get('rejected_fingerprint')!=fingerprint(e['rejected'])
                or ref.get('chosen_targets')!=n[0] or ref.get('rejected_targets')!=n[1]
                or any(type(ref.get(k)) not in {float,int} or not math.isfinite(ref[k]) for k in ['chosen_logp','rejected_logp'])):
            raise ValueError('Every original whole question/program/mask and reference score must match')
        refs.append(ref)
    measured=logps(model,examples,device,amp);records=[];total=0.
    for e,n,ref,current in zip(examples,counts,refs,measured):
        margin=(current['chosen']-current['rejected'])-(ref['chosen_logp']-ref['rejected_logp'])
        x=beta*margin
        dpo=max(-x,0.)+math.log1p(math.exp(-abs(x)))
        sigmoid=math.exp(-x)/(1+math.exp(-x)) if x>=0 else 1/(1+math.exp(x))
        coefficient=alpha*beta*sigmoid/len(examples)
        if coefficient:
            (loss(model,e['chosen'],device,amp)*(coefficient*n[0])).backward()
            (loss(model,e['rejected'],device,amp)*(-coefficient*n[1])).backward()
        total+=dpo/len(examples)
        records.append(dict(id=e['id'],reference_anchored_margin=margin,unscaled_DPO=dpo,
            chosen_sum_logp=current['chosen'],rejected_sum_logp=current['rejected'],
            chosen_targets=n[0],rejected_targets=n[1],gradient_coefficient=coefficient))
    return dict(unscaled_mean_DPO=total,weighted_DPO=alpha*total,alpha=alpha,beta=beta,records=records,
        feedback_source='machine_execution_of_original_publisher_tests',human_preferences=False)
