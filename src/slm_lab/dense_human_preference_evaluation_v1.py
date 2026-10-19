"""All original held human-ranking pairs, raw policy/reference and length controls.

These are teacher preference metrics on other models' human-ranked answers.
They do not evaluate newly generated own answers or certify alignment quality.
"""
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path

import torch

from .corpus import file_hash
from .dense_human_preference_v1 import sequence_logp
from .diverse_images import atomic_json
from .own_dense_actor_v1 import OwnDenseActor
from .whole_human_data_v2 import WholeHumanData
from .whole_human_reward_v1 import source_fingerprint


def information(data,position):
    row=data.rows[position];values={f:data.sequence(position,f) for f in ('chosen','rejected')}
    starts=[row['native'][f]['target_start'] for f in ('chosen','rejected')]
    if (row['split']!='valid' or row['original_split']!='test' or row['feedback_source']!='human'
            or starts[0]!=starts[1] or values['chosen'][0][:starts[0]]!=values['rejected'][0][:starts[1]]
            or any(ids[-1]!=2 for ids,_ in values.values())):
        raise ValueError('Complete original held human pair and identical original prompt required')
    result=dict(original_held_position=position,id=row['id'],source_record_sha256=row['source_record_sha256'],
        source_fingerprint=source_fingerprint(data,position),complete_context_tokens=sum(len(ids) for ids,_ in values.values()),
        response_origin=row['response_origin'],source_blob=row['source_blob'])
    for field,(ids,labels) in values.items():
        targets=[v for v in labels[1:] if v!=-100]
        result[field+'_response_EOS_targets']=len(targets)
        result[field+'_target_ids_sha256']=hashlib.sha256(json.dumps(targets,separators=(',',':')).encode()).hexdigest()
    return result


def scored(row,beta):
    margin=(row['policy_chosen_logp']-row['policy_rejected_logp'])-(row['reference_chosen_logp']-row['reference_rejected_logp'])
    raw=row['policy_chosen_logp']-row['policy_rejected_logp']
    normalized=row['policy_chosen_logp']/row['chosen_response_EOS_targets']-row['policy_rejected_logp']/row['rejected_response_EOS_targets']
    x=-beta*margin
    def credit(value):return float(value>0)+.5*float(value==0)
    return dict(implicit_preference_margin=margin,pair_DPO=max(x,0)+math.log1p(math.exp(-abs(x))),
        implicit_preference_credit=credit(margin),raw_sequence_preference_credit=credit(raw),
        token_mean_preference_credit=credit(normalized),
        shorter_response_prior_credit=credit(row['rejected_response_EOS_targets']-row['chosen_response_EOS_targets']))


def summary(rows):
    return dict(pairs=len(rows),pair_mean_DPO=math.fsum(r['pair_DPO'] for r in rows)/len(rows),
        implicit_preference_accuracy=math.fsum(r['implicit_preference_credit'] for r in rows)/len(rows),
        implicit_preference_ties=sum(r['implicit_preference_margin']==0 for r in rows),
        raw_sequence_preference_accuracy=math.fsum(r['raw_sequence_preference_credit'] for r in rows)/len(rows),
        token_mean_preference_accuracy=math.fsum(r['token_mean_preference_credit'] for r in rows)/len(rows),
        shorter_response_prior_accuracy=math.fsum(r['shorter_response_prior_credit'] for r in rows)/len(rows),
        chosen_token_mean_NLL=-math.fsum(r['policy_chosen_logp'] for r in rows)/sum(r['chosen_response_EOS_targets'] for r in rows),
        rejected_token_mean_NLL=-math.fsum(r['policy_rejected_logp'] for r in rows)/sum(r['rejected_response_EOS_targets'] for r in rows),
        complete_context_tokens=sum(r['complete_context_tokens'] for r in rows),
        complete_response_EOS_targets=sum(r['chosen_response_EOS_targets']+r['rejected_response_EOS_targets'] for r in rows))


@torch.no_grad()
def evaluate(model,data,path,*,beta=.1,amp=None,checked=lambda:None):
    if (type(model) is not OwnDenseActor or type(data) is not WholeHumanData
            or (data.family,data.role,data.purpose)!=('preferences','valid','evaluation') or not 0<beta<=1):
        raise ValueError('Exact dense actor and complete evaluation-only held human rankings required')
    path=Path(path);journal=path.with_suffix('.jsonl')
    if path.exists() or journal.exists():raise ValueError('Preserve earlier full or partial preference deliveries')
    amp=amp or nullcontext;model.eval();device=next(model.parameters()).device;rows=[]
    with journal.open('x') as stream:
        for p in range(len(data)):
            checked();row=information(data,p)
            for field in ('chosen','rejected'):
                batch=data.batch([p],field,device);inputs={k:v for k,v in batch.items() if k!='labels'}
                with amp():policy=model(**inputs)['logits'];reference=model.base(**inputs)['logits']
                for name,logits in [('policy',policy),('reference',reference)]:
                    value,count=sequence_logp(logits,batch['labels'])
                    if not math.isfinite(float(value)) or value>0 or count!=row[field+'_response_EOS_targets']:
                        raise ValueError('Finite exact full held response/EOS likelihoods required')
                    row[name+'_'+field+'_logp']=float(value)
            row.update(scored(row,beta));rows.append(row);stream.write(json.dumps(row)+'\n');stream.flush()
            if (p+1)%100==0:print('full_held_human_preference',p+1,'/',len(data),flush=True)
    report=dict(DPO_beta=beta,overall=summary(rows),by_source={n:summary([r for r in rows if r['source_blob']==n]) for n in sorted({r['source_blob'] for r in rows})},
        results=rows,every_original_held_pair_once=True,teacher_other_model_responses_not_own_output_human_judgments=True,
        ties_receive_half_credit=True,held_gradients=False,candidate_promoted=False,goal_achieved=False)
    atomic_json(path,report);return report


def readback(data,path):
    if (data.family,data.role,data.purpose)!=('preferences','valid','evaluation'):
        raise ValueError('Original held evaluation-only human sources required')
    path=Path(path);report=json.loads(path.read_text());rows=report['results']
    if (rows!=[json.loads(l) for l in path.with_suffix('.jsonl').open()]
            or [r['original_held_position'] for r in rows]!=list(range(len(data)))):
        raise ValueError('Every original held pair and flushed original delivery exactly once required')
    for p,row in enumerate(rows):
        expected=information(data,p)
        if (any(row[k]!=v for k,v in expected.items())
                or any(not math.isfinite(row[n+'_'+f+'_logp']) or row[n+'_'+f+'_logp']>0
                    for n in ('policy','reference') for f in ('chosen','rejected'))
                or any(row[k]!=v for k,v in scored(row,report['DPO_beta']).items())):
            raise ValueError('Every complete held source/target fingerprint and preference/length score must reconstruct')
    if (report['overall']!=summary(rows) or report['by_source']!={n:summary([r for r in rows if r['source_blob']==n]) for n in sorted({r['source_blob'] for r in rows})}):
        raise ValueError('Every complete source/target preference aggregate must reconstruct')
    return dict(overall=report['overall'],by_source=report['by_source'],report_sha256=file_hash(path),
        every_original_held_pair_target_hash_and_score_reconstructed=True,numerical_neural_likelihoods_not_replayed=True)
