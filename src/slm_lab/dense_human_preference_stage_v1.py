"""Whole-source human DPO epochs with exact policy-only Adam/RNG recovery."""
import math

import torch
from torch.nn import functional as F

from .codenet_stage_v1 import EpochOrder
from .dense_human_preference_v1 import pair_information,preference_gradients,sequence_logp
from contextlib import nullcontext
from .own_dense_actor_v1 import OwnDenseActor
from .own_dense_actor_checkpoint_v1 import selected,make_state,restore_state
from .whole_human_data_v2 import WholeHumanData
from .whole_human_sft_evidence_v1 import tensor_digest

FORMAT='own-dense-complete-human-DPO-epoch-Adam-RNG-v1'


def signature(parameters):return [(id(p),p._version,p.data_ptr(),p.requires_grad) for p in parameters]


def chosen_gradients(model,data,positions,setting,device='cpu',amp=None):
    """Matched chosen CE control: no rejected or reference neural forwards."""
    info=[pair_information(data,p) for p in positions];total=sum(r['chosen_response_EOS_targets'] for r in info)
    parameters=[p for _,p in model.named_selected(('actor',))];amp=amp or nullcontext;rows=[]
    model.train();model.policy.gradient_checkpointing=True
    for row in info:
        batch=data.batch([row['original_train_position']],'chosen',device)
        with amp():logits=model(**{k:v for k,v in batch.items() if k!='labels'})['logits']
        logp,count=sequence_logp(logits,batch['labels'])
        if count!=row['chosen_response_EOS_targets'] or not bool(torch.isfinite(logp)):
            raise ValueError('Every complete chosen response/EOS target must enter the matched CE control')
        (-setting['chosen_CE_weight']*logp/total).backward(inputs=parameters)
        rows.append(dict(**row,policy_chosen_logp=float(logp.detach()),policy_rejected_logp=None,
            reference_chosen_logp=None,reference_rejected_logp=None,implicit_preference_margin=None,pair_DPO=0.))
    ce=-math.fsum(r['policy_chosen_logp'] for r in rows)/total
    return dict(original_source_positions=positions,original_human_pairs=rows,DPO_beta=setting['DPO_beta'],chosen_CE_weight=setting['chosen_CE_weight'],
        chosen_response_EOS_targets=total,rejected_response_EOS_targets=sum(r['rejected_response_EOS_targets'] for r in info),
        complete_context_tokens=sum(r['complete_context_tokens'] for r in info),pair_mean_DPO=0.,chosen_token_mean_NLL=ce,
        summed_objective=-setting['chosen_CE_weight']*math.fsum(r['policy_chosen_logp'] for r in rows)/total,
        original_frozen_own_reference_shares_the_one_counted_dense_backbone=True,
        human_ranked_other_model_responses_not_fresh_own_output_judgments=True,policy_critic_not_trained_by_this_DPO_objective=True)


def actual_exposures(data,positions,setting):
    chosen_context=sum(data.rows[p]['native']['chosen']['length'] for p in positions)
    rejected_context=sum(data.rows[p]['native']['rejected']['length'] for p in positions)
    chosen=sum(data.rows[p]['native']['chosen']['length']-data.rows[p]['native']['chosen']['target_start'] for p in positions)
    rejected=sum(data.rows[p]['native']['rejected']['length']-data.rows[p]['native']['rejected']['target_start'] for p in positions)
    dpo=setting['DPO_weight']
    return dict(DPO_weight=dpo,reference_queried=bool(dpo),actual_policy_context_tokens=chosen_context+dpo*rejected_context,
        actual_reference_context_tokens=dpo*(chosen_context+rejected_context),actual_policy_chosen_targets=chosen,
        actual_policy_rejected_targets=dpo*rejected,actual_reference_chosen_targets=dpo*chosen,actual_reference_rejected_targets=dpo*rejected)


def validate_metric(data,positions,setting,record,step):
    wanted=[pair_information(data,p) for p in positions];pairs=record.get('original_human_pairs',[])
    if (record.get('step')!=step+1 or record.get('original_source_positions')!=positions or len(pairs)!=len(wanted)
            or record.get('learning_rate')!=setting['learning_rate'] or record.get('DPO_beta')!=setting['DPO_beta']
            or record.get('chosen_CE_weight')!=setting['chosen_CE_weight']
            or not math.isfinite(record.get('gradient_norm_before_clip',float('nan'))) or record['gradient_norm_before_clip']<0
            or record.get('original_frozen_own_reference_shares_the_one_counted_dense_backbone') is not True
            or record.get('human_ranked_other_model_responses_not_fresh_own_output_judgments') is not True
            or record.get('policy_critic_not_trained_by_this_DPO_objective') is not True):
        raise ValueError('Every actual complete human source draw, fixed objective and gradient metric required')
    if any(record.get(k)!=v for k,v in actual_exposures(data,positions,setting).items()):
        raise ValueError('Actual policy/reference chosen/rejected exposures must match the CE or DPO profile')
    for info,row in zip(wanted,pairs):
        if any(row.get(k)!=v for k,v in info.items()):raise ValueError('Every complete original chosen/rejected source identity and target mask must reconstruct')
        keys=['policy_chosen_logp','policy_rejected_logp','reference_chosen_logp','reference_rejected_logp']
        if not math.isfinite(row['policy_chosen_logp']) or row['policy_chosen_logp']>0:
            raise ValueError('Every complete chosen response log probability must be finite and nonpositive')
        if not setting['DPO_weight']:
            if any(row[k] is not None for k in keys[1:]) or row['implicit_preference_margin'] is not None or row['pair_DPO']!=0:
                raise ValueError('CE control must not fabricate rejected/reference scores or DPO margins')
            continue
        if any(not math.isfinite(row[k]) or row[k]>0 for k in keys):raise ValueError('Every complete response log probability must be finite and nonpositive')
        # Reproduce FP32 scalar subtraction instead of subtracting Python
        # doubles. Long whole responses can otherwise fail by cancellation.
        pc,pr,qc,qr=[torch.tensor(row[k],dtype=torch.float32) for k in keys]
        margin=float((pc-pr)-(qc-qr));dpo=float(F.softplus(-setting['DPO_beta']*torch.tensor(margin,dtype=torch.float32)))
        if row['implicit_preference_margin']!=margin or not math.isclose(row['pair_DPO'],dpo,abs_tol=3e-6,rel_tol=3e-6):
            raise ValueError('The actual full-response FP32 own-reference DPO margin and loss must reconstruct')
    for key in ('chosen_response_EOS_targets','rejected_response_EOS_targets','complete_context_tokens'):
        if record.get(key)!=sum(r[key] for r in wanted):raise ValueError('Every original complete human response/EOS and context target must enter the denominator')
    mean=math.fsum(r['pair_DPO'] for r in pairs)/len(pairs)
    ce=-math.fsum(r['policy_chosen_logp'] for r in pairs)/record['chosen_response_EOS_targets']
    if (record.get('pair_mean_DPO')!=mean or record.get('chosen_token_mean_NLL')!=ce
            or record.get('summed_objective')!=mean+setting['chosen_CE_weight']*ce):
        # The kernel uses an equivalent division after coefficient scaling;
        # require its exact declared Python aggregation rather than a new
        # floating-point operation order.
        exact=mean-setting['chosen_CE_weight']*math.fsum(r['policy_chosen_logp'] for r in pairs)/record['chosen_response_EOS_targets']
        if record.get('pair_mean_DPO')!=mean or record.get('chosen_token_mean_NLL')!=ce or record.get('summed_objective')!=exact:
            raise ValueError('The complete pair mean and chosen-token-normalized objective must reconstruct')


class DenseHumanPreferenceStage:
    def __init__(self,model,optimizer,data,options,sources,device='cpu',amp=None):
        keys={'seed','epochs','steps','microbatch','accumulation','learning_rate','betas','weight_decay','gradient_clip','DPO_beta','DPO_weight','chosen_CE_weight'}
        if (type(model) is not OwnDenseActor or type(data) is not WholeHumanData
                or (data.family,data.role,data.purpose)!=('preferences','train','optimization') or set(options)!=keys
                or any(type(options[k]) is not int or options[k]<1 for k in ('seed','epochs','steps','microbatch','accumulation'))
                or options['microbatch']!=1 or options['steps']!=options['epochs']*math.ceil(len(data)/options['accumulation'])
                or any(not math.isfinite(options[k]) or options[k]<=0 for k in ('learning_rate','gradient_clip','DPO_beta'))
                or options['DPO_beta']>1 or type(options['DPO_weight']) is not int or options['DPO_weight'] not in (0,1)
                or not math.isfinite(options['chosen_CE_weight']) or not 0<options['chosen_CE_weight']<=1
                or not math.isfinite(options['weight_decay']) or options['weight_decay']<0
                or len(options['betas'])!=2 or any(not 0<=v<1 for v in options['betas'])):
            raise ValueError('Exact dense policy, original train-only human pairs and complete unrepeated fixed objective/Adam epoch required')
        self.model,self.optimizer,self.data=model,optimizer,data
        self.options,self.sources,self.device,self.amp=dict(options),dict(sources),torch.device(device),amp
        named=selected(model,optimizer,('actor',));self.parameters=[p for _,p in named]
        group=optimizer.param_groups[0]
        if group['lr']!=options['learning_rate'] or tuple(group['betas'])!=tuple(options['betas']) or group['weight_decay']!=options['weight_decay']:
            raise ValueError('The ordinary policy Adam must use the exact fixed declared hyperparameters')
        ids={id(p) for p in self.parameters};self.fixed=[p for p in model.parameters() if id(p) not in ids]
        self.fixed_signature=signature(self.fixed);self.frozen_digest=tensor_digest(model.base.named_parameters())
        self.order=EpochOrder(len(data),options['seed']);self.per_epoch=math.ceil(len(data)/options['accumulation'])
        self.completed=0;self.metrics=[]

    def positions(self,step):
        if type(step) is not int or not 0<=step<self.options['steps']:raise ValueError('Exact bounded whole-source preference update required')
        epoch,within=divmod(step,self.per_epoch);width=self.options['accumulation']
        return [self.order.at(epoch*len(self.data)+p) for p in range(within*width,min((within+1)*width,len(self.data)))]

    def cursor(self,completed):
        epoch,within=divmod(completed,self.per_epoch);offset=min(within*self.options['accumulation'],len(self.data))
        return dict(completed_updates=completed,source_population=len(self.data),complete_passes=epoch,next_pass_sources=offset,
            actual_pair_draws=epoch*len(self.data)+offset,repeated_rounding_pairs=0,seed=self.options['seed'],world_size=1)

    def update(self):
        positions=self.positions(self.completed)
        if signature(self.fixed)!=self.fixed_signature:raise ValueError('Every frozen dense/head and unused separate critic parameter must remain exact')
        self.optimizer.zero_grad(set_to_none=True)
        try:
            if self.options['DPO_weight']:
                record=preference_gradients(self.model,self.data,positions,beta=self.options['DPO_beta'],
                    chosen_CE_weight=self.options['chosen_CE_weight'],device=self.device,amp=self.amp)
            else:record=chosen_gradients(self.model,self.data,positions,self.options,self.device,self.amp)
            record.update(actual_exposures(self.data,positions,self.options))
            if (any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in self.parameters)
                    or any(p.grad is not None for p in self.fixed)):
                raise ValueError('Every actor and no frozen dense/unused critic parameter needs finite gradients')
            norm=float(torch.nn.utils.clip_grad_norm_(self.parameters,self.options['gradient_clip'],error_if_nonfinite=True))
            record.update(step=self.completed+1,learning_rate=self.options['learning_rate'],gradient_norm_before_clip=norm)
            validate_metric(self.data,positions,self.options,record,self.completed)
            self.optimizer.step();self.optimizer.zero_grad(set_to_none=True)
        except BaseException:
            self.optimizer.zero_grad(set_to_none=True);raise
        self.completed+=1;self.metrics.append(record);return record

    def state(self):
        if signature(self.fixed)!=self.fixed_signature:raise ValueError('Every original/fixed/unused tensor metadata must stay exact')
        core=make_state(self.model,self.optimizer,groups=('actor',),sources=self.sources,completed=self.completed,
            planned=self.options['steps'],cursor=self.cursor(self.completed),metrics=self.metrics,device=self.device)
        if core['model']['frozen_dense_sha256']!=self.frozen_digest:raise ValueError('Every frozen dense/auxiliary value must remain exact')
        return dict(format=FORMAT,core=core,options=self.options,source_population=len(self.data),
            frozen_dense_sha256=self.frozen_digest,candidate_promoted=False,goal_achieved=False)

    def resume(self,state,*,allow_completed=False):
        core=state.get('core',{});completed=core.get('completed')
        if (state.get('format')!=FORMAT or state.get('options')!=self.options or state.get('source_population')!=len(self.data)
                or state.get('frozen_dense_sha256')!=self.frozen_digest or type(completed) is not int
                or not 0<completed<=self.options['steps'] or len(core.get('metrics',[]))!=completed
                or state.get('candidate_promoted') is not False or state.get('goal_achieved') is not False):
            raise ValueError('Exact complete human epoch, frozen representation and objective state required before mutation')
        for step,record in enumerate(core['metrics']):validate_metric(self.data,self.positions(step),self.options,record,step)
        self.completed,self.metrics=restore_state(self.model,self.optimizer,core,groups=('actor',),expected_sources=self.sources,
            planned=self.options['steps'],expected_cursor=self.cursor(completed),device=self.device,allow_completed=allow_completed)
