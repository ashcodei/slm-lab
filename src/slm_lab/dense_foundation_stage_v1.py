"""Full dense-language CE with optional exact existing own-route KL anchors.

Every source is visited once per epoch, including the last partial update.
Only the new dense model's language matrices/table/norms enter Adam. The
optional reference is our already trained original general/code model, held
separately for training, never an external teacher or part of the exported
under1B inference inventory.
"""
from collections import Counter
from contextlib import nullcontext
import math

import torch
from torch.nn import functional as F

from .codenet_stage_v1 import EpochOrder
from .dense_foundation_data_v1 import DenseFoundationSources,NAMES,REFERENCE_ROUTES
from .own_assistant_v1 import OwnAssistant
from .own_dense_growth_v1 import OwnDenseAssistant
from .own_dense_checkpoint_v1 import make_state,restore_state,selected
from .point_grounding_evaluation_v1 import state_digest

FORMAT='own-full-dense-foundation-complete-books-code-human-CE-KL-v1'


def objectives(logits,reference_logits,labels):
    target=labels[:,1:];mask=target!=-100;student=logits[:,:-1].float()[mask]
    if not len(student):raise ValueError('Every complete original source needs at least one real next/responseEOS target')
    ce=F.cross_entropy(student,target[mask],reduction='sum')
    if reference_logits is None:return ce,ce.detach()*0,len(student)
    reference=reference_logits[:,:-1].float()[mask].detach()
    if reference.shape!=student.shape:raise ValueError('Exact complete target positions and whole vocabulary required for the own-reference KL')
    logp=student.log_softmax(-1);logq=reference.log_softmax(-1)
    return ce,(logq.exp()*(logq-logp)).sum(),len(student)


def accumulate_gradients(model,reference,sources,positions,options,device='cpu',amp=None):
    if (type(model) is not OwnDenseAssistant or type(sources) is not DenseFoundationSources or not positions
            or (options['reference_KL_alpha'] and type(reference) is not OwnAssistant)):
        raise ValueError('Exact own dense model, complete admitted train sources and optional own reference required')
    parameters=[p for _,p in model.language_named_parameters()];amp=amp or nullcontext
    information=[sources.information(p) for p in positions]
    normalizer=math.fsum(options['coefficients'][r['family']]*r['response_EOS_targets'] for r in information)
    if normalizer<=0:raise ValueError('Every original weighted target must enter its global normalizer')
    totals={};reference_queried=False
    for info in information:
        family=info['family'];batch=sources.batch(info['global_position'],device)
        inputs={k:v for k,v in batch.items() if k!='labels'}
        with amp():
            student=model(**inputs)['logits'];teacher=None
            if options['reference_KL_alpha']:
                with torch.no_grad():teacher=reference(**inputs,route=REFERENCE_ROUTES[family])['logits']
                reference_queried=True
        ce,kl,count=objectives(student,teacher,batch['labels'])
        if (count!=info['response_EOS_targets'] or int(batch['attention_mask'].sum())!=info['complete_context_tokens']
                or not bool(torch.isfinite(ce)) or not bool(torch.isfinite(kl))):
            raise ValueError('Every complete source target/context and exact CE/KL scalar must be finite and match before mutation')
        (options['coefficients'][family]*(ce+options['reference_KL_alpha']*kl)/normalizer).backward(inputs=parameters)
        r=totals.setdefault(family,dict(examples=0,complete_context_tokens=0,response_EOS_targets=0,CE_sum=0.,reference_KL_sum=0.))
        r['examples']+=1;r['complete_context_tokens']+=info['complete_context_tokens'];r['response_EOS_targets']+=count
        r['CE_sum']+=float(ce.detach());r['reference_KL_sum']+=float(kl.detach())
        del student,teacher,ce,kl
    streams={name:dict(**record,token_mean_CE=record['CE_sum']/record['response_EOS_targets'],
        token_mean_reference_KL=record['reference_KL_sum']/record['response_EOS_targets'],reference_route=REFERENCE_ROUTES[name],coefficient=options['coefficients'][name])
        for name,record in totals.items()}
    objective=math.fsum(r['coefficient']*(r['CE_sum']+options['reference_KL_alpha']*r['reference_KL_sum']) for r in streams.values())/normalizer
    return dict(source_positions=positions,original_source_records=information,streams=streams,weighted_target_normalizer=normalizer,
                summed_objective=objective,reference_queried=reference_queried)


def signature(model):return [(id(p),p._version,p.data_ptr(),p.requires_grad) for p in model.parameters()]


class DenseFoundationStage:
    def __init__(self,model,optimizer,reference,data,options,sources,device='cpu',amp=None):
        keys={'seed','epochs','steps','microbatch','accumulation','learning_rate','betas','weight_decay','gradient_clip','reference_KL_alpha','coefficients'}
        if (type(model) is not OwnDenseAssistant or type(data) is not DenseFoundationSources
                or (data.role,data.purpose)!=('train','optimization') or set(options)!=keys
                or any(type(options[k]) is not int or options[k]<1 for k in ('epochs','steps','microbatch','accumulation'))
                or type(options['seed']) is not int or options['microbatch']!=1
                or any(type(options[k]) not in (float,int) or not math.isfinite(options[k]) or options[k]<=0
                       for k in ('learning_rate','weight_decay','gradient_clip'))
                or not math.isfinite(options['reference_KL_alpha']) or not 0<=options['reference_KL_alpha']<=1
                or set(options['coefficients'])!=set(NAMES) or any(not math.isfinite(v) or v<=0 for v in options['coefficients'].values())
                or len(options['betas'])!=2 or any(not 0<=v<1 for v in options['betas'])):
            raise ValueError('Exact full dense language/admitted train corpus and finite fixed whole-source Adam/objective contract required')
        self.width=options['accumulation'];self.steps_per_epoch=math.ceil(len(data)/self.width)
        if options['steps']!=options['epochs']*self.steps_per_epoch:
            raise ValueError('Every complete original source epoch and unrepeated last partial batch required')
        if (options['reference_KL_alpha'] and type(reference) is not OwnAssistant) or (not options['reference_KL_alpha'] and reference is not None):
            raise ValueError('Own frozen reference required only for nonzero KL; control must not load an unused teacher')
        self.model,self.optimizer,self.reference,self.data=model,optimizer,reference,data
        self.options,self.sources,self.device=dict(options),dict(sources),torch.device(device);self.amp=amp or nullcontext
        self.parameters=[p for _,p in selected(model,optimizer)];group=optimizer.param_groups[0]
        if (group['lr']!=options['learning_rate'] or group['weight_decay']!=options['weight_decay'] or tuple(group['betas'])!=tuple(options['betas'])):
            raise ValueError('Exact fixed ordinary full dense-language Adam hyperparameters required')
        ids={id(p) for p in self.parameters};self.fixed=[p for p in model.parameters() if id(p) not in ids]
        self.reference_signature=signature(reference) if reference is not None else None
        self.reference_digest=state_digest(reference) if reference is not None else None
        self.order=EpochOrder(len(data),options['seed']);self.completed=0;self.metrics=[]

    def positions(self,step):
        if type(step) is not int or not 0<=step<self.options['steps']:raise ValueError('Exact bounded full-corpus update required')
        epoch,within=divmod(step,self.steps_per_epoch)
        return [self.order.at(epoch*len(self.data)+i) for i in range(within*self.width,min((within+1)*self.width,len(self.data)))]

    def cursor(self,completed):
        epochs,within=divmod(completed,self.steps_per_epoch);offset=min(within*self.width,len(self.data))
        return dict(completed_updates=completed,source_population=len(self.data),complete_passes=epochs,next_pass_sources=offset,
            actual_draws=epochs*len(self.data)+offset,repeated_rounding_sources=0,seed=self.options['seed'],world_size=1)

    def update(self):
        positions=self.positions(self.completed);self.model.train();self.model.network.gradient_checkpointing=True
        if self.reference is not None:
            if signature(self.reference)!=self.reference_signature:raise ValueError('Every own reference parameter and its metadata must remain fixed')
            self.reference.eval()
        self.optimizer.zero_grad(set_to_none=True)
        try:
            record=accumulate_gradients(self.model,self.reference,self.data,positions,self.options,self.device,self.amp)
            if (any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in self.parameters)
                    or any(p.grad is not None for p in self.fixed)
                    or (self.reference is not None and any(p.grad is not None for p in self.reference.parameters()))):
                raise ValueError('Every full dense language tensor and no fixed/ref/vision/head tensor must receive finite gradients')
            norm=float(torch.nn.utils.clip_grad_norm_(self.parameters,self.options['gradient_clip'],error_if_nonfinite=True))
            self.optimizer.step();self.optimizer.zero_grad(set_to_none=True)
        except BaseException:
            self.optimizer.zero_grad(set_to_none=True);raise
        self.completed+=1;record.update(step=self.completed,learning_rate=self.options['learning_rate'],gradient_norm_before_clip=norm)
        self.metrics.append(record);return record

    def state(self):
        if self.reference is not None and (signature(self.reference)!=self.reference_signature or state_digest(self.reference)!=self.reference_digest):
            raise ValueError('The complete own reference weights must remain exact before snapshot')
        return dict(format=FORMAT,core=make_state(self.model,self.optimizer,sources=self.sources,completed=self.completed,
            planned=self.options['steps'],cursor=self.cursor(self.completed),metrics=self.metrics,device=self.device),options=self.options,
            source_populations=self.data.populations,reference_digest=self.reference_digest,candidate_promoted=False,goal_achieved=False)

    def resume(self,state,*,allow_completed=False):
        core=state.get('core',{});steps=core.get('completed')
        if (state.get('format')!=FORMAT or state.get('options')!=self.options or state.get('source_populations')!=self.data.populations
                or state.get('reference_digest')!=self.reference_digest or type(steps) is not int
                or not 0<steps<=self.options['steps'] or len(core.get('metrics',[]))!=steps
                or state.get('candidate_promoted') is not False or state.get('goal_achieved') is not False):
            raise ValueError('Exact complete dense corpus/options/reference/update state required before mutation')
        for step,record in enumerate(core['metrics']):
            positions=self.positions(step);info=[self.data.information(p) for p in positions]
            if (record['step']!=step+1 or record['source_positions']!=positions or record['original_source_records']!=info
                    or record['learning_rate']!=self.options['learning_rate'] or record['reference_queried']!=bool(self.options['reference_KL_alpha'])
                    or not math.isfinite(record['gradient_norm_before_clip']) or record['gradient_norm_before_clip']<0):
                raise ValueError('Every original complete source draw and fixed reference/objective profile must reconstruct')
            expected_names={r['family'] for r in info};normalizer=math.fsum(self.options['coefficients'][r['family']]*r['response_EOS_targets'] for r in info)
            if set(record['streams'])!=expected_names or record['weighted_target_normalizer']!=normalizer:
                raise ValueError('Every complete original weighted target normalizer required')
            for name,item in record['streams'].items():
                chosen=[r for r in info if r['family']==name];count=sum(r['response_EOS_targets'] for r in chosen)
                if (item['examples']!=len(chosen) or item['complete_context_tokens']!=sum(r['complete_context_tokens'] for r in chosen)
                        or item['response_EOS_targets']!=count or item['reference_route']!=REFERENCE_ROUTES[name]
                        or item['coefficient']!=self.options['coefficients'][name]
                        or any(not math.isfinite(item[k]) for k in ('CE_sum','reference_KL_sum','token_mean_CE','token_mean_reference_KL'))
                        or item['token_mean_CE']!=item['CE_sum']/count or item['token_mean_reference_KL']!=item['reference_KL_sum']/count
                        or (self.options['reference_KL_alpha']==0 and item['reference_KL_sum']!=0)):
                    raise ValueError('Each source family count/CE/KL and original reference route must reconstruct')
            expected=math.fsum(r['coefficient']*(r['CE_sum']+self.options['reference_KL_alpha']*r['reference_KL_sum']) for r in record['streams'].values())/normalizer
            if record['summed_objective']!=expected:raise ValueError('The complete weighted full-vocabulary objective must reconstruct')
        if self.reference is not None and (signature(self.reference)!=self.reference_signature or state_digest(self.reference)!=self.reference_digest):
            raise ValueError('Own frozen reference changed before recovery')
        self.completed,self.metrics=restore_state(self.model,self.optimizer,core,expected_sources=self.sources,
            planned=self.options['steps'],expected_cursor=self.cursor(steps),device=self.device,allow_completed=allow_completed)
