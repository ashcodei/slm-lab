"""Whole human/program CE with an optional exact existing-route KL anchor.

Both original train sources are fully covered. The shorter human stream may
repeat to match the program stream; every exposure is explicit. References are
the already counted original general/code routes, not a second policy or new
teacher weights. This kernel alone establishes no capability improvement.
"""
import math

import torch
from torch.nn import functional as F

from .codenet_stage_v1 import EpochOrder
from .own_aligned_assistant_v3 import OwnAlignedAssistantV3
from .own_alignment_checkpoint_v1 import make_state, parameters, restore_state
from .whole_human_sft_stage_v1 import _LanguageRestoreView

FORMAT = 'own-code-initialized-human-program-source-Adam-KL-v1'
REFERENCE_ROUTES = dict(human='general', generation='code')


def source_batch(streams, name, positions, device):
    if name == 'human': return streams[name].batch(positions,'response',device)
    return streams[name].encoder.batch([dict(packed_code_index=p) for p in positions],'response',device)


def objectives(student_logits, reference_logits, labels):
    labels = labels[:,1:]; mask = labels != -100
    student = student_logits[:,:-1].float()[mask]
    if not len(student): raise ValueError('Every original response requires its complete nonempty target/EOS mask')
    ce = F.cross_entropy(student,labels[mask],reduction='sum')
    if reference_logits is None: return ce,ce.detach()*0,len(student)
    reference = reference_logits[:,:-1].float()[mask].detach()
    if reference.shape != student.shape: raise ValueError('Same whole source positions and complete vocabulary required for the KL anchor')
    logp = student.log_softmax(-1); logq = reference.log_softmax(-1)
    kl = (logq.exp()*(logq-logp)).sum()
    return ce,kl,len(student)


class CodeInitializedAlignmentStage:
    def __init__(self, model, optimizer, streams, options, sources, device='cpu', amp=None):
        from contextlib import nullcontext
        if (type(model) is not OwnAlignedAssistantV3 or tuple(streams) != ('human','generation')
                or (streams['human'].family,streams['human'].role,streams['human'].purpose) != ('dialogue','train','optimization')
                or (streams['generation'].name,streams['generation'].role,streams['generation'].purpose) != ('generation','train','optimization')
                or any(not len(stream) for stream in streams.values())):
            raise ValueError('Exact code-initialized model and both original admitted train-only human/program streams required')
        keys = {'seed','steps','human_per_update','programs_per_update','learning_rate','betas','weight_decay','gradient_clip','reference_KL_alpha'}
        if (set(options) != keys or any(type(options[k]) is not int or options[k] < 1 for k in ('seed','steps','human_per_update','programs_per_update'))
                or options['human_per_update'] != 4 or options['programs_per_update'] != 1
                or options['steps'] != max(math.ceil(len(streams['human'])/4),len(streams['generation']))
                or any(type(options[k]) not in (int,float) or not math.isfinite(options[k]) or options[k] <= 0
                       for k in ('learning_rate','weight_decay','gradient_clip'))
                or not 0 <= options['reference_KL_alpha'] <= 1 or not math.isfinite(options['reference_KL_alpha'])
                or len(options['betas']) != 2 or any(not 0 <= beta < 1 for beta in options['betas'])):
            raise ValueError('Both complete source passes, explicit source widths and finite fixed objective/Adam options required')
        self.model,self.optimizer,self.streams = model,optimizer,streams
        self.options,self.sources,self.device = dict(options),dict(sources),torch.device(device)
        self.amp = amp or nullcontext; self.parameters = parameters(model,optimizer,('language',))
        group = optimizer.param_groups[0]
        if (group['lr'] != options['learning_rate'] or group['weight_decay'] != options['weight_decay']
                or tuple(group['betas']) != tuple(options['betas'])):
            raise ValueError('Exact fixed ordinary new-language-only AdamW hyperparameters required')
        selected = {id(p) for p in self.parameters}; self.fixed = [p for p in model.parameters() if id(p) not in selected]
        self.orders = {name:EpochOrder(len(stream),options['seed']+i*1009) for i,(name,stream) in enumerate(streams.items())}
        self.completed = 0; self.metrics = []

    def positions(self, step):
        if type(step) is not int or not 0 <= step < self.options['steps']: raise ValueError('Exact bounded original-source update required')
        return {name:[order.at(step*width+i) for i in range(width)] for name,order,width in
                [('human',self.orders['human'],4),('generation',self.orders['generation'],1)]}

    def cursor(self, completed):
        return {name:dict(source_population=len(self.streams[name]),actual_draws=completed*width,
            complete_passes=completed*width//len(self.streams[name]),next_pass_remainder=completed*width%len(self.streams[name]),
            seed=self.options['seed']+i*1009) for i,(name,width) in enumerate([('human',4),('generation',1)])}

    def update(self):
        positions = self.positions(self.completed); self.model.assert_word_contract()
        self.model.aligned.train(); self.model.original.eval(); self.optimizer.zero_grad(set_to_none=True)
        record = dict(step=self.completed+1,source_positions=positions,learning_rate=self.options['learning_rate'],
                      reference_KL_alpha=self.options['reference_KL_alpha'],streams={})
        try:
            for name in self.streams:
                batches = [source_batch(self.streams,name,[p],self.device) for p in positions[name]]
                expected = sum(int((batch['labels'][:,1:] != -100).sum()) for batch in batches)
                context = sum(int(batch['attention_mask'].sum()) for batch in batches)
                ce_sum = kl_sum = 0.; actual = 0
                for batch in batches:
                    inputs = {key:value for key,value in batch.items() if key != 'labels'}
                    with self.amp():
                        student = self.model.aligned(**inputs)['logits']; teacher = None
                        if self.options['reference_KL_alpha']:
                            with torch.no_grad(): teacher = self.model.route(REFERENCE_ROUTES[name])(**inputs)['logits']
                    ce,kl,count = objectives(student,teacher,batch['labels'])
                    if not bool(torch.isfinite(ce)) or not bool(torch.isfinite(kl)):
                        raise ValueError('Every complete source CE and exact-vocabulary KL scalar must be finite before mutation')
                    ((ce+self.options['reference_KL_alpha']*kl)/expected).backward()
                    actual += count; ce_sum += float(ce.detach()); kl_sum += float(kl.detach())
                if actual != expected: raise ValueError('Every complete source response target must enter its global branch normalizer')
                record['streams'][name] = dict(complete_context_tokens=context,response_EOS_targets=actual,
                    token_mean_CE=ce_sum/expected,token_mean_reference_KL=kl_sum/expected,reference_route=REFERENCE_ROUTES[name])
            if (any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in self.parameters)
                    or any(p.grad is not None for p in self.fixed)):
                raise ValueError('Every new language tensor and no existing own code/general/unused tensor must receive finite gradients')
            norm = float(torch.nn.utils.clip_grad_norm_(self.parameters,self.options['gradient_clip'],error_if_nonfinite=True))
            self.optimizer.step(); self.optimizer.zero_grad(set_to_none=True)
        except BaseException:
            self.optimizer.zero_grad(set_to_none=True); raise
        self.completed += 1; record['gradient_norm_before_clip'] = norm
        record['summed_objective'] = sum(r['token_mean_CE']+self.options['reference_KL_alpha']*r['token_mean_reference_KL'] for r in record['streams'].values())
        if not math.isfinite(record['summed_objective']): raise ValueError('Actual complete mixed source objective must be finite')
        self.metrics.append(record); return record

    def state(self):
        core = make_state(self.model,self.optimizer,groups=('language',),sources=self.sources,
            completed_steps=self.completed,planned_steps=self.options['steps'],sampler_state=self.cursor(self.completed),
            metrics=self.metrics,device=self.device)
        return dict(format=FORMAT,core=core,options=self.options,
                    source_populations={n:len(s) for n,s in self.streams.items()},candidate_promoted=False,goal_achieved=False)

    def resume(self, state, *, allow_completed=False):
        core = state.get('core',{}); steps = core.get('steps')
        if (state.get('format') != FORMAT or state.get('options') != self.options
                or state.get('source_populations') != {n:len(s) for n,s in self.streams.items()}
                or state.get('candidate_promoted') is not False or state.get('goal_achieved') is not False
                or type(steps) is not int or not 0 < steps <= self.options['steps'] or len(core.get('metrics',[])) != steps):
            raise ValueError('Exact complete original-source model/options/update contract required before mutation')
        for step,metric in enumerate(core['metrics']):
            positions = self.positions(step)
            if (metric.get('step') != step+1 or metric.get('source_positions') != positions
                    or metric.get('reference_KL_alpha') != self.options['reference_KL_alpha']
                    or metric.get('learning_rate') != self.options['learning_rate'] or set(metric.get('streams',{})) != set(self.streams)
                    or not math.isfinite(metric.get('gradient_norm_before_clip',float('nan')))
                    or metric['gradient_norm_before_clip'] < 0):
                raise ValueError('Every original mixed-source draw and fixed objective must reconstruct before resume')
            for name,stream_record in metric['streams'].items():
                batches = [source_batch(self.streams,name,[p],'cpu') for p in positions[name]]
                if (stream_record['complete_context_tokens'] != sum(int(b['attention_mask'].sum()) for b in batches)
                        or stream_record['response_EOS_targets'] != sum(int((b['labels'][:,1:]!=-100).sum()) for b in batches)
                        or stream_record['reference_route'] != REFERENCE_ROUTES[name]
                        or any(not math.isfinite(stream_record[k]) for k in ('token_mean_CE','token_mean_reference_KL'))
                        or stream_record['token_mean_CE'] < 0
                        or (self.options['reference_KL_alpha']==0 and stream_record['token_mean_reference_KL']!=0)):
                    raise ValueError('Complete source context/target normalizers and own frozen-reference routes must reconstruct')
            expected = sum(r['token_mean_CE']+self.options['reference_KL_alpha']*r['token_mean_reference_KL'] for r in metric['streams'].values())
            if metric.get('summed_objective') != expected:
                raise ValueError('Every complete mixed source objective aggregate must reconstruct before resume')
        self.completed,self.metrics = restore_state(_LanguageRestoreView(self.model),self.optimizer,core,
            groups=('language',),expected_sources=self.sources,planned_steps=self.options['steps'],
            expected_sampler_state=self.cursor(steps),device=self.device,allow_completed=allow_completed)
