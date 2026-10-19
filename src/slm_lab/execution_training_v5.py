"""Complete-code/all30-source training plus admitted execution CE and DPO.

This is a training kernel, not a launcher or an exception to final-source,
final-own-base, baseline, capacity, resource, or genuine-human-feedback gates.
Reference scores and every auxiliary cursor are part of the complete saved
sampler contract, so restoring never redefines the original frozen reference.
"""
from copy import deepcopy
import math

import torch
from torch import distributed as dist

from .code_residual_checkpoint_v1 import _digest
from .code_residual_collective_checkpoint_v2 import _same,_validated,fingerprint
from .code_residual_collective_checkpoint_v2 import FORMAT as TRAIN_FORMAT
from .code_residual_source_contract_v4 import SourceContractResidualKernel,FIELDS
from .code_residual_stage_v3 import _EncodedBatches
from .codenet_runner_v1 import average_gradients
from .codenet_stage_v1 import EpochOrder
from .codenet_checkpoint_v1 import FileBindings
from .execution_collective_v2 import (BUNDLE_FORMAT,feedback_identity,policy_identity,
    measure_preferences,backward_preferences)
from .execution_preference_v1 import FORMAT as CACHE_FORMAT
from .recurrent_ablation import state_digest

FORMAT='own-complete-code-execution-training-v5'
INT_OPTIONS={'positive_batch_size','positive_accumulation','pair_batch_size','pair_accumulation'}
FLOAT_OPTIONS={'positive_coefficient','DPO_alpha','DPO_beta'}


def reference_from_checkpoint(path,expected_checkpoint_sha256):
    """Recover the original frozen scores before constructing a resumed kernel.

    No model, optimizer or RNG is applied here. The kernel's ordinary collective
    restore still verifies every source, weight, Adam, RNG and sampler byte.
    """
    def read():
        guard=FileBindings({str(path):expected_checkpoint_sha256})
        state=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        progress=state.get('sampler_state',{})
        if (state.get('format')!=TRAIN_FORMAT or state.get('world_size')!=4 or progress.get('kernel')!=FORMAT
                or progress.get('updates')!=state.get('steps') or state.get('candidate_promoted') is not False
                or state.get('goal_achieved') is not False):
            raise ValueError('Exact original collective execution checkpoint required')
        bundle=progress['execution_reference_bundle'];options=progress['execution_options']
        if auxiliary_fingerprint(bundle,options)!=progress['execution_auxiliary_fingerprint']:
            raise ValueError('Original embedded frozen reference/options fingerprint differs')
        guard.check(full=True)
        return deepcopy(bundle)
    bundle=_validated(read);_same(fingerprint(bundle))
    return bundle


def validate_execution_options(options):
    if (set(options)!=INT_OPTIONS|FLOAT_OPTIONS
            or any(type(options[k]) is not int or options[k]<1 for k in INT_OPTIONS)
            or any(type(options[k]) not in {int,float} or not math.isfinite(options[k]) for k in FLOAT_OPTIONS)
            or options['positive_coefficient']<=0 or not 0<=options['DPO_alpha']<=1 or not 0<options['DPO_beta']<=1):
        raise ValueError('Exact explicit finite execution CE/DPO objectives and complete exposure layout required')


def auxiliary_fingerprint(reference_bundle,options):
    validate_execution_options(options)
    return fingerprint(dict(format=FORMAT,reference_bundle=reference_bundle,execution_options=options))


class _CompletePositiveEncoder:
    def __init__(self,feedback,config):
        self.feedback,self.config=feedback,config
        self.truncated=0

    def batch(self,rows,field,device):
        if field!='response' or any(set(r)!={'execution_positive_index'} for r in rows):
            raise ValueError('Explicit original complete verified-source positives required')
        sequences=[self.feedback.positive(row['execution_positive_index']) for row in rows]
        length=max(len(ids) for ids,_ in sequences)
        if length>self.config.max_text_length:raise ValueError('Whole original program must fit; no clipping')
        ids=torch.zeros((len(rows),length),device=device,dtype=torch.long)
        labels=torch.full_like(ids,-100);mask=torch.zeros_like(ids,dtype=torch.bool)
        for i,(tokens,target) in enumerate(sequences):
            ids[i,:len(tokens)]=torch.tensor(tokens,device=device)
            labels[i,:len(tokens)]=torch.tensor(target,device=device)
            mask[i,:len(tokens)]=True
        return dict(input_ids=ids,labels=labels,attention_mask=mask,pixel_values=None)


def global_CE_gradients(model,streams,config,device,amp):
    """Separately global-token-normalized CE; report local errors collectively.

    Forward/backward has no intermediate collectives. A one-worker numerical
    or shape error cannot strand other workers in the next objective reduction.
    The caller applies one fixed-order worker average to all accumulated losses.
    """
    world=dist.get_world_size()
    def prepare():
        counts=[]
        for _,encoder,coefficient in streams:
            if not encoder.batches or not math.isfinite(coefficient) or coefficient<=0:
                raise ValueError('Complete nonempty finite positive CE streams required')
            local=[]
            for batch in encoder.batches:
                ids,labels=batch['input_ids'],batch['labels']
                if (ids.ndim!=2 or ids.dtype!=torch.long or labels.shape!=ids.shape or labels.dtype!=torch.long
                        or not 2<=ids.shape[1]<=config.max_text_length
                        or bool(torch.any(ids<0)) or bool(torch.any(ids>=config.vocab_size))
                        or not torch.equal(labels[labels!=-100],ids[labels!=-100])):
                    raise ValueError('Exact original complete causal inputs/response masks and vocabulary required')
                n=int((labels[:,1:]!=-100).sum())
                if n<1:raise ValueError('Each original CE microbatch must retain its complete response targets')
                local.append(n)
            counts.append(local)
        return counts
    counts=_validated(prepare)
    _same(fingerprint([(name,c) for name,_,c in streams]))
    totals=torch.tensor([sum(n) for n in counts],device=device,dtype=torch.long);dist.all_reduce(totals)
    loss_sums=torch.zeros(len(streams),device=device,dtype=torch.float64)
    def backward():
        for j,((_,encoder,coefficient),local_counts) in enumerate(zip(streams,counts)):
            for batch,n in zip(encoder.batches,local_counts):
                with amp():loss=model(**batch)['loss']
                if loss.ndim!=0 or not bool(torch.isfinite(loss)):
                    raise ValueError('Finite complete original-code/image/pretraining CE required')
                (loss*(coefficient*world*n/int(totals[j]))).backward()
                loss_sums[j]+=loss.detach().double()*n
    _validated(backward)
    dist.all_reduce(loss_sums)
    objectives={name:dict(global_target_tokens=int(totals[j]),coefficient=coefficient,
        token_mean_nll=float(loss_sums[j]/totals[j]),weighted_objective=coefficient*float(loss_sums[j]/totals[j]))
        for j,(name,_,coefficient) in enumerate(streams)}
    return dict(objectives=objectives,CE_summed_objective=sum(r['weighted_objective'] for r in objectives.values()),
        CE_total_target_tokens=int(totals.sum()))


class ExecutionResidualKernel(SourceContractResidualKernel):
    def __init__(self,*args,feedback,reference_bundle,execution_options,expected_auxiliary_fingerprint,**kwargs):
        self.feedback=feedback
        self.reference_bundle=deepcopy(reference_bundle)
        self.execution_options=deepcopy(execution_options)
        self.auxiliary_fingerprint=expected_auxiliary_fingerprint
        # The parent requires all30 original sources and their original budget,
        # objective, image, parameter partition and external contract bindings.
        super().__init__(*args,**kwargs)
        def preflight():
            self._validate_auxiliary()
            policy=self.reference_bundle['policy']
            actual=policy_identity(self.model,self.sources['base_checkpoint_sha256'],self.sources['tokenizer_sha256'],
                self.device,policy['precision'])
            if actual!=policy:raise ValueError('Begin from the exact original own reference weights; restore the saved policy only afterwards')
            for key,size in [('positive',len(feedback.positives)),('pair',len(feedback.pairs))]:
                exposures=self.steps*self.world*self.execution_options[key+'_batch_size']*self.execution_options[key+'_accumulation']
                if exposures<size:raise ValueError('Complete declared source budget must cover every admitted auxiliary source')
            return fingerprint(dict(aux=self.auxiliary_fingerprint,code=self.budget,replay=self.contract_fingerprint))
        _same(_validated(preflight))
        self.positive_encoder=_CompletePositiveEncoder(feedback,self.model.config)

    def _validate_auxiliary(self):
        bundle=self.reference_bundle;cache=bundle.get('cache',{})
        if (not _digest(self.auxiliary_fingerprint) or bundle.get('format')!=BUNDLE_FORMAT
                or auxiliary_fingerprint(bundle,self.execution_options)!=self.auxiliary_fingerprint
                or bundle.get('feedback')!=feedback_identity(self.feedback,self.sources['tokenizer_sha256'])
                or bundle.get('human_preferences') is not False or cache.get('format')!=CACHE_FORMAT
                or bundle.get('cache_fingerprint')!=fingerprint(cache)
                or cache.get('policy_sha256')!=fingerprint(bundle['policy'])
                or set(cache.get('records',{}))!={r['id'] for r in self.feedback.pairs}):
            raise ValueError('Exact admitted auxiliary source/reference/options contract must remain unchanged')

    def _guard(self,full=False):
        super()._guard(full=full)
        def check():
            self.feedback.guard.check(full=full)
            self._validate_auxiliary()
            if full and state_digest(self.model.base.state_dict())!=self.reference_bundle['policy']['frozen_base_state_sha256']:
                raise ValueError('Every original own frozen backbone tensor must match its reference')
        _validated(check)

    def sampler_state(self,step):
        self._validate_auxiliary()
        state=super().sampler_state(step)
        aux={}
        for key,size in [('positive',len(self.feedback.positives)),('pair',len(self.feedback.pairs))]:
            draws=step*self.world*self.execution_options[key+'_batch_size']*self.execution_options[key+'_accumulation']
            aux[key]=dict(source_records=size,examples=draws,epoch_and_cursor=list(divmod(draws,size)),
                seed=self.options['seed']+(2000 if key=='positive' else 3000))
        return dict(**{k:v for k,v in state.items() if k!='kernel'},kernel=FORMAT,
            execution_auxiliary_fingerprint=self.auxiliary_fingerprint,execution_options=deepcopy(self.execution_options),
            execution_source_cursors=aux,execution_reference_bundle=deepcopy(self.reference_bundle))

    def update(self):
        def boundary():
            self._validate_auxiliary()
            if not 0<=self.completed<self.steps or len(self.metrics)!=self.completed:
                raise ValueError('The complete declared execution training budget is finished or inconsistent')
            return fingerprint(self.sampler_state(self.completed))
        _same(_validated(boundary));self._guard()
        self.model.train();self.model.code.gradient_checkpointing=True
        step=self.completed
        def prepare():
            name,occurrence=self.schedule.at(step);field=FIELDS[self.objectives[name]]
            code_groups=[[{'packed_code_index':i} for i in group] for group in EpochOrder(len(self.code),self.options['seed']).rank_groups(
                step,self.rank,self.world,self.options['batch_size'],self.options['gradient_accumulation'])]
            replay_groups=[[self.replay.row(name,i) for i in group] for group in EpochOrder(self.replay.size(name),
                self.options['seed']+1000+self.replay.families.index(name)).rank_groups(occurrence,self.rank,self.world,
                    self.options['replay_batch_size'],self.options['replay_accumulation'])]
            positive_groups=[[{'execution_positive_index':i} for i in group] for group in EpochOrder(len(self.feedback.positives),
                self.options['seed']+2000).rank_groups(step,self.rank,self.world,
                    self.execution_options['positive_batch_size'],self.execution_options['positive_accumulation'])]
            pair_positions=[i for group in EpochOrder(len(self.feedback.pairs),self.options['seed']+3000).rank_groups(
                step,self.rank,self.world,self.execution_options['pair_batch_size'],self.execution_options['pair_accumulation']) for i in group]
            encoded=[_EncodedBatches(self.code_encoder,code_groups,'response',self.device),
                _EncodedBatches(self.replay_encoder,replay_groups,field,self.device),
                _EncodedBatches(self.positive_encoder,positive_groups,'response',self.device)]
            if self.code_encoder.truncated or self.replay_encoder.truncated:
                raise ValueError('Every original code/replay input must remain complete')
            return name,occurrence,field,encoded,[self.feedback.example(i) for i in pair_positions],pair_positions,
        name,occurrence,field,encoded,examples,pair_positions=_validated(prepare)
        measured,DPO=measure_preferences(self.model.code,examples,self.reference_bundle['cache'],
            expected_policy_sha256=self.reference_bundle['cache']['policy_sha256'],
            alpha=self.execution_options['DPO_alpha'],beta=self.execution_options['DPO_beta'],device=self.device,amp=self.amp)
        lr=self.options['learning_rate']*min(1.,(step+1)/max(1,self.steps//10))
        lr*=.1+.9*.5*(1+math.cos(math.pi*step/self.steps))
        for group in self.optimizer.param_groups:group['lr']=lr
        self.optimizer.zero_grad(set_to_none=True)
        try:
            metrics=global_CE_gradients(self.wrapped,[('complete_code',encoded[0],1.),
                (name,encoded[1],self.options['replay_coefficient']),
                ('verified_original_program',encoded[2],self.execution_options['positive_coefficient'])],
                self.model.config,self.device,self.amp)
            backward_preferences(self.model.code,examples,measured,DPO,self.device,self.amp)
            def finite():
                if (any(p.grad is not None for p in self.model.base.parameters())
                        or any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in self.model.adapter_parameters())):
                    raise ValueError('All and only own code residual tensors require finite gradients')
            _validated(finite)
            average_gradients(self.wrapped,self.device,self.world)
            norm=torch.nn.utils.clip_grad_norm_(self.model.adapter_parameters(),self.options['gradient_clip'],error_if_nonfinite=True)
            self.optimizer.step();self.optimizer.zero_grad(set_to_none=True)
        except BaseException:
            self.optimizer.zero_grad(set_to_none=True)
            raise
        result=dict(**metrics,step=step+1,execution_DPO=DPO,
            summed_objective=metrics['CE_summed_objective']+DPO['weighted_DPO'],
            execution_pair_target_tokens=sum(r['chosen_targets']+r['rejected_targets'] for r in DPO['records']),
            total_target_tokens=metrics['CE_total_target_tokens']+sum(r['chosen_targets']+r['rejected_targets'] for r in DPO['records']),
            replay_family=name,replay_occurrence=occurrence,replay_objective=self.objectives[name],replay_response_field=field,
            learning_rate=lr,gradient_norm_before_clipping=float(norm),
            global_execution_pairs=DPO['global_pairs'],execution_reference_fingerprint=fingerprint(self.reference_bundle),
            machine_feedback_not_human_preferences=True)
        self.completed+=1;self.metrics.append(result)
        return result
