"""Complete-program machine DPO and a matched chosen-code CE control.

Sources are separately admitted execution preferences, never human labels.
The reference is a frozen snapshot of our actual starting actor, including
nonzero learned adapters, sharing the single counted frozen dense backbone.
"""
from contextlib import nullcontext
import hashlib
import json
import math
import random

import torch
from torch.nn import functional as F

from .dense_human_preference_v1 import sequence_logp
from .execution_contrast_data_v1 import ExecutionContrastData
from .frozen_own_actor_reference_v1 import FrozenOwnActorReference, signature
from .own_dense_actor_v1 import OwnDenseActor
from .own_dense_actor_checkpoint_v1 import selected, make_state, restore_state
from .whole_human_sft_evidence_v1 import tensor_digest

FORMAT = 'own-full-program-machine-preference-source-epoch-v1'
METHODS = ('chosen_CE', 'machine_DPO')


def pair_information(data, position):
    row = data.rows[position]; chosen, cl = data.sequence(position, 'chosen'); rejected, rl = data.sequence(position, 'rejected')
    starts = [next((i for i, value in enumerate(labels) if value != -100), None) for labels in (cl, rl)]
    if (row['split'] != 'train' or row['publisher_split'] != 'train' or row['human_preferences'] is not False
            or row['optimization_admitted'] is not True or starts[0] is None or starts[0] != starts[1]
            or chosen[:starts[0]] != rejected[:starts[1]] or chosen[-1] != 2 or rejected[-1] != 2
            or cl != [-100] * starts[0] + chosen[starts[0]:] or rl != [-100] * starts[1] + rejected[starts[1]:]
            or row['counterexample']['two_fresh_completed_byte_identical_wrong_outputs'] is not True):
        raise ValueError('An admitted original train-only machine pair with complete shared prompt, programs and EOS is required')
    fingerprint = hashlib.sha256(json.dumps([chosen, cl, rejected, rl], separators=(',', ':')).encode()).hexdigest()
    return dict(original_train_position=position, id=row['id'], source_problem_id=row['source_problem_id'],
        original_record_sha256=row['original_record_sha256'], whole_native_pair_sha256=fingerprint,
        feedback_source='bounded_original_program_execution', human_preferences=False,
        complete_context_tokens=len(chosen)+len(rejected), chosen_response_EOS_targets=sum(v != -100 for v in cl[1:]),
        rejected_response_EOS_targets=sum(v != -100 for v in rl[1:]))


def gradients(model, data, positions, reference, *, method, beta=.1, chosen_CE_weight=.1,
              device='cpu', amp=None, normalization_pairs=None, normalization_chosen_targets=None):
    if (type(model) is not OwnDenseActor or type(data) is not ExecutionContrastData
            or (data.family, data.role, data.purpose) != ('machine_execution_preferences','train','optimization')
            or type(reference) is not FrozenOwnActorReference or method not in METHODS or not positions
            or len(set(positions)) != len(positions) or any(type(p) is not int or not 0 <= p < len(data) for p in positions)
            or not math.isfinite(beta) or not 0 < beta <= 1 or not math.isfinite(chosen_CE_weight)
            or not 0 < chosen_CE_weight <= 1):
        raise ValueError('An exact own actor, own starting reference and admitted complete train-only machine preferences are required')
    reference.assert_bound(model)
    information = [pair_information(data, p) for p in positions]
    targets = sum(row['chosen_response_EOS_targets'] for row in information)
    pairs = len(positions) if normalization_pairs is None else normalization_pairs
    denominator = targets if normalization_chosen_targets is None else normalization_chosen_targets
    if type(pairs) is not int or pairs < len(positions) or type(denominator) is not int or denominator < targets:
        raise ValueError('Explicit positive full global pair and chosen-target denominators are required')
    parameters = [p for _, p in model.named_selected(('actor',))]; amp = amp or nullcontext
    model.train(); model.policy.gradient_checkpointing = True; records = []
    for info in information:
        values = {}; originals = {}
        for field in (('chosen',) if method == 'chosen_CE' else ('chosen','rejected')):
            batch = data.batch([info['original_train_position']], field, device)
            inputs = {k:v for k,v in batch.items() if k != 'labels'}
            with amp():
                logits = model(**inputs)['logits']
                original = reference(**inputs)['logits'] if method == 'machine_DPO' else None
            values[field], count = sequence_logp(logits, batch['labels'])
            if count != info[field+'_response_EOS_targets']:
                raise ValueError('Every original complete response and EOS must reach the objective')
            if original is not None:
                originals[field], ref_count = sequence_logp(original, batch['labels'])
                if count != ref_count:
                    raise ValueError('The exact full same target mask must reach the frozen own reference')
        if method == 'machine_DPO':
            margin = (values['chosen']-values['rejected'])-(originals['chosen']-originals['rejected'])
            DPO = F.softplus(-beta*margin)
            objective = DPO/pairs-chosen_CE_weight*values['chosen']/denominator
        else:
            margin = DPO = None; objective = -chosen_CE_weight*values['chosen']/denominator
        if not bool(torch.isfinite(objective)):
            raise ValueError('Every actual complete machine preference objective must be finite')
        objective.backward(inputs=parameters)
        records.append(dict(**info, policy_chosen_logp=float(values['chosen'].detach()),
            policy_rejected_logp=None if method == 'chosen_CE' else float(values['rejected'].detach()),
            reference_chosen_logp=None if method == 'chosen_CE' else float(originals['chosen']),
            reference_rejected_logp=None if method == 'chosen_CE' else float(originals['rejected']),
            own_reference_preference_margin=None if margin is None else float(margin.detach()),
            pair_DPO=0. if DPO is None else float(DPO.detach())))
    result = dict(method=method, original_source_positions=list(positions), original_machine_pairs=records,
        beta=beta, chosen_CE_weight=chosen_CE_weight, normalization_pairs=pairs, normalization_chosen_targets=denominator,
        chosen_response_EOS_targets=targets,
        rejected_response_EOS_targets=sum(r['rejected_response_EOS_targets'] for r in information),
        complete_context_tokens=sum(r['complete_context_tokens'] for r in information),
        pair_mean_DPO=math.fsum(r['pair_DPO'] for r in records)/pairs,
        chosen_token_mean_NLL=-math.fsum(r['policy_chosen_logp'] for r in records)/targets,
        summed_objective=math.fsum(r['pair_DPO'] for r in records)/pairs
            -chosen_CE_weight*math.fsum(r['policy_chosen_logp'] for r in records)/denominator,
        own_starting_actor_reference_queried=method == 'machine_DPO', human_preferences=False,
        rejected_programs_never_positive_CE_targets=True, critic_and_frozen_dense_not_trained=True)
    reference.assert_bound(model)
    return result


def validate_metric(data, positions, options, record, step):
    expected = [pair_information(data, p) for p in positions]
    pairs = record.get('original_machine_pairs', [])
    if (record.get('step') != step+1 or record.get('method') != options['method']
            or record.get('original_source_positions') != positions or len(pairs) != len(expected)
            or record.get('human_preferences') is not False
            or record.get('rejected_programs_never_positive_CE_targets') is not True
            or record.get('critic_and_frozen_dense_not_trained') is not True
            or record.get('beta') != options['beta'] or record.get('chosen_CE_weight') != options['chosen_CE_weight']
            or record.get('normalization_pairs') != len(positions)
            or record.get('learning_rate') != options['learning_rate']
            or not math.isfinite(record.get('gradient_norm_before_clip', float('nan')))):
        raise ValueError('Exact machine source positions, objective and actual optimizer metrics are required before recovery')
    for info, row in zip(expected, pairs):
        if any(row.get(k) != v for k,v in info.items()):
            raise ValueError('Every original whole native pair and target fingerprint must rebuild before recovery')
        if not math.isfinite(row['policy_chosen_logp']) or row['policy_chosen_logp'] > 0:
            raise ValueError('Actual complete chosen log probability must be finite and nonpositive')
        fields = ['policy_chosen_logp','policy_rejected_logp','reference_chosen_logp','reference_rejected_logp']
        if options['method'] == 'chosen_CE':
            if any(row[f] is not None for f in fields[1:]) or row['own_reference_preference_margin'] is not None or row['pair_DPO'] != 0:
                raise ValueError('The chosen-code control must not fabricate rejected or reference neural observations')
        else:
            if any(not math.isfinite(row[f]) or row[f] > 0 for f in fields):
                raise ValueError('Every actual whole-policy/reference log probability must be finite')
            pc,pr,qc,qr = [torch.tensor(row[f],dtype=torch.float32) for f in fields]
            margin = float((pc-pr)-(qc-qr)); loss = float(F.softplus(-options['beta']*torch.tensor(margin)))
            if row['own_reference_preference_margin'] != margin or not math.isclose(row['pair_DPO'],loss,abs_tol=3e-6,rel_tol=3e-6):
                raise ValueError('The full-response FP32 own-reference margin and machine DPO must reconstruct')
    for key in ['chosen_response_EOS_targets','rejected_response_EOS_targets','complete_context_tokens']:
        if record.get(key) != sum(row[key] for row in expected):
            raise ValueError('Every original complete-context and response/EOS denominator must agree')
    targets = record['chosen_response_EOS_targets']
    mean = math.fsum(row['pair_DPO'] for row in pairs)/len(pairs)
    chosen = -math.fsum(row['policy_chosen_logp'] for row in pairs)/targets
    objective = mean-options['chosen_CE_weight']*math.fsum(row['policy_chosen_logp'] for row in pairs)/targets
    if (record.get('pair_mean_DPO') != mean or record.get('chosen_token_mean_NLL') != chosen
            or record.get('summed_objective') != objective or record.get('normalization_chosen_targets') != targets
            or record.get('own_starting_actor_reference_queried') != (options['method'] == 'machine_DPO')):
        raise ValueError('Every actual machine/control objective and full target normalization must reconstruct')


class MachineCodePreferenceStage:
    def __init__(self, model, optimizer, data, reference, options, sources, *, CPU_order=None):
        keys = {'method','seed','global_batch_size','learning_rate','betas','weight_decay','gradient_clip','beta','chosen_CE_weight'}
        if (type(model) is not OwnDenseActor or type(data) is not ExecutionContrastData
                or (data.family,data.role,data.purpose) != ('machine_execution_preferences','train','optimization')
                or set(options) != keys or options['method'] not in METHODS
                or any(type(options[k]) is not int or options[k] < 1 for k in ['seed','global_batch_size'])
                or any(not math.isfinite(options[k]) or options[k] <= 0 for k in ['learning_rate','gradient_clip','beta','chosen_CE_weight'])
                or options['beta'] > 1 or options['chosen_CE_weight'] > 1
                or not math.isfinite(options['weight_decay']) or options['weight_decay'] < 0
                or len(options['betas']) != 2 or any(not 0 <= v < 1 for v in options['betas'])
                or not sources or any(not isinstance(k,str) or not isinstance(v,str) or len(v)!=64
                    or any(c not in '0123456789abcdef' for c in v) for k,v in sources.items())):
            raise ValueError('Exact admitted machine source, complete fixed objective and own actor are required')
        self.model,self.optimizer,self.data,self.reference = model,optimizer,data,reference
        self.device=next(model.parameters()).device; self.options=dict(options); self.sources=dict(sources)
        reference.assert_bound(model)
        named=selected(model,optimizer,('actor',)); self.parameters=[p for _,p in named]
        group=optimizer.param_groups[0]
        if group['lr'] != options['learning_rate'] or tuple(group['betas']) != tuple(options['betas']) or group['weight_decay'] != options['weight_decay']:
            raise ValueError('The exact fixed selected actor AdamW profile is required')
        self.order=list(range(len(data)));random.Random(options['seed']).shuffle(self.order)
        if CPU_order is not None:
            if (self.device.type != 'cpu' or not CPU_order or len(set(CPU_order)) != len(CPU_order)
                    or any(type(p) is not int or not 0 <= p < len(data) for p in CPU_order)):
                raise ValueError('Explicit limited mechanics source positions are CPU-only, never a production curriculum')
            self.order=list(CPU_order)
        self.CPU_order=None if CPU_order is None else list(CPU_order)
        self.planned=math.ceil(len(self.order)/options['global_batch_size']); self.completed=0; self.metrics=[]
        selected_ids={id(p) for p in self.parameters}; self.fixed=[p for p in model.parameters() if id(p) not in selected_ids]
        self.fixed_signature=signature(self.fixed);self.frozen_dense_sha256=tensor_digest(model.base.named_parameters())

    def positions(self, step):
        if type(step) is not int or not 0 <= step < self.planned:
            raise ValueError('The complete declared original machine source order is exhausted')
        width=self.options['global_batch_size'];return self.order[step*width:(step+1)*width]

    def cursor(self, completed):
        return dict(completed_updates=completed,planned_updates=self.planned,original_source_population=len(self.data),
                    source_order=self.order,actual_pair_draws=min(completed*self.options['global_batch_size'],len(self.order)),
                    CPU_mechanics_order=self.CPU_order,full_source_epoch=self.CPU_order is None,repeated_rounding_pairs=0)

    def update(self):
        if signature(self.fixed) != self.fixed_signature:
            raise ValueError('A frozen dense/vision/word/critic parameter changed before machine preference learning')
        positions=self.positions(self.completed);self.optimizer.zero_grad(set_to_none=True)
        try:
            row=gradients(self.model,self.data,positions,self.reference,method=self.options['method'],beta=self.options['beta'],
                          chosen_CE_weight=self.options['chosen_CE_weight'],device=self.device)
            if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in self.parameters) or any(p.grad is not None for p in self.fixed):
                raise ValueError('Every actor and no frozen base/head/critic leaf must have finite gradients')
            norm=float(torch.nn.utils.clip_grad_norm_(self.parameters,self.options['gradient_clip'],error_if_nonfinite=True))
            row.update(step=self.completed+1,learning_rate=self.options['learning_rate'],gradient_norm_before_clip=norm)
            validate_metric(self.data,positions,self.options,row,self.completed)
            self.optimizer.step();self.optimizer.zero_grad(set_to_none=True)
        except BaseException:
            self.optimizer.zero_grad(set_to_none=True);raise
        self.completed+=1;self.metrics.append(row);self.reference.assert_bound(self.model);return row

    def state(self):
        self.reference.assert_bound(self.model)
        core=make_state(self.model,self.optimizer,groups=('actor',),sources=self.sources,completed=self.completed,
                        planned=self.planned,cursor=self.cursor(self.completed),metrics=self.metrics,device=self.device)
        if core['model']['frozen_dense_sha256'] != self.frozen_dense_sha256:
            raise ValueError('Every original frozen dense/vision/word value must remain exact')
        return dict(format=FORMAT,core=core,options=self.options,source_population=len(self.data),
                    CPU_order=self.CPU_order,own_reference_binding=self.reference.binding(),
                    human_preferences=False,candidate_promoted=False,goal_achieved=False)

    def resume(self, state, *, allow_completed=False):
        completed=state.get('core',{}).get('completed')
        if (state.get('format') != FORMAT or state.get('options') != self.options
                or state.get('source_population') != len(self.data) or state.get('CPU_order') != self.CPU_order
                or state.get('own_reference_binding') != self.reference.binding() or state.get('human_preferences') is not False
                or state.get('candidate_promoted') is not False or state.get('goal_achieved') is not False
                or type(completed) is not int or not 0 < completed <= self.planned
                or len(state.get('core',{}).get('metrics',[])) != completed):
            raise ValueError('Every complete machine source, objective and own starting-reference binding must match before recovery')
        self.reference.assert_bound(self.model)
        for step,row in enumerate(state['core']['metrics']):validate_metric(self.data,self.positions(step),self.options,row,step)
        self.completed,self.metrics=restore_state(self.model,self.optimizer,state['core'],groups=('actor',),
            expected_sources=self.sources,planned=self.planned,expected_cursor=self.cursor(completed),device=self.device,
            allow_completed=allow_completed)
