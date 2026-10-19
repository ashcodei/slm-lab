"""Whole-source, target-matched repair CE with explicit code-weight ablation."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from .code_target_regions_v1 import regions,weighted_CE
from .corpus import file_hash
from .function_repair_curriculum_v1 import FunctionRepairCurriculum
from .own_dense_actor_v1 import OwnDenseActor
from .own_dense_actor_checkpoint_v1 import make_state,restore_state

FORMAT='own-full-original-function-and-semantic-repair-CE-v1'
PROFILES={'plain_repeat':('plain_repeat',False), 'semantic_repair':('semantic_repair',False),
          'weighted_semantic_repair':('semantic_repair',True)}


class CodeWeights:
    def __init__(self,data):
        root=data.root;folder=root/'reports/full-function-code-target-regions-CPU-v1'
        report=json.loads((folder/'complete.json').read_text())
        if (not report.get('complete') or report.get('original_train_functions')!=826
                or not report.get('all_original_target_and_EOS_IDs_exact') or report.get('true_EOS_weight')!=1.):
            raise ValueError('Complete original source/code-region annotation proof required')
        for path,digest in report['source_files_sha256'].items():
            if file_hash(path)!=digest:raise ValueError('Original code-region source or annotation bytes changed')
        stored=np.load(folder/'complete-original-response-weight-candidate.npz',allow_pickle=False)
        index=stored['index'];values=stored['weights'];tokenizer=Tokenizer.from_file(str(data.tokenizer_path));self.rows=[];offset=0
        if len(index)!=826 or values.dtype!=np.dtype('<f4') or values.ndim!=1 or not np.isfinite(values).all():
            raise ValueError('Every finite original FP32 target weight required')
        for position,(entry,source) in enumerate(zip(index,data.original_rows)):
            actual=regions(source['response'],tokenizer,code_weight=4.)
            expected=np.asarray(actual['positive_target_weights'],dtype='<f4')
            if (int(entry['train_position'])!=position or int(entry['original_global_index'])!=int(data.original.positions[position])
                    or int(entry['offset'])!=offset or int(entry['targets'])!=len(expected)
                    or not np.array_equal(values[offset:offset+len(expected)],expected)
                    or actual['whole_original_response_EOS_ids']!=[v for v in data.plain[position][1] if v!=-100]):
                raise ValueError('Every actual original target ID and code/docstring weight must rebuild')
            self.rows.append(expected.copy());offset+=len(expected)
        if offset!=len(values):raise ValueError('No unreferenced target weight permitted')
        self.source_files_sha256={**report['source_files_sha256'],str(folder/'complete.json'):file_hash(folder/'complete.json')}

    def for_slot(self,data,slot):return self.rows[data.slots[slot][0]]


def metadata(data,positions,profile,weights):
    if profile not in PROFILES or not positions or len(set(positions))!=len(positions):
        raise ValueError('Distinct declared source presentation slots and a fixed matched profile required')
    arm,weighted=PROFILES[profile];denominator=0.;targets=contexts=0;fingerprints=[]
    if weighted and type(weights) is not CodeWeights:raise ValueError('Exact original source code-region weights required')
    for slot in positions:
        ids,labels=data.native(arm,slot);original,alternate=data.slots[slot];positive=[v for v in labels if v!=-100]
        if weighted:
            value=weights.for_slot(data,slot)
            if len(value)!=len(positive) or float(value[-1])!=1. or np.any(value<1) or np.any(value>4):
                raise ValueError('Every original target and true EOS must retain its declared positive weight')
            denominator+=math.fsum(map(float,value))
        else:denominator+=len(positive)
        targets+=len(positive);contexts+=len(ids)
        fingerprints.append(dict(slot=slot,original_train_position=original,second_view=alternate,
            original_source_id=data.original_rows[original]['id'],target_ids_sha256=hashlib.sha256(np.asarray(positive,dtype='<u2').tobytes()).hexdigest()))
    return dict(profile=profile,data_arm=arm,code_weighted=weighted,original_presentation_slots=list(positions),
        original_source_fingerprints=fingerprints,response_EOS_targets=targets,complete_context_tokens=contexts,
        actual_global_weighted_targets=denominator)


def gradients(model,data,positions,profile,options,weights=None,amp=None):
    if (type(model) is not OwnDenseActor or type(data) is not FunctionRepairCurriculum
            or data.role!='train' or data.purpose!='optimization' or profile not in PROFILES):
        raise ValueError('Only admitted complete source-train functions and the exact own actor may supply gradients')
    from contextlib import nullcontext
    info=metadata(data,positions,profile,weights);arm,weighted=PROFILES[profile]
    parameters=[p for _,p in model.named_selected(('actor',))]
    width=max(1,min(options['maximum_microbatch'],options['maximum_microbatch_tokens']//max(data.lengths[p] for p in positions)))
    model.train();model.policy.gradient_checkpointing=True;total=unweighted=0.;actual_targets=0;microbatches=0
    amp=amp or nullcontext
    for begin in range(0,len(positions),width):
        slots=positions[begin:begin+width];batch=data.batch(arm,slots,for_optimization=True)
        target_weights=torch.zeros_like(batch['labels'],dtype=torch.float32)
        for i,slot in enumerate(slots):
            mask=batch['labels'][i]!=-100
            if weighted:target_weights[i,mask]=torch.from_numpy(weights.for_slot(data,slot).copy())
            else:target_weights[i,mask]=1.
        with amp():logits=model(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'])['logits']
        numerator,observed=weighted_CE(logits,batch['labels'],target_weights)
        if not bool(torch.isfinite(numerator)):raise ValueError('Every complete original source objective must be finite')
        (numerator/info['actual_global_weighted_targets']).backward(inputs=parameters)
        total+=float(numerator.detach());actual_targets+=int((batch['labels'][:,1:]!=-100).sum());microbatches+=1
        # The flat target NLL is retained even when code spans are emphasized.
        with torch.no_grad():
            flat,_=weighted_CE(logits.detach(),batch['labels'],(batch['labels']!=-100).float());unweighted+=float(flat)
        del logits,numerator,observed,batch,target_weights
    if actual_targets!=info['response_EOS_targets']:raise ValueError('No original source target or true EOS may be dropped')
    return dict(format=FORMAT,**info,weighted_token_mean_NLL=total/info['actual_global_weighted_targets'],
        original_token_mean_NLL=unweighted/actual_targets,actual_microbatches=microbatches,
        neural_teacher_or_reference_queried=False,failed_program_tokens_never_positive_targets=True,
        all_original_targets_and_EOS_retained=True)


class FunctionRepairStage:
    def __init__(self,model,optimizer,data,entries,profile,options,sources,weights=None,cache=None,mode='train'):
        if profile not in PROFILES or mode not in {'train','capacity'}:raise ValueError('Exact matched function profile and mode required')
        self.model,self.optimizer,self.data=model,optimizer,data;self.entries=entries;self.profile=profile
        self.options,self.sources,self.weights,self.cache,self.mode=options,sources,weights,cache,mode
        self.completed=0;self.metrics=[];self.parameters=[p for _,p in model.named_selected(('actor',))]
        selected={id(p) for p in self.parameters};self.fixed=[p for p in model.parameters() if id(p) not in selected]
        self.fixed_signature=self.signature();self.schedule_identity=data.identity(entries)
        self.expected=[metadata(data,positions,profile,weights) for positions in entries]

    def signature(self):return [(id(p),p._version,p.data_ptr(),p.requires_grad) for p in self.fixed]

    def update(self):
        if self.completed>=len(self.entries):raise ValueError('Complete source epoch cannot be silently extended')
        if self.signature()!=self.fixed_signature:raise ValueError('Every original frozen base/vision/word/critic must remain unchanged')
        if self.cache is not None:self.cache.assert_current()
        self.optimizer.zero_grad(set_to_none=True)
        record=gradients(self.model,self.data,self.entries[self.completed],self.profile,self.options,self.weights,
            amp=lambda:torch.autocast('cpu',dtype=torch.bfloat16))
        if (any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in self.parameters)
                or any(p.grad is not None for p in self.fixed)):
            raise ValueError('Every and only selected own actor tensor needs finite gradients')
        norm=float(torch.nn.utils.clip_grad_norm_(self.parameters,self.options['gradient_clip'],error_if_nonfinite=True))
        self.optimizer.step();self.optimizer.zero_grad(set_to_none=True)
        if self.signature()!=self.fixed_signature:raise ValueError('An unused frozen value changed during the actor update')
        if self.cache is not None:self.cache.assert_current()
        self.completed+=1;record.update(step=self.completed,learning_rate=self.options['learning_rate'],gradient_norm_before_clip=norm)
        self.metrics.append(record);return record

    def cursor(self,completed):
        return dict(completed=completed,profile=self.profile,mode=self.mode,schedule_identity_sha256=self.schedule_identity,
            complete_original_slot_batches=self.entries[:completed],original_source_roles_preserved=True)

    def validate(self,rows):
        if not isinstance(rows,list) or not 0<len(rows)<=len(self.entries):raise ValueError('Exact nonempty complete source prefix required')
        for i,row in enumerate(rows):
            if (row.get('format')!=FORMAT or row.get('step')!=i+1 or any(row.get(k)!=v for k,v in self.expected[i].items())
                    or row.get('learning_rate')!=self.options['learning_rate']
                    or row.get('neural_teacher_or_reference_queried') is not False
                    or row.get('failed_program_tokens_never_positive_targets') is not True
                    or row.get('all_original_targets_and_EOS_retained') is not True
                    or any(not isinstance(row.get(k),(int,float)) or not math.isfinite(row[k]) for k in
                        ['weighted_token_mean_NLL','original_token_mean_NLL','gradient_norm_before_clip'])):
                raise ValueError('Every actual source slot, complete target mask, weight and optimizer cursor must reconstruct before mutation')

    def state(self):
        self.validate(self.metrics)
        return make_state(self.model,self.optimizer,groups=('actor',),sources=self.sources,
            completed=self.completed,planned=len(self.entries),cursor=self.cursor(self.completed),metrics=self.metrics,device='cpu')

    def resume(self,state,*,allow_completed=False):
        self.validate(state.get('metrics'));count,records=restore_state(self.model,self.optimizer,state,groups=('actor',),
            expected_sources=self.sources,planned=len(self.entries),expected_cursor=self.cursor(state['completed']),
            device='cpu',allow_completed=allow_completed)
        self.completed,self.metrics=count,records
        if self.signature()!=self.fixed_signature:raise ValueError('Recovery must preserve every frozen original value and flag')
        return count
