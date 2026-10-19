"""Source-bound parallel CE with one verified shared policy/Adam snapshot.

Every replica proves the same base, actor, optimizer and global records before
root stores shared values once. All rank-private RNGs/cursors are retained.
Resume requires an independently retained semantic manifest and validates all
workers before any live tensor, optimizer or private generator is changed.
"""
from contextlib import nullcontext
import hashlib
import json
import math

import torch
from torch import distributed as dist

from .codenet_checkpoint_v1 import capture_rng,restore_rng
from .code_residual_checkpoint_v1 import _cpu,_validate_adam,_validate_rng
from .full_function_repair_stage_v1 import PROFILES,metadata
from .function_repair_curriculum_v1 import FunctionRepairCurriculum
from .own_dense_actor_checkpoint_v1 import selected
from .parallel_function_repair_v1 import FORMAT as OBJECTIVE,signature,update
from .whole_human_sft_evidence_v1 import tensor_digest

FORMAT='own-parallel-function-repair-shared-policy-Adam-all-private-RNG-v1'


def value_digest(value):
    digest=hashlib.sha256()
    def visit(x):
        if isinstance(x,torch.Tensor):
            t=x.detach().cpu().contiguous();digest.update(json.dumps(['tensor',str(t.dtype),list(t.shape)]).encode())
            digest.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x,dict):
            digest.update(b'dict{')
            for key in sorted(x,key=lambda k:(type(k).__name__,str(k))):visit(key);visit(x[key])
            digest.update(b'}')
        elif isinstance(x,(list,tuple)):
            digest.update(type(x).__name__.encode()+b'[')
            for item in x:visit(item)
            digest.update(b']')
        elif x is None or type(x) in {str,int,float,bool}:
            digest.update(json.dumps([type(x).__name__,x],ensure_ascii=True,allow_nan=False,separators=(',',':')).encode())
        else:raise ValueError('Only explicit tensor/primitive checkpoint values may be hashed')
    visit(value);return digest.hexdigest()


def manifest(state):
    """Semantic commit identity; retain externally and verify the file SHA too."""
    return dict(format=state['format'],world_size=state['world_size'],profile=state['profile'],mode=state['mode'],precision=state['precision'],
        completed=state['completed'],planned=state['planned'],schedule_identity_sha256=state['schedule_identity_sha256'],
        options_sha256=state['options_sha256'],sources_sha256=value_digest(state['model']['sources']),
        frozen_base_sha256=state['model']['frozen_dense_sha256'],model_parts_sha256=tensor_digest(state['model']['parts'].items()),
        optimizer_sha256=value_digest(state['optimizer']),global_metrics_sha256=value_digest(state['metrics']),
        global_cursor_sha256=value_digest(state['global_cursor']),
        all_private_worker_bindings=[value_digest(worker) for worker in state['private_workers']])


class ParallelFunctionRepairStage:
    def __init__(self,model,optimizer,data,entries,profile,options,sources,weights=None,*,rank=0,world=1,
                 control_group=None,gradient_group=None,mode='train',precision='bf16'):
        if (type(world) is not int or not 1<=world<=4 or type(rank) is not int or not 0<=rank<world
                or world>1 and (not dist.is_initialized() or dist.get_world_size(control_group)!=world
                    or dist.get_world_size(gradient_group)!=world or dist.get_rank(control_group)!=rank
                    or dist.get_backend(control_group)!='gloo' or dist.get_backend(gradient_group)!='gloo')):
            raise ValueError('Actual exact Gloo replica/control/host-gradient groups required')
        self.rank,self.world,self.control_group,self.gradient_group=rank,world,control_group,gradient_group
        error=None
        try:
            if (type(data) is not FunctionRepairCurriculum or data.role!='train' or data.purpose!='optimization'
                    or profile not in PROFILES or mode not in {'capacity','train'} or precision not in {'fp32','bf16'}
                    or not isinstance(entries,list) or not entries or optimizer.state):
                raise ValueError('Fresh selected optimizer, complete optimization-only source schedule and explicit mode/precision required')
            self.model,self.optimizer,self.data=model,optimizer,data;self.profile,self.mode,self.precision=profile,mode,precision
            self.entries=[list(batch) for batch in entries];self.options=_cpu(options);self.sources=dict(sources);self.weights=weights
            if (not self.sources or any(not isinstance(k,str) or not isinstance(v,str) or len(v)!=64
                    or any(c not in '0123456789abcdef' for c in v) for k,v in self.sources.items())
                    or any(type(self.options.get(k)) is not int or self.options[k]<1 for k in ['maximum_microbatch','maximum_microbatch_tokens'])
                    or any(type(self.options.get(k)) not in {int,float} or not math.isfinite(self.options[k]) or self.options[k]<=0
                        for k in ['learning_rate','gradient_clip'])):
                raise ValueError('Explicit source SHA identities and finite positive original update settings required')
            self.device=next(model.parameters()).device;self.named=selected(model,optimizer,('actor',));self.parameters=[p for _,p in self.named]
            chosen={id(p) for p in self.parameters};self.fixed=[p for p in model.parameters() if id(p) not in chosen]
            if any(p.dtype!=torch.float32 or p.device!=self.device or p.grad is not None for p in model.parameters()):
                raise ValueError('Actual complete FP32 own replicas at a clear optimizer boundary required')
            self.fixed_signature=signature(self.fixed);self.base_digest=tensor_digest(model.base.named_parameters())
            self.schedule_identity=data.identity(self.entries);self.options_digest=value_digest(self.options)
            self.expected=[metadata(data,batch,profile,weights) for batch in self.entries]
            self.completed=0;self.metrics=[];self.failed=False
            initial=dict(base_sha256=self.base_digest,parts_sha256=tensor_digest(model.parts.named_parameters()),
                options_sha256=self.options_digest,sources_sha256=value_digest(self.sources),schedule_sha256=self.schedule_identity,
                profile=profile,mode=mode,precision=precision,planned=len(entries),device_type=self.device.type,
                actual_counts=model.counts(),Adam_settings={k:v for k,v in optimizer.param_groups[0].items() if k!='params'})
        except Exception as exc:error=type(exc).__name__+': '+str(exc)
        self.coordinated(error,'stage construction')
        if any(row!=initial for row in self.gather(initial)):raise ValueError('Every actual initial replica, actor, source schedule and optimizer setting must agree')

    def gather(self,value):
        if self.world==1:return [value]
        result=[None]*self.world;dist.all_gather_object(result,value,group=self.control_group);return result

    def coordinated(self,error,phase):
        errors=self.gather(error)
        if any(x is not None for x in errors):raise ValueError('Coordinated '+phase+' rejection: '+repr(errors))

    def cursor(self,completed,rank=None):
        value=dict(completed=completed,profile=self.profile,mode=self.mode,precision=self.precision,world_size=self.world,
            schedule_identity_sha256=self.schedule_identity,all_consumed_global_source_batches=self.entries[:completed],original_source_role='train')
        if rank is not None:value.update(rank=rank,all_consumed_local_source_batches=[batch[rank::self.world] for batch in self.entries[:completed]])
        return value

    def validate_records(self,rows):
        if not isinstance(rows,list) or not 0<len(rows)<=len(self.entries):raise ValueError('Every actual complete source-prefix record required')
        arm=PROFILES[self.profile][0]
        for i,row in enumerate(rows):
            positions=self.entries[i]
            if (row.get('format')!=OBJECTIVE or row.get('step')!=i+1 or any(row.get(k)!=v for k,v in self.expected[i].items())
                    or row.get('actual_world_size')!=self.world or row.get('actual_device_type')!=self.device.type
                    or row.get('learning_rate')!=self.options['learning_rate'] or row.get('private_RNG_unchanged_by_CE_update') is not True
                    or row.get('every_original_target_and_true_EOS_retained') is not True
                    or row.get('neural_teacher_or_reference_queried') is not False
                    or row.get('empty_tail_ranks_never_forward_or_duplicate_examples') is not True
                    or row.get('actual_rank_source_shards')!=[positions[r::self.world] for r in range(self.world)]
                    or row.get('gradient_transport')!=('Gloo_host_SUM_already_global_normalized' if self.world>1 else 'serial_global_normalized')
                    or any(type(row.get(k)) not in {int,float} or not math.isfinite(row[k]) or row[k]<0 for k in
                        ['weighted_CE_sum','original_CE_sum','weighted_token_mean_NLL','original_token_mean_NLL','gradient_norm_before_clip'])):
                raise ValueError('Every original global source, positive mask, objective, clock and actual precision must reconstruct')
            actual=row.get('actual_rank_observations');width=max(1,min(self.options['maximum_microbatch'],
                self.options['maximum_microbatch_tokens']//max(self.data.lengths[p] for p in positions)))
            if not isinstance(actual,list) or len(actual)!=self.world:raise ValueError('Every actual replica source observation required')
            for rank,item in enumerate(actual):
                slots=positions[rank::self.world];microbatches=math.ceil(len(slots)/width)
                expected_precision=dict(actual_logits_dtype='torch.bfloat16' if self.precision=='bf16' else 'torch.float32',
                    autocast_enabled=self.precision=='bf16',actual_autocast_dtype='torch.bfloat16' if self.precision=='bf16' else None)
                if (item.get('rank')!=rank or item.get('actual_original_slots')!=slots
                        or item.get('actual_response_EOS_targets')!=sum(sum(v!=-100 for v in self.data.native(arm,p)[1]) for p in slots)
                        or item.get('complete_context_tokens')!=sum(len(self.data.native(arm,p)[0]) for p in slots)
                        or item.get('actual_microbatches')!=microbatches or item.get('no_forward_for_empty_tail')!=bool(not slots)
                        or item.get('actual_precision_per_forward')!=[expected_precision]*microbatches
                        or any(type(item.get(k)) not in {int,float} or not math.isfinite(item[k]) or item[k]<0 for k in ['weighted_CE_sum','original_CE_sum'])
                        or not slots and (item['weighted_CE_sum']!=0 or item['original_CE_sum']!=0)):
                    raise ValueError('Every real rank target, complete context, empty tail and actual forward precision required')
            weighted=math.fsum(v['weighted_CE_sum'] for v in actual);flat=math.fsum(v['original_CE_sum'] for v in actual)
            if (row['weighted_CE_sum']!=weighted or row['original_CE_sum']!=flat
                    or row['weighted_token_mean_NLL']!=weighted/row['actual_global_weighted_targets']
                    or row['original_token_mean_NLL']!=flat/row['response_EOS_targets']):
                raise ValueError('Actual globally summed complete-source NLL and declared denominators must agree')

    def update(self):
        error=None
        try:
            if self.failed or self.completed>=len(self.entries) or signature(self.fixed)!=self.fixed_signature:
                raise ValueError('Only a healthy remaining complete source step with unchanged frozen values may update')
            before_rng=value_digest(capture_rng(self.device))
        except Exception as exc:error=type(exc).__name__+': '+str(exc)
        self.coordinated(error,'stage update preflight')
        error=None
        try:
            amp=(lambda:torch.autocast(self.device.type,dtype=torch.bfloat16)) if self.precision=='bf16' else nullcontext
            row=update(self.model,self.optimizer,self.data,self.entries[self.completed],self.profile,self.options,self.weights,
                rank=self.rank,world=self.world,control_group=self.control_group,gradient_group=self.gradient_group,amp=amp)
            if before_rng!=value_digest(capture_rng(self.device)):raise ValueError('Zero-dropout source CE must not advance a private generator')
            row.update(step=self.completed+1,private_RNG_unchanged_by_CE_update=True)
            self.validate_records([*self.metrics,row])
        except Exception as exc:error=type(exc).__name__+': '+str(exc)
        errors=self.gather(error)
        if any(x is not None for x in errors):
            self.failed=True;self.optimizer.zero_grad(set_to_none=True)
            raise ValueError('No complete global update can be committed after a replica failure: '+repr(errors))
        if any(v!=value_digest(row) for v in self.gather(value_digest(row))):
            self.failed=True;raise ValueError('All actual global objective/source records must agree before publishing a completed step')
        self.completed+=1;self.metrics.append(row);return row

    def snapshot(self):
        error=None
        try:
            if self.failed or self.completed<1 or signature(self.fixed)!=self.fixed_signature or any(p.grad is not None for p in self.model.parameters()):
                raise ValueError('Shared state requires a healthy complete optimizer boundary with no partial gradients')
            self.validate_records(self.metrics);base=tensor_digest(self.model.base.named_parameters())
            if base!=self.base_digest:raise ValueError('Every original frozen neural value must remain exact')
            adam=_cpu(self.optimizer.state_dict());_validate_adam(adam,self.parameters,self.completed)
            rng=capture_rng(self.device);_validate_rng(rng,self.device)
            local=dict(rank=self.rank,local_source_cursor=self.cursor(self.completed,self.rank),rng=rng,
                replica=dict(frozen_base_sha256=base,model_parts_sha256=tensor_digest(self.model.parts.named_parameters()),
                    optimizer_sha256=value_digest(adam),global_metrics_sha256=value_digest(self.metrics)))
        except Exception as exc:error=type(exc).__name__+': '+str(exc)
        self.coordinated(error,'snapshot before shared serialization')
        workers=self.gather(local);state=receipt=None;error=None
        try:
            if [w['rank'] for w in workers]!=list(range(self.world)) or any(w['replica']!=workers[0]['replica'] for w in workers):
                raise ValueError('Every actual base/actor/Adam/global source record must be identical before storing one shared copy')
            if self.rank==0:
                payload=self.model.export(self.sources);self.model.validate_export(payload,expected_sources=self.sources)
                state=dict(format=FORMAT,world_size=self.world,profile=self.profile,mode=self.mode,precision=self.precision,
                    completed=self.completed,planned=len(self.entries),schedule_identity_sha256=self.schedule_identity,options_sha256=self.options_digest,
                    groups=['actor'],optimized_names=[n for n,_ in self.named],model=payload,optimizer=adam,
                    global_cursor=self.cursor(self.completed),metrics=_cpu(self.metrics),private_workers=_cpu(workers),
                    one_shared_policy_and_Adam_copy_verified_across_all_replicas=True,all_private_RNGs_and_source_cursors_retained=True,
                    original_frozen_own_base_required_for_actor_inference=True,candidate_promoted=False,goal_achieved=False)
                receipt=manifest(state)
        except Exception as exc:error=type(exc).__name__+': '+str(exc)
        self.coordinated(error,'shared snapshot construction')
        if self.world>1:
            value=[receipt];dist.broadcast_object_list(value,src=0,group=self.control_group);receipt=value[0]
        return state,receipt

    def validate_state(self,state,expected_manifest,allow_completed):
        if (type(self.data) is not FunctionRepairCurriculum or self.data.role!='train' or self.data.purpose!='optimization'
                or not isinstance(state,dict) or not isinstance(expected_manifest,dict) or not expected_manifest
                or manifest(state)!=expected_manifest or state.get('format')!=FORMAT or state.get('world_size')!=self.world
                or state.get('profile')!=self.profile or state.get('mode')!=self.mode or state.get('precision')!=self.precision
                or state.get('planned')!=len(self.entries) or type(state.get('completed')) is not int
                or not 0<state['completed']<=len(self.entries) or state['completed']==len(self.entries) and not allow_completed
                or state.get('schedule_identity_sha256')!=self.schedule_identity or state.get('options_sha256')!=self.options_digest
                or expected_manifest.get('frozen_base_sha256')!=self.base_digest
                or state.get('groups')!=['actor'] or state.get('optimized_names')!=[n for n,_ in self.named]
                or state.get('global_cursor')!=self.cursor(state['completed']) or len(state.get('metrics',[]))!=state['completed']
                or state.get('one_shared_policy_and_Adam_copy_verified_across_all_replicas') is not True
                or state.get('all_private_RNGs_and_source_cursors_retained') is not True
                or state.get('original_frozen_own_base_required_for_actor_inference') is not True
                or state.get('candidate_promoted') is not False or state.get('goal_achieved') is not False):
            raise ValueError('Exact independently retained shared manifest, complete source prefix, replica budget and mode required before mutation')
        self.validate_records(state['metrics']);self.model.validate_export(state['model'],expected_sources=self.sources)
        _validate_adam(state['optimizer'],self.parameters,state['completed'])
        if state['optimizer']['param_groups']!=self.optimizer.state_dict()['param_groups']:raise ValueError('Every original selected Adam setting must remain exact')
        selected_names={n for n,_ in self.named}
        if any(not torch.equal(t.detach().cpu(),state['model']['parts'][n]) for n,t in self.model.parts.state_dict().items() if n not in selected_names):
            raise ValueError('Every unused critic/actor value must remain exact before recovery')
        expected_replica=dict(frozen_base_sha256=self.base_digest,model_parts_sha256=expected_manifest['model_parts_sha256'],
            optimizer_sha256=expected_manifest['optimizer_sha256'],global_metrics_sha256=expected_manifest['global_metrics_sha256'])
        workers=state['private_workers']
        if not isinstance(workers,list) or len(workers)!=self.world:raise ValueError('Every original private worker state is required')
        for rank,worker in enumerate(workers):
            if (worker.get('rank')!=rank or worker.get('local_source_cursor')!=self.cursor(state['completed'],rank)
                    or worker.get('replica')!=expected_replica):raise ValueError('Every actual rank cursor and shared replica identity must reconstruct')
            # Validate every CUDA-format state against the local device class;
            # never initialize a different physical GPU while checking peers.
            _validate_rng(worker.get('rng'),self.device)
        if signature(self.fixed)!=self.fixed_signature:raise ValueError('Every original frozen parameter identity must remain unchanged')

    def resume(self,state,*,expected_manifest,allow_completed=False):
        error=None
        try:self.validate_state(state,expected_manifest,allow_completed)
        except Exception as exc:error=type(exc).__name__+': '+str(exc)
        self.coordinated(error,'complete shared/private recovery validation before mutation')
        if any(v!=expected_manifest for v in self.gather(expected_manifest)):
            raise ValueError('Every replica must use the same independently retained checkpoint manifest')
        with torch.no_grad():
            for name,p in self.named:p.copy_(state['model']['parts'][name])
        self.optimizer.load_state_dict(_cpu(state['optimizer']));self.optimizer.zero_grad(set_to_none=True)
        restore_rng(state['private_workers'][self.rank]['rng'],self.device)
        self.completed=state['completed'];self.metrics=_cpu(state['metrics']);self.failed=False
        if signature(self.fixed)!=self.fixed_signature:raise ValueError('Recovery must preserve every original frozen value and unused critic identity')
        return self.completed
