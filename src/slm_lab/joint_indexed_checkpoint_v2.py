"""Same complete joint state with source-only checked record batches.

Every original record is read and validated before checkpoint publication or
recovery. Only bounded receipt envelopes remain resident. Actual task-derived
sparse Adam clocks, all selected values and every private generator remain.
This protocol grants no source, full-size capacity or GPU training admission.
"""
import copy
import math
from pathlib import Path
import threading

import torch

from .codenet_checkpoint_v1 import capture_rng,restore_rng
from .code_residual_checkpoint_v1 import _cpu,_validate_rng
from .joint_indexed_history_v1 import JointRecords,Reader
from .closed_source_recovery_guard_v1 import SourceOnlyRecoveryGuard
from .indexed_native_history_v1 import limits,canonical,sha as bytes_sha
from .parallel_function_repair_stage_v1 import value_digest
from .shared_joint_checkpoint_v2 import context,validate_adam,agree,control,sha,common_digest,gather

FORMAT='own-indexed-joint-selected-Adam-all-original-records-private-RNG-v1'
ENVELOPE='own-joint-original-pre-Adam-metric-receipt-v1'
METRIC_FIELDS={'step','entry_sha256','active_parameter_names','normalized_task_losses','gradient_norm_before_clip',
    'globally_reduced_task_gradients_before_combining','gradient_interaction'}


def checked_owner(resource_guard):
    """Full resource checks bracket source-only reconstruction, not each row."""
    resource_guard();owner=threading.get_ident()
    def check():
        if threading.get_ident()!=owner:raise ValueError('Another thread cannot borrow the source-only joint validation batch')
    return check


def configuration(value,*,sources,metadata,world,optimizer,planned):
    if type(value) is not dict or set(value)!={'binding','budgets','maximum_attempts','history_root'}:
        raise ValueError('Exact joint history binding, complete budgets and bounded immutable attempts are required')
    budgets=limits(value['budgets']);binding=value['binding'];JointRecords(binding)
    root=Path(value['history_root']).absolute()
    if (binding['sources']!=sources or binding['model_metadata']!=metadata or binding['world_size']!=world
            or binding['optimizer_settings_sha256']!=value_digest(optimizer.state_dict()['param_groups'])
            or budgets['maximum_records']!=planned or type(value['maximum_attempts']) is not int or not 1<=value['maximum_attempts']<=16
            or str(root)!=value['history_root'] or root.resolve()!=root):
        raise ValueError('The whole original joint source/model/Adam/world/horizon/attempt bounds must match')
    return value


def validate_metric(view,entry,metric):
    active={n for task in entry['objectives'] for n in view.task_names(task['task'])}
    names=[n for n,_ in view.named_selected() if n in active]
    if (type(metric) is not dict or set(metric)!=METRIC_FIELDS or type(metric['step']) is not int or metric['step']!=entry['step']
            or metric['entry_sha256']!=value_digest(entry) or metric['active_parameter_names']!=names
            or metric['globally_reduced_task_gradients_before_combining'] is not True
            or type(metric['normalized_task_losses']) is not list or len(metric['normalized_task_losses'])!=len(entry['objectives'])
            or any(type(v) not in [int,float] or not math.isfinite(v) or v<0 for v in [*metric['normalized_task_losses'],metric['gradient_norm_before_clip']])
            or type(metric['gradient_interaction']) is not dict or not metric['gradient_interaction']):
        raise ValueError('Every original joint objective/target-normalizer/active leaf/global gradient metric must reconstruct')
    value_digest(metric)
    return names


def scan(view,entries,completed,receipts,history,guard):
    if type(completed) is not int or not 0<=completed<=len(entries) or type(receipts) is not list or len(receipts)!=completed:
        raise ValueError('Exactly the complete committed original joint receipt prefix is required')
    clocks={n:0 for n,_ in view.named_selected()};readers={};previous_key=None;seen=set();sequence=0;previous=None
    source_chain=metric_chain=receipt_chain=value_digest(dict(format=FORMAT,complete_schedule_sha256=value_digest(entries)))
    logical=physical=0
    for index,envelope in enumerate(receipts):
        if (type(envelope) is not dict or set(envelope)!={'format','directory','header_sha256','record'} or envelope['format']!=ENVELOPE
                or not sha(envelope['header_sha256']) or type(envelope['directory']) is not str):raise ValueError('A complete independently hashed original joint receipt envelope is required')
        directory=Path(envelope['directory']);key=(str(directory),envelope['header_sha256'])
        if directory.parent!=Path(history['history_root']):raise ValueError('A receipt cannot import a different joint experiment directory')
        if key!=previous_key:
            if key in seen or len(seen)>=history['maximum_attempts']:raise ValueError('Original attempts cannot repeat, reorder or exceed their fixed bound')
            reader=Reader(directory,expected_header_sha256=key[1],maximum_compact_record_bytes=history['budgets']['maximum_compact_record_bytes'],
                maximum_expanded_record_bytes=history['budgets']['maximum_expanded_record_bytes'],guard=guard)
            if (reader.header['starting_completed']!=index or reader.budgets!=history['budgets']
                    or value_digest(reader.codec.reference)!=value_digest(history['binding'])):raise ValueError('Each original attempt must begin at the exact complete source prefix and binding')
            readers[key]=reader;seen.add(key);sequence=0;previous=reader.header_sha256
            logical=len(reader.codec.original_bytes);physical=reader.header_physical_bytes
        reader=readers[key];receipt=envelope['record'];sequence+=1
        if receipt.get('sequence')!=sequence or receipt.get('previous_receipt_sha256')!=previous:raise ValueError('No original within-attempt metric may be omitted, reordered or replaced')
        metric=reader.compact(receipt);names=validate_metric(view,entries[index],metric)
        for name in names:clocks[name]+=1
        previous=bytes_sha(canonical(receipt));previous_key=key
        logical+=receipt['original_expanded_bytes'];physical+=receipt['compressed_bytes']+len(canonical(receipt))
        if logical>history['budgets']['maximum_logical_bytes'] or physical>history['budgets']['maximum_physical_bytes']:
            raise ValueError('Every complete original attempt must retain its entire logical/physical quota')
        source_chain=value_digest([source_chain,value_digest(entries[index])]);metric_chain=value_digest([metric_chain,value_digest(metric)])
        receipt_chain=value_digest([receipt_chain,value_digest(envelope)])
    return dict(completed=completed,planned=len(entries),actual_per_parameter_Adam_steps=clocks,complete_schedule_sha256=value_digest(entries),
        consumed_source_prefix_chain_sha256=source_chain,all_original_metrics_chain_sha256=metric_chain,all_original_receipts_chain_sha256=receipt_chain,
        actual_committed_attempts=len(seen),next_entry_sha256=None if completed==len(entries) else value_digest(entries[completed]))


def make_state(view,optimizer,fixed,*,sources,entries,completed,history_receipts,history_configuration,source_files,resource_guard,
               rank=0,world=1,device='cpu',control_group=None):
    _,actual_world=control(control_group);error=common=state=None
    try:
        named=context(view,optimizer,fixed,sources,entries,rank,world,device,control_group)
        history=configuration(history_configuration,sources=sources,metadata=fixed.metadata,world=world,optimizer=optimizer,planned=len(entries))
        checked=SourceOnlyRecoveryGuard(source_files,checked_owner(resource_guard))
        boundary=scan(view,entries,completed,history_receipts,history,checked.check)
        if any(not bool(torch.isfinite(p).all()) for _,p in named):raise ValueError('All actual selected joint weights must be finite')
        adam=_cpu(optimizer.state_dict());validate_adam(adam,optimizer,named,boundary['actual_per_parameter_Adam_steps'])
        rng=_cpu(capture_rng(device));_validate_rng(rng,device)
        state=dict(format=FORMAT,sources=copy.deepcopy(sources),world_size=world,rank=rank,model_metadata=copy.deepcopy(fixed.metadata),cursor=boundary,
            history_configuration=copy.deepcopy(history),history_receipts=copy.deepcopy(history_receipts),optimized_names=[n for n,_ in named],
            selected_weights={n:_cpu(p) for n,p in named},optimizer=adam,private_RNG=rng,private_RNG_sha256=value_digest(rng),
            grants_production_training_admission=False,candidate_promoted=False,goal_achieved=False)
        common={k:value_digest(v) for k,v in state.items() if k not in ['rank','private_RNG','private_RNG_sha256']}
        checked.verify_final()
        resource_guard()
    except Exception as exc:error=type(exc).__name__+': '+str(exc)
    agree(error,common,control_group,actual_world)
    return state


def restore_state(view,optimizer,fixed,state,*,sources,entries,history_configuration,source_files,resource_guard,expected_common_sha256,
                  expected_private_RNG_sha256,expected_completed,rank=0,world=1,device='cpu',control_group=None,allow_completed=False):
    _,actual_world=control(control_group);error=common=boundary=None
    try:
        named=context(view,optimizer,fixed,sources,entries,rank,world,device,control_group)
        history=configuration(history_configuration,sources=sources,metadata=fixed.metadata,world=world,optimizer=optimizer,planned=len(entries))
        checked=SourceOnlyRecoveryGuard(source_files,checked_owner(resource_guard))
        boundary=scan(view,entries,state.get('cursor',{}).get('completed'),state.get('history_receipts'),history,checked.check)
        if (state.get('format')!=FORMAT or state.get('sources')!=sources or type(state.get('world_size')) is not int or state['world_size']!=world
                or type(state.get('rank')) is not int or state['rank']!=rank or state.get('model_metadata')!=fixed.metadata
                or state.get('history_configuration')!=history or state.get('cursor')!=boundary or type(expected_completed) is not int or boundary['completed']!=expected_completed
                or boundary['completed']==len(entries) and not allow_completed or state.get('optimized_names')!=[n for n,_ in named]
                or list(state.get('selected_weights',{}))!=[n for n,_ in named]
                or any(state.get(k) is not False for k in ['grants_production_training_admission','candidate_promoted','goal_achieved'])):
            raise ValueError('Exact original joint values/Adam/records/private RNG and complete source prefix are required before mutation')
        for name,parameter in named:
            value=state['selected_weights'][name]
            if not isinstance(value,torch.Tensor) or value.device.type!='cpu' or value.shape!=parameter.shape or value.dtype!=parameter.dtype or not bool(torch.isfinite(value).all()):
                raise ValueError('All complete finite selected joint weights must validate before recovery')
        validate_adam(state.get('optimizer'),optimizer,named,boundary['actual_per_parameter_Adam_steps']);_validate_rng(state.get('private_RNG'),device)
        if not sha(expected_private_RNG_sha256) or state.get('private_RNG_sha256')!=value_digest(state['private_RNG']) or state['private_RNG_sha256']!=expected_private_RNG_sha256:
            raise ValueError('The independently committed actual private generator must match')
        common=common_digest(state)
        if not sha(expected_common_sha256) or common!=expected_common_sha256:raise ValueError('Every selected value/Adam moment/original record receipt must match the external identity')
        checked.verify_final()
        resource_guard()
    except Exception as exc:error=type(exc).__name__+': '+str(exc)
    agree(error,common,control_group,actual_world)
    with torch.no_grad():
        for name,parameter in named:parameter.copy_(state['selected_weights'][name])
    optimizer.load_state_dict(_cpu(state['optimizer']));optimizer.zero_grad(set_to_none=True);restore_rng(state['private_RNG'],device);torch.clear_autocast_cache()
    fixed.check(view,optimizer,full=True)
    return boundary['completed'],copy.deepcopy(state['history_receipts'])
