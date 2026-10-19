"""Exact new-source continuation cursor and existing full-state recovery.

All updates use a frozen copy of the complete own starting policy. There is
no new unanchored foundation phase. The original phase checkpoint machinery
still validates every group clock, reference, complete model, Adam and RNG.
"""
from .focused_code_continuation_data_v3 import CODE,EXECUTION,STATE,FocusedContinuationSchedule
from .phase_joint_checkpoint_v1 import make_state as phase_state,restore_state as phase_restore


def capacity_batches(data,width=16):
    def longest(name,field='lengths'):
        values=getattr(data,field)[name]
        return [name,sorted(range(len(values)),key=lambda p:(values[p],p),reverse=True)[:width]]
    return [longest(CODE),longest(CODE,'targets'),longest(EXECUTION),longest(STATE),
        longest('packed:original_packed_code'),longest('education'),
        longest('images:additional_photo_captions'),longest('words')]


def schedule_for(data,plan,mode):
    if mode not in {'capacity','train'} or data.purpose!='optimization':raise ValueError('An admitted production source and explicit mode required')
    schedule=FocusedContinuationSchedule(data,seed=plan['seed'],width=plan['global_batch_width'])
    if (schedule.identity_sha256!=plan['source_schedule_sha256'] or len(schedule)!=plan['planned_steps']
            or schedule.selected_budget(data)!=plan['selected_source_budget'] or data.budget()!=plan['complete_source_budget']):
        raise ValueError('Every admitted new/replay source, whole-context target and selected position must equal the frozen plan')
    if mode=='capacity':schedule.entries=[(n,list(p)) for n,p in plan['capacity_global_batches']]
    schedule.phase_for_step=['instruction_and_vision']*len(schedule.entries)
    schedule.phase_bounds=dict(foundation=[0,0],instruction_and_vision=[0,len(schedule.entries)])
    expected=plan['capacity_phase_bounds'] if mode=='capacity' else plan['phase_bounds']
    if schedule.phase_bounds!=expected:raise ValueError('Exact zero-foundation anchored continuation bounds required')
    return schedule


def validate_metrics(metrics,schedule):
    if not isinstance(metrics,list) or not 0<len(metrics)<=len(schedule):raise ValueError('A nonempty actual prefix of the exact source schedule required')
    for i,row in enumerate(metrics):
        family,positions=schedule.at(i)
        if (row.get('step')!=i+1 or row.get('family')!=family or row.get('global_original_positions')!=positions
                or row.get('global_examples')!=len(positions)):
            raise ValueError('Every actual source-family cursor and every original global position must reconstruct before mutation')


def make_state(model,optimizer,plan,digest,completed,metrics,rngs,mode,*,schedule,reference_binding,reference):
    if completed!=len(metrics):raise ValueError('Every actual update record required')
    validate_metrics(metrics,schedule)
    return phase_state(model,optimizer,plan,digest,completed,metrics,rngs,mode,
        reference_binding=reference_binding,reference=reference)


def restore_state(model,optimizer,state,plan,digest,device,rank,world,mode,*,schedule,reference_binding,reference):
    validate_metrics(state.get('core',{}).get('metrics'),schedule)
    return phase_restore(model,optimizer,state,plan,digest,device,rank,world,mode,
        reference_binding=reference_binding,reference=reference)
