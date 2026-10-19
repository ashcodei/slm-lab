"""Phase/reference bindings and exact active-group Adam clocks before recovery."""
from collections import Counter

from .corpus import file_hash
from .fast_joint_checkpoint_v2 import bundle,restore
from .whole_human_sft_evidence_v1 import tensor_digest

FORMAT='own-complete-foundation-then-instruction-reference-checkpoint-v1'


def bounds(plan,mode):return plan['capacity_phase_bounds'] if mode=='capacity' else plan['phase_bounds']


def validate_reference(plan,mode,completed,binding,reference):
    boundary=bounds(plan,mode)['foundation'][1]
    if completed<boundary:
        if binding is not None or reference is not None:raise ValueError('Raw foundation snapshot must have no reference copy or binding')
        return
    if (not isinstance(binding,dict) or set(binding)!={'path','checkpoint_sha256','parameter_sha256','foundation_completed_updates'}
            or binding['foundation_completed_updates']!=boundary or reference is None or reference.training
            or any(p.requires_grad or p.grad is not None for p in reference.parameters())
            or file_hash(binding['path'])!=binding['checkpoint_sha256']
            or tensor_digest(reference.named_parameters())!=binding['parameter_sha256']):
        raise ValueError('Exact immutable newly trained own foundation reference required before recovery')


def validate_clocks(core,plan,mode):
    completed=core.get('completed');limits=bounds(plan,mode)
    if type(completed) is not int or not 0<completed<=limits['instruction_and_vision'][1] or len(core.get('metrics',[]))!=completed:
        raise ValueError('Exact completed phase cursor and every original update record required')
    expected=Counter()
    for i,r in enumerate(core['metrics']):
        phase=next((name for name,(a,b) in limits.items() if a<=i<b),None)
        if (phase is None or r.get('step')!=i+1 or r.get('phase')!=phase
                or r.get('phase_step')!=i-limits[phase][0]+1 or r.get('phase_planned_steps')!=limits[phase][1]-limits[phase][0]):
            raise ValueError('Every phase-local optimizer cursor must reconstruct before mutation')
        family=r['family'];kind=plan['complete_source_budget'][family]['kind']
        active=['language'] if kind=='text' else ['vision']+(['word'] if kind=='word' else [])
        if (r.get('actual_active_groups')!=active or phase=='foundation' and kind!='text'
                or r.get('reference_forward_queried')!=(phase=='instruction_and_vision' and kind=='text')):
            raise ValueError('Actual source-family active groups and reference-query decisions must reconstruct')
        expected.update(active)
    adam=core['optimizer']
    for group in adam['param_groups']:
        steps=expected[group['name']]
        for key in group['params']:
            state=adam['state'].get(key)
            if (steps==0 and state is not None or steps>0 and (state is None or float(state['step'])!=steps)):
                raise ValueError('Every and only active family Adam clock must match its real updates')


def make_state(model,optimizer,plan,digest,completed,metrics,rngs,mode,*,reference_binding=None,reference=None):
    validate_reference(plan,mode,completed,reference_binding,reference)
    core=bundle(model,optimizer,plan,digest,completed,metrics,rngs,mode);validate_clocks(core,plan,mode)
    return dict(format=FORMAT,phase_bounds=bounds(plan,mode),reference_binding=reference_binding,core=core,
        reference_is_training_only_own_foundation=True,goal_achieved=False)


def restore_state(model,optimizer,state,plan,digest,device,rank,world,mode,*,reference_binding=None,reference=None):
    core=state.get('core',{})
    if (state.get('format')!=FORMAT or state.get('phase_bounds')!=bounds(plan,mode)
            or state.get('reference_binding')!=reference_binding or state.get('reference_is_training_only_own_foundation') is not True
            or state.get('goal_achieved') is not False):raise ValueError('Exact phase/reference checkpoint profile required')
    validate_clocks(core,plan,mode)
    validate_reference(plan,mode,core['completed'],reference_binding,reference)
    # The underlying complete snapshot validator checks every model/Adam/RNG
    # before copying any live parameter. Reference validation above is read-only.
    return restore(model,optimizer,core,plan,digest,device,rank,world,mode)
