"""Real state-source backward and exact in-memory recovery, on an own CPU copy."""
import random
import math

import numpy as np
import torch
from torch.nn import functional as F

from .codenet_checkpoint_v1 import capture_rng
from .fast_joint_stage_v1 import make_optimizer,parameter_groups
from .focused_continuation_run_contract_v2 import make_state,restore_state
from .phase_joint_update_v1 import update
from .whole_human_sft_evidence_v1 import tensor_digest


def equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a) is type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


@torch.no_grad()
def loss(model,batch):
    model.eval();kwargs={k:batch[k] for k in ('input_ids','attention_mask','pixel_values') if k in batch}
    logits=model(**kwargs)['logits'];targets=batch['labels'][:,1:];mask=targets!=-100
    return float(F.cross_entropy(logits[:,:-1].float()[mask],targets[mask],reduction='mean'))


def rehearse(model,data,schedule,plan,digest,*,reference,reference_binding,progress=lambda **row:None):
    if (next(model.parameters()).device.type!='cpu' or len(schedule)!=2
            or any(data.kinds[schedule.at(i)[0]]!='text' or len(schedule.at(i)[1])!=1 for i in range(2))):
        raise ValueError('Two whole single-source text updates on a private CPU model required')
    groups=parameter_groups(model);language=groups['language'];selected={id(p) for _,p in language}
    fixed=[(n,p) for n,p in model.named_parameters() if id(p) not in selected]
    initial={n:tensor_digest([(n,p)]) for n,p in language};fixed_digest=tensor_digest(fixed)
    reference_digest=tensor_digest(reference.named_parameters());optimizer=make_optimizer(model,plan['training_options']);rows=[]
    def moments():return tensor_digest((str(i)+'.'+k,s[k]) for i,s in enumerate(optimizer.state.values()) for k in ('step','exp_avg','exp_avg_sq'))
    def step(i):
        name,positions=schedule.at(i);progress(phase='actual full state backward',logical_step=i+1,source_position=positions[0],
            complete_context_positions=data.lengths[name][positions[0]],response_EOS_targets=data.targets[name][positions[0]])
        row=update(model,optimizer,data,schedule,i,plan['training_options'],reference=reference,device='cpu')
        if row['actual_active_groups']!=['language'] or row['active_parameter_tensors']!=len(language):
            raise ValueError('Every and only actual dense language leaf must participate in state learning')
        rows.append(row);return row
    def draws():return [random.random(),float(np.random.random()),float(torch.rand(()))]
    step(0);progress(phase='copying complete private recovery state',logical_updates=1)
    saved=make_state(model,optimizer,plan,digest,1,rows,[capture_rng('cpu')],'train',
        schedule=schedule,reference_binding=reference_binding,reference=reference)
    expected_draws=draws();expected=step(1);expected_rng=capture_rng('cpu')
    expected_weights=tensor_digest(model.named_parameters());expected_moments=moments();rejections=[]
    for kind in ['source_position','private_CPU_RNG']:
        core=dict(saved['core']);bad=dict(saved,core=core)
        if kind=='source_position':
            core['metrics']=[dict(r) for r in core['metrics']];core['metrics'][0]['global_original_positions']=[-1]
        else:
            core['rngs']=[dict(r) for r in core['rngs']];core['rngs'][0]['torch']=torch.zeros(4,dtype=torch.uint8)
        signature=[(id(p),p._version,p.data_ptr(),p.requires_grad) for p in model.parameters()];rng=capture_rng('cpu')
        try:restore_state(model,optimizer,bad,plan,digest,'cpu',0,1,'train',schedule=schedule,reference_binding=reference_binding,reference=reference)
        except (ValueError,RuntimeError) as error:rejections.append(dict(kind=kind,error_type=type(error).__name__))
        else:raise ValueError('Corrupt source or private CPU RNG must reject before recovery mutation')
        if (signature!=[(id(p),p._version,p.data_ptr(),p.requires_grad) for p in model.parameters()]
                or not equal(rng,capture_rng('cpu')) or moments()!=expected_moments):
            raise ValueError('Rejected recovery changed live values, Adam or private RNG')
    progress(phase='restoring actual state before repeated longest-source update',logical_updates=1)
    count,rows=restore_state(model,optimizer,saved,plan,digest,'cpu',0,1,'train',schedule=schedule,reference_binding=reference_binding,reference=reference)
    if count!=1 or draws()!=expected_draws or step(1)!=expected or tensor_digest(model.named_parameters())!=expected_weights or moments()!=expected_moments or not equal(expected_rng,capture_rng('cpu')):
        raise ValueError('Actual complete next source update, weights, Adam and private CPU RNG must replay exactly')
    if fixed_digest!=tensor_digest(fixed) or reference_digest!=tensor_digest(reference.named_parameters()):raise ValueError('Every nonlanguage own head and frozen reference must stay exact')
    changed=[n for n,p in language if tensor_digest([(n,p)])!=initial[n]]
    if not changed or len(optimizer.state)!=len(language) or any(float(s['step'])!=2 for s in optimizer.state.values()):raise ValueError('Real dense state learning and every active Adam clock required')
    name,positions=schedule.at(1);progress(phase='measuring same-source after-update loss',logical_updates=2)
    after=loss(model,data.batch(name,positions,'cpu'))
    if not math.isfinite(after):raise ValueError('Finite actual same-source after-update loss required')
    return dict(actual_logical_updates=2,physical_optimizer_calls=3,actual_active_language_tensors=len(language),
        actual_changed_language_tensor_names=changed,actual_full_language_Adam_states=len(optimizer.state),
        actual_update_records=rows,all_nonlanguage_heads_and_reference_parameters_exact=True,
        actual_in_memory_next_update_weights_Adam_and_private_CPU_RNG_exact=True,
        invalid_state_rejections=rejections,rejected_recovery_preserves_live_values_Adam_and_RNG=True,
        longest_source_pre_update_NLL=expected['token_mean_CE'],longest_source_post_update_NLL=after,
        source_training_loss_is_not_held_code_or_visual_quality=True,durable_optimizer_snapshot_not_written_or_certified=True)
