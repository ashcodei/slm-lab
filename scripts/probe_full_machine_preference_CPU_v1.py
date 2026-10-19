"""Matched actual999MCPU machine-DPO/control learning and durable recovery.

Uses three whole, predeclared shortest admitted train pairs on an isolated
complete own-main copy. One chosen-code update makes the starting actor
nonzero; both two-example arms then start from those same actor values.
This is source-learning/mechanics research, never a held capability result or
a production initializer, full1620pair epoch, or substitute for GPU training.
"""
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch

from slm_lab.codenet_checkpoint_v1 import FileBindings, atomic_checkpoint, capture_rng
from slm_lab.corpus import file_hash
from slm_lab.dense_human_preference_v1 import sequence_logp
from slm_lab.diverse_images import atomic_json
from slm_lab.execution_contrast_data_v1 import ExecutionContrastData
from slm_lab.fast_joint_inference_v2 import load
from slm_lab.frozen_own_actor_reference_v1 import FrozenOwnActorReference, signature
from slm_lab.local_source_closure_v2 import python_sources
from slm_lab.machine_code_preference_stage_v1 import MachineCodePreferenceStage
from slm_lab.own_dense_actor_v1 import OwnDenseActor
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/full-machine-preference-CPU-v1'
SAVE=ROOT/'checkpoints/full-machine-preference-CPU-v1'


def equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a) is type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


@contextmanager
def neural_calls(model,reference):
    counts=dict(policy=0,reference=0)
    def policy_hook(*args):counts['policy']+=1
    def reference_hook(*args):counts['reference']+=1
    a=model.register_forward_hook(policy_hook);b=reference.register_forward_hook(reference_hook)
    try:yield counts
    finally:a.remove();b.remove()


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists() or SAVE.exists():
        raise ValueError('A fresh isolated CPU-only matched machine preference experiment is required')
    tests=ROOT/'reports/machine-code-preference-stage-v1-tests.log'
    if '4 passed' not in tests.read_text():
        raise ValueError('Actual independent-gradient/control/normalization/durable-recovery/source corruption tests must pass')
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    data=ExecutionContrastData(ROOT,purpose='optimization')
    positions=sorted(range(len(data)),key=lambda p:(len(data.rows[p]['chosen_ids'])+len(data.rows[p]['rejected_ids']),p))[:3]
    if len(data)!=1620 or positions!=[873,396,27]:
        raise ValueError('All1620admitted sources and the predeclared three shortest whole contexts must remain exact')
    checkpoint=ROOT/'checkpoints/phase-joint-v3/trained/model.pt';main_plan=ROOT/'reports/phase-joint-v3/plan.json'
    previous=ROOT/'reports/completed-phase-joint-v3-CPU-readback-v1/complete.json';proof=json.loads(previous.read_text())
    inputs=dict(checkpoint=str(checkpoint),plan=str(main_plan),checkpoint_sha256=proof['source_files_sha256'][str(checkpoint)],
                plan_sha256=proof['source_files_sha256'][str(main_plan)])
    files={str(checkpoint):inputs['checkpoint_sha256'],str(main_plan):inputs['plan_sha256']}
    for p in [previous,tests,data.folder/'admission.json',data.folder/'manifest.json',data.folder/'pairs.jsonl',data.tokenizer_path,
              *python_sources(ROOT,[Path(__file__),ROOT/'tests/test_machine_code_preference_stage_v1.py'])]:files[str(p)]=file_hash(p)
    guard=FileBindings(files);OUT.mkdir();SAVE.mkdir();started=time.monotonic()
    owner=dict(pid=os.getpid(),ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19])
    options=dict(method='machine_DPO',seed=992417,global_batch_size=1,learning_rate=1e-6,
                 betas=[.9,.95],weight_decay=.01,gradient_clip=1.,beta=.1,chosen_CE_weight=.1)
    plan=dict(at=datetime.now(timezone.utc).isoformat(),owner=owner,source_files_sha256=files,starting_main_inputs=inputs,
        own_actor_settings=dict(rank=16,seed=992317,parameter_limit=10**9),training_options=options,
        warm_chosen_source_position=positions[0],matched_control_and_DPO_positions=positions[1:],
        selected_sources_by_whole_native_context_length_before_any_neural_score=True,
        actual_admitted_source_population=1620,full_source_epoch_complete=False,
        own_nonzero_starting_actor_reference_required=True,durable_actual_after1_next_update_replay_required=True,
        original_live_GPU_run_and_all_queued_models_unchanged=True,
        no_held_generation_broad_visual_quality_or_production_initializer_claim=True,
        CPU_threads=2,production_updates=0,goal_achieved=False)
    atomic_json(OUT/'plan.json',plan)
    sources=dict(own_complete_main_checkpoint_sha256=inputs['checkpoint_sha256'],
                 admitted_machine_preferences_sha256=file_hash(data.folder/'admission.json'),experiment_plan_sha256=file_hash(OUT/'plan.json'))
    def progress(phase,**extra):
        atomic_json(OUT/'progress.json',dict(owner=owner,phase=phase,elapsed_seconds=time.monotonic()-started,
            actual_CUDA_initialized=torch.cuda.is_initialized(),production_updates=0,goal_achieved=False,**extra))
    def optimizer(model):
        return torch.optim.AdamW([p for _,p in model.named_selected(('actor',))],lr=options['learning_rate'],
            betas=tuple(options['betas']),weight_decay=options['weight_decay'],foreach=False)
    def draws():return [random.random(),float(np.random.random()),float(torch.rand(()))]
    def adam(opt):
        return tensor_digest((str(i)+'.'+key,state[key]) for i,state in enumerate(opt.state.values()) for key in ['step','exp_avg','exp_avg_sq'])
    try:
        progress('loading complete own984Mmain and bounded actor')
        base,_=load(checkpoint,main_plan,'trained',expected_checkpoint_sha256=inputs['checkpoint_sha256'],
                    expected_plan_sha256=inputs['plan_sha256'],device='cpu')
        model=OwnDenseActor(base,**plan['own_actor_settings']).eval()
        if model.counts()['actual_unique']!=999706442:raise ValueError('The actual whole counted student must remain under1B')
        frozen=tensor_digest(model.base.named_parameters());critic=model.parts['critic'].weight.detach().clone()
        initial_actor=tensor_digest(model.parts.named_parameters())
        random.seed(992917);np.random.seed(992917);torch.manual_seed(992917)
        initial_reference=FrozenOwnActorReference(model,sources=sources)
        warm=MachineCodePreferenceStage(model,optimizer(model),data,initial_reference,dict(options,method='chosen_CE'),sources,CPU_order=[positions[0]])
        progress('one actual whole chosen-code warm update to create a nonzero starting actor')
        with neural_calls(model,initial_reference) as warm_calls:warm_record=warm.update()
        if warm_calls!=dict(policy=1,reference=0) or tensor_digest(model.parts.named_parameters())==initial_actor:
            raise ValueError('Actual nonzero source-trained warm actor and no rejected/reference control forwards required')
        warm_parts={name:value.detach().clone() for name,value in model.parts.state_dict().items()}
        del warm,initial_reference;gc.collect()
        reference=FrozenOwnActorReference(model,sources=sources);binding=reference.binding()
        if binding['frozen_actor_parts_sha256']==initial_actor:
            raise ValueError('The actual own starting reference must retain nonzero learned adapters')
        atomic_checkpoint(SAVE/'private-own-starting-actor-reference.pt',dict(binding=binding,parts=warm_parts),512*1024**2,32*1024**3)
        ref_digest=tensor_digest(reference.parts.named_parameters())
        @torch.no_grad()
        def measure():
            model.eval();rows=[]
            for position in positions[1:]:
                values={};original={};counts={};logits_hashes={}
                for field in ['chosen','rejected']:
                    batch=data.batch([position],field,'cpu');inputs={k:v for k,v in batch.items() if k!='labels'}
                    logits=model(**inputs)['logits'];logits_hashes[field]=tensor_digest([(field,logits)])
                    p,count=sequence_logp(logits,batch['labels']);q,again=sequence_logp(reference(**inputs)['logits'],batch['labels'])
                    if count!=again:raise ValueError('All whole policy/reference targets and EOS must match')
                    values[field]=p;original[field]=q;counts[field]=count
                margin=(values['chosen']-values['rejected'])-(original['chosen']-original['rejected'])
                rows.append(dict(position=position,policy_chosen_logp=float(values['chosen']),policy_rejected_logp=float(values['rejected']),
                    reference_chosen_logp=float(original['chosen']),reference_rejected_logp=float(original['rejected']),
                    own_reference_margin=float(margin),DPO_loss=float(torch.nn.functional.softplus(-options['beta']*margin)),
                    chosen_token_NLL=-float(values['chosen'])/counts['chosen'],whole_input_logit_sha256=logits_hashes))
            return rows
        progress('measuring both whole source pairs before either matched training arm')
        baseline=measure();atomic_json(OUT/'baseline-source-measurements.json',baseline)
        dpo=MachineCodePreferenceStage(model,optimizer(model),data,reference,options,sources,CPU_order=positions[1:])
        with neural_calls(model,reference) as DPO_calls:
            progress('first actual whole-source machine DPO update')
            first=dpo.update();saved=dpo.state();private=SAVE/'machine-DPO-after1.pt'
            atomic_checkpoint(private,saved,512*1024**2,32*1024**3);saved_sha=file_hash(private);del saved;gc.collect()
            expected_draws=draws();progress('second actual whole-source machine DPO update')
            expected=dpo.update();expected_weights=tensor_digest(model.parts.named_parameters());expected_adam=adam(dpo.optimizer);expected_rng=capture_rng('cpu')
            saved=torch.load(private,map_location='cpu',weights_only=True,mmap=True);rejections=[]
            for kind in ['original_machine_source','private_CPU_RNG','own_starting_reference']:
                bad=dict(saved,core=dict(saved['core']))
                if kind=='original_machine_source':
                    bad['core']['metrics']=[dict(r) for r in bad['core']['metrics']]
                    bad['core']['metrics'][0]['original_source_positions']=[-1]
                elif kind=='private_CPU_RNG':
                    bad['core']['rng']=dict(bad['core']['rng'],torch=torch.zeros(4,dtype=torch.uint8))
                else:bad['own_reference_binding']=dict(saved['own_reference_binding'],frozen_actor_parts_sha256='0'*64)
                before=signature(model.parameters())
                try:dpo.resume(bad)
                except (ValueError,RuntimeError) as error:rejections.append(dict(kind=kind,error_type=type(error).__name__))
                else:raise ValueError('Corrupt source/private RNG/own reference must reject before recovery mutation')
                if (signature(model.parameters())!=before or tensor_digest(model.parts.named_parameters())!=expected_weights
                        or adam(dpo.optimizer)!=expected_adam or not equal(capture_rng('cpu'),expected_rng)):
                    raise ValueError('Rejected recovery changed live actor, optimizer or private CPU RNG')
            progress('replaying the actual second machine source from the durable after1 checkpoint')
            dpo.resume(saved)
            if draws()!=expected_draws or dpo.update()!=expected or tensor_digest(model.parts.named_parameters())!=expected_weights or adam(dpo.optimizer)!=expected_adam or not equal(capture_rng('cpu'),expected_rng):
                raise ValueError('Actual next complete source update, every actor/Adam value and private CPU RNG must replay exactly')
            if file_hash(private)!=saved_sha:raise ValueError('Recovery mutated its retained immutable checkpoint file')
        if DPO_calls!=dict(policy=6,reference=6):raise ValueError('Every actual DPO/replay policy and reference forward must be accounted for')
        progress('measuring machine-DPO source outcomes')
        DPO_after=measure();DPO_payload=model.export(sources)
        atomic_checkpoint(SAVE/'machine-DPO-policy.pt',DPO_payload,512*1024**2,32*1024**3)
        DPO_records=list(dpo.metrics);del dpo,saved;gc.collect()
        with torch.no_grad():
            for name,p in model.named_selected(('actor',)):p.copy_(warm_parts[name])
        control=MachineCodePreferenceStage(model,optimizer(model),data,reference,dict(options,method='chosen_CE'),sources,CPU_order=positions[1:])
        progress('matched chosen-code control from the same exact nonzero starting actor')
        with neural_calls(model,reference) as control_calls:
            control_records=[control.update(),control.update()]
        if control_calls!=dict(policy=2,reference=0):raise ValueError('Chosen-code control must make no rejected or reference neural queries')
        control_after=measure();control_payload=model.export(sources)
        atomic_checkpoint(SAVE/'chosen-CE-policy.pt',control_payload,512*1024**2,32*1024**3)
        for label,payload,expected_rows in [('chosen_CE',control_payload,control_after),('machine_DPO',DPO_payload,DPO_after)]:
            progress('cold importing actual selected actor values and comparing every whole-source logit',arm=label)
            with torch.no_grad():
                for _,p in model.named_selected(('actor',)):p.add_(.01)
            raw=torch.load(SAVE/('chosen-CE-policy.pt' if label=='chosen_CE' else 'machine-DPO-policy.pt'),map_location='cpu',weights_only=True,mmap=True)
            model.import_export(raw,expected_sources=sources)
            if any(not torch.equal(raw['parts'][name],value) for name,value in model.parts.state_dict().items()) or measure()!=expected_rows:
                raise ValueError('Every actual stored actor value and complete short-source output must cold-import exactly on the unchanged shared base')
        if (tensor_digest(model.base.named_parameters())!=frozen or not torch.equal(model.parts['critic'].weight,critic)
                or tensor_digest(reference.parts.named_parameters())!=ref_digest or torch.cuda.is_initialized()):
            raise ValueError('Every original frozen dense/head/critic/reference value and CPU-only scope must remain exact')
        guard.check(full=True);data.guard.check(full=True)
        actual=dict(warm_chosen_CE=warm_record,machine_DPO=DPO_records,chosen_CE=control_records)
        atomic_json(OUT/'all-actual-update-records.json',actual)
        atomic_json(OUT/'complete.json',dict(at=datetime.now(timezone.utc).isoformat(),complete=True,
            actual_unique_exported_student_values=999706442,actual_frozen_dense_values=984399690,
            actual_warm_chosen_updates=1,actual_machine_DPO_logical_updates=2,actual_machine_DPO_physical_optimizer_calls=3,
            actual_matched_chosen_CE_updates=2,actual_selected_source_positions=positions,
            actual_policy_reference_forward_counts=dict(warm=warm_calls,machine_DPO=DPO_calls,chosen_CE=control_calls),
            actual_nonzero_own_starting_actor_reference_retained=True,
            actual_durable_next_source_actor_Adam_private_CPU_RNG_replay_exact=True,
            actual_invalid_state_rejections=rejections,retained_durable_checkpoint_bytes_unchanged=True,
            actual_baseline_source_measurements=baseline,actual_machine_DPO_after=DPO_after,actual_chosen_CE_after=control_after,
            every_actor_export_value_and_whole_short_source_output_cold_import_exact=True,
            every_original_dense_vision_word_auxiliary_critic_and_reference_value_unchanged=True,
            frozen_visual_encoder_bytes_do_not_prove_unchanged_visual_answers=True,
            full1620pair_epoch_and_GPU_capacity_and_held_capability_not_proved=True,
            original_production_pipeline_and_queued_models_unchanged=True,
            source_files_sha256={str(p):file_hash(p) for p in [OUT/'plan.json',OUT/'baseline-source-measurements.json',
                OUT/'all-actual-update-records.json',*sorted(SAVE.iterdir())]},
            production_updates=0,candidate_promoted=False,goal_achieved=False))
        print(json.dumps(dict(complete=str(OUT/'complete.json'),baseline=baseline,machine_DPO_after=DPO_after,
                              chosen_CE_after=control_after,production_updates=0)),flush=True)
    except BaseException as error:
        atomic_json(OUT/'failure.json',dict(owner=owner,error=repr(error),traceback=traceback.format_exc(),
            original_production_and_queues_untouched=True,production_updates=0,goal_achieved=False));raise


if __name__=='__main__':main()
