"""Five actual full-size phase controls, not the467M-target production pass."""
from datetime import datetime,timezone
import copy
import gc
import json
import os
from pathlib import Path
import random
import time

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports/phase-joint-full-CPU-v2'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Hide all GPUs for the CPU controls')
    if REPORTS.exists():raise ValueError('Preserve earlier complete/partial actual controls')
    import numpy as np
    import torch
    from slm_lab.codenet_checkpoint_v1 import FileBindings,atomic_checkpoint,capture_rng
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    from slm_lab.expanded_joint_sources_v1 import ExpandedJointSources
    from slm_lab.fast_joint_checkpoint_v2 import bundle,restore
    from slm_lab.fast_joint_inference_v2 import fresh,export
    from slm_lab.fast_joint_stage_v1 import make_optimizer,parameter_groups
    from slm_lab.phase_joint_update_v1 import update
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);started=time.monotonic()
    for name in ['phase-joint-update-v1','expanded-joint-sources-v1']:
        if '2 passed' not in (ROOT/'reports'/(name+'-tests.log')).read_text():raise ValueError('Actual objective/source tests required')
    prior_path=ROOT/'reports/protected-joint-v1/plan.json';prior=json.loads(prior_path.read_text())
    model,_=fresh(prior,'cpu');data=ExpandedJointSources(ROOT,model.config)
    names=['variety','facts:human_fact_questions','later:complete_JS_parameter_headers','images:additional_photo_captions','words']
    entries=[(name,[min(range(len(data.lengths[name])),key=lambda p:data.lengths[name][p])]) for name in names]
    class Schedule:
        phase_for_step=['foundation']+['instruction_and_vision']*4
        phase_bounds=dict(foundation=[0,1],instruction_and_vision=[1,5])
        def __len__(self):return 5
        def at(self,step):return entries[step]
    schedule=Schedule();phase_rates=dict(foundation=dict(language=1e-5,vision=1e-4,word=3e-4),
        instruction_and_vision=dict(language=1e-5,vision=1e-4,word=3e-4))
    options=dict(learning_rates=phase_rates['foundation'],phase_learning_rates=phase_rates,betas=[.9,.95],weight_decay=.01,
        gradient_clip=1.,warmup_fraction=.02,cosine_floor=.1,maximum_microbatch=8,maximum_microbatch_tokens=8192,
        ctc_weight=1.,word_native_CE_weight=1.,gradient_bucket_values=16*1024**2,image_size=448,reference_KL_weight=.01)
    files=[Path(__file__),prior_path,ROOT/'src/slm_lab/phase_joint_update_v1.py',ROOT/'src/slm_lab/expanded_joint_sources_v1.py',
        ROOT/'reports/expanded-joint-sources-v1/complete.json',ROOT/'reports/phase-joint-update-v1-tests.log',ROOT/'reports/expanded-joint-sources-v1-tests.log']
    bindings={str(p):file_hash(p) for p in files};bindings.update({p:d for p,d in prior['source_files_sha256'].items() if Path(p).suffix=='.py'})
    guard=FileBindings(bindings);REPORTS.mkdir()
    control_budget={name:dict(examples=1,complete_context_tokens=data.lengths[name][p[0]],response_EOS_targets=data.targets[name][p[0]],kind=data.kinds[name]) for name,p in entries}
    plan=dict(initial_dense_settings=prior['initial_dense_settings'],actual_counts=model.counts(),planned_steps=5,training_options=options,
        complete_source_budget=control_budget,available_full_original_source_budget=data.budget(),actual_control_batches=entries,
        phase_bounds=schedule.phase_bounds,source_files_sha256=bindings,all_original_sources_once=False,all_selected_control_sources_once=True,
        full_original_source_epoch_and_GPU_capacity_not_tested=True,all_updates_disposable=True,goal_achieved=False)
    atomic_json(REPORTS/'plan.json',plan);digest=file_hash(REPORTS/'plan.json');optimizer=make_optimizer(model,options,'cpu')
    random.seed(958319);np.random.seed(958319);torch.manual_seed(958319);records=[];reference=None
    groups=parameter_groups(model);selected={id(p) for rows in groups.values() for _,p in rows}
    unused=tensor_digest((n,p) for n,p in model.named_parameters() if id(p) not in selected)
    def phase(name):atomic_json(REPORTS/'progress.json',dict(phase=name,actual_logical_control_updates=len(records),seconds=time.monotonic()-started,production_updates=0,goal_achieved=False))
    def step(i):
        draws=[random.random(),float(np.random.random()),float(torch.rand(()))]
        r=update(model,optimizer,data,schedule,i,options,reference=reference)
        r['diagnostic_CPU_RNG_draws']=draws;return r
    phase('raw complete original CodeNet source without reference');records.append(step(0))
    reference=copy.deepcopy(model).eval().requires_grad_(False);ref_digest=tensor_digest(reference.named_parameters())
    reference_path=REPORTS/'own-foundation-reference.pt';atomic_checkpoint(reference_path,
        reference.export(dict(own_control_plan_sha256=digest)),5*1024**3,26*1024**3)
    ref_binding=dict(actual_foundation_completed_updates=1,path=str(reference_path),sha256=file_hash(reference_path),actual_frozen_reference_sha256=ref_digest)
    phase('complete original human factual QA against newly trained own foundation');records.append(step(1))
    state=bundle(model,optimizer,plan,digest,2,records,[capture_rng('cpu')],'train');state['own_reference_binding']=ref_binding
    atomic_checkpoint(REPORTS/'resume-step2.pt',state,12*1024**3,26*1024**3);del state
    phase('complete original JavaScript function/interface with positive own-foundation KL');expected=step(2)
    expected_weights=tensor_digest(model.named_parameters())
    def adam():return tensor_digest((str(i)+'.'+k,s[k]) for i,s in enumerate(optimizer.state.values()) for k in ('step','exp_avg','exp_avg_sq'))
    expected_adam=adam();payload=torch.load(REPORTS/'resume-step2.pt',map_location='cpu',weights_only=True,mmap=True)
    if payload['own_reference_binding']!=ref_binding or ref_digest!=tensor_digest(reference.named_parameters()):raise ValueError('Exact newly trained own foundation reference required')
    completed,records=restore(model,optimizer,payload,plan,digest,'cpu',0,1,'train');del payload
    if completed!=2:raise ValueError('Actual phase cursor must restore')
    phase('exact next phase source/weights/Adam/RNG recovery');repeated=step(2)
    if (expected!=repeated or expected_weights!=tensor_digest(model.named_parameters()) or expected_adam!=adam()
            or expected['full_context_reference_KL_sum']<=0):raise ValueError('Actual next whole-source weights/Adam/RNG/positive KL must recover exactly')
    records.append(repeated);language=tensor_digest(groups['language'])
    phase('complete regular photo through fixed decoder');records.append(step(3))
    phase('complete original word crop with CTC and native CE through fixed decoder');records.append(step(4))
    if (language!=tensor_digest(groups['language']) or ref_digest!=tensor_digest(reference.named_parameters())
            or unused!=tensor_digest((n,p) for n,p in model.named_parameters() if id(p) not in selected)):
        raise ValueError('Image/word controls cannot change language, own foundation reference or unused heads')
    final=export(model,plan,digest,'trained',5);atomic_checkpoint(REPORTS/'model.pt',final,5*1024**3,26*1024**3)
    state=bundle(model,optimizer,plan,digest,5,records,[capture_rng('cpu')],'train');state['own_reference_binding']=ref_binding
    atomic_checkpoint(REPORTS/'resume-step5.pt',state,12*1024**3,26*1024**3);del state
    ids=torch.tensor([[1,261,267,71,2]]);pixels=torch.zeros(1,3,448,448);words=torch.zeros(1,3,32,224)
    def outputs(m):
        m.eval()
        with torch.no_grad():return dict(text=m(ids)['logits'].clone(),image=m(ids,pixel_values=pixels)['logits'].clone(),word=m.word_logits(words).clone())
    phase('trained native full-size output controls');expected_outputs=outputs(model);counts=model.counts();states=len(optimizer.state)
    del model,optimizer,reference,groups;gc.collect();phase('fresh counted native text/image/word cold reload')
    cold,_=fresh(plan,'cpu')
    from slm_lab.fast_joint_inference_v2 import sources
    cold.import_export(final['model'],expected_sources=sources(plan,digest));actual=outputs(cold)
    if any(not torch.equal(expected_outputs[k],actual[k]) for k in actual):raise ValueError('Every actual trained native cold output must match')
    guard.check(full=True)
    result=dict(at=datetime.now(timezone.utc).isoformat(),actual_counts=counts,actual_logical_control_updates=5,actual_physical_optimizer_calls=6,
        complete_Adam_states=states,records=records,exact_next_phase_weights_Adam_RNG_and_metrics=True,
        actual_own_trained_foundation_reference_binding=ref_binding,positive_later_full_context_KL=True,
        raw_foundation_reference_not_queried=True,decoder_bytes_exact_during_photo_and_word=True,all_unused_heads_and_reference_exact=True,
        trained_native_text_image_word_cold_reload_exact=True,source_files_sha256=bindings,seconds=time.monotonic()-started,
        available_full_source_epoch_not_trained=True,full_profile_GPU_capacity_still_required=True,production_updates=0,quality_gain_proven=False,goal_achieved=False)
    atomic_json(REPORTS/'complete.json',result);print(json.dumps(dict(passed=True,logical_control_updates=5,Adam_states=states,seconds=result['seconds'])),flush=True)


if __name__=='__main__':main()
