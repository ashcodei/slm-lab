"""Two complete original-source protected updates with full-size recovery.

Short real controls are not full4096-token GPU capacity or a quality result.
The frozen reference is an extra 984M training-only copy, never exported.
"""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports/protected-joint-full-CPU-v1'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU experiment must hide every GPU')
    if REPORTS.exists():raise ValueError('Preserve earlier protected experiment')
    import torch
    from slm_lab.corpus import file_hash
    from slm_lab.codenet_checkpoint_v1 import atomic_checkpoint,capture_rng
    from slm_lab.diverse_images import atomic_json
    from slm_lab.fast_joint_data_v1 import FastJointSources
    from slm_lab.fast_joint_stage_v1 import make_optimizer,parameter_groups
    from slm_lab.fast_joint_inference_v2 import fresh,export,load
    from slm_lab.fast_joint_checkpoint_v2 import bundle,restore
    from slm_lab.protected_joint_stage_v1 import backward_protected
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.manual_seed(957523);started=time.monotonic()
    original=json.loads((ROOT/'reports/fast-joint-v3/plan.json').read_text());plan=dict(original)
    plan.pop('owner');plan.pop('waiter');plan.pop('expires_at_unix')
    files=[Path(__file__),ROOT/'src/slm_lab/protected_joint_stage_v1.py',ROOT/'reports/protected-joint-stage-v1-tests.log',
           ROOT/'reports/fast-joint-v3/plan.json',ROOT/'reports/fast-joint-step1024-readback-v1/complete.json']
    plan['source_files_sha256']={**plan['source_files_sha256'],**{str(p):file_hash(p) for p in files}}
    plan['planned_steps']=2;plan['training_options']=json.loads(json.dumps(plan['training_options']))
    plan['training_options']['learning_rates']['language']=5e-6
    plan['training_options'].update(protected_objective='visual_language_gradient_isolation_and_full_context_own_reference_KL_v1',reference_KL_weight=.1)
    REPORTS.mkdir();model,_=fresh(plan,'cpu');reference,_=fresh(plan,'cpu');reference.eval().requires_grad_(False)
    optimizer=make_optimizer(model,plan['training_options']);data=FastJointSources(ROOT,model.config)
    program=min(range(len(data.lengths['program'])),key=lambda p:(data.lengths['program'][p],p));word=0
    plan.update(actual_short_source_controls=[['program',program],['words',word]],
        actual_training_only_reference_parameters=reference.counts()['total_unique'],full_size_model=True,production_updates=0,goal_achieved=False)
    atomic_json(REPORTS/'plan.json',plan);digest=file_hash(REPORTS/'plan.json');records=[]
    fixed_reference=tensor_digest(reference.named_parameters());groups=parameter_groups(model)
    selected={id(p) for rows in groups.values() for _,p in rows};fixed=tensor_digest((n,p) for n,p in model.named_parameters() if id(p) not in selected)
    def phase(name):atomic_json(REPORTS/'progress.json',dict(phase=name,seconds=time.monotonic()-started,production_updates=0,goal_achieved=False))
    def update(name,position):
        batch=data.batch(name,[position],'cpu');kind=data.kinds[name];model.train();model.network.gradient_checkpointing=True;optimizer.zero_grad(set_to_none=True)
        record=backward_protected(model,batch,kind=kind,global_targets=data.targets[name][position],
            global_context_targets=data.lengths[name][position]-1,global_words=int(kind=='word'),reference=reference)
        active=[p for rows in groups.values() for _,p in rows if p.grad is not None]
        if any(not bool(torch.isfinite(p.grad).all()) for p in active):raise ValueError('Finite selected gradients required')
        record.update(family=name,original_train_position=position,complete_context_tokens=data.lengths[name][position],
            response_EOS_targets=data.targets[name][position],actual_active_gradient_tensors=len(active),
            gradient_norm_before_clip=float(torch.nn.utils.clip_grad_norm_(active,1.,error_if_nonfinite=True)))
        optimizer.step();optimizer.zero_grad(set_to_none=True);return record
    phase('complete original public program; own-reference-anchored full language update')
    records.append(update('program',program));language_after_text=tensor_digest(groups['language'])
    state=bundle(model,optimizer,plan,digest,1,records,[capture_rng('cpu')],'protected_CPU')
    atomic_checkpoint(REPORTS/'resume-step1.pt',state,16*1024**3,26*1024**3);del state
    phase('complete original word crop; vision and word only; frozen decoder')
    second=update('words',word);expected=tensor_digest(model.named_parameters())
    def adam():return tensor_digest((str(i)+'.'+k,s[k]) for i,s in enumerate(optimizer.state.values()) for k in ('step','exp_avg','exp_avg_sq'))
    expected_adam=adam()
    if language_after_text!=tensor_digest(groups['language']):raise ValueError('Word learning or Adam decay changed any language weight')
    saved=torch.load(REPORTS/'resume-step1.pt',map_location='cpu',weights_only=True,mmap=True)
    completed,records=restore(model,optimizer,saved,plan,digest,'cpu',0,1,'protected_CPU')
    phase('exact next protected word update/Adam/CPU RNG recovery')
    repeated=update('words',word)
    if repeated!=second or expected!=tensor_digest(model.named_parameters()) or expected_adam!=adam():raise ValueError('Actual protected next update and every model/Adam state must recover exactly')
    records.append(repeated)
    if fixed_reference!=tensor_digest(reference.named_parameters()) or fixed!=tensor_digest((n,p) for n,p in model.named_parameters() if id(p) not in selected):
        raise ValueError('Every reference or inactive policy value must stay exact')
    atomic_checkpoint(REPORTS/'model.pt',export(model,plan,digest,'trained',2),5*1024**3,26*1024**3)
    from shutil import copyfile
    copyfile(ROOT/'checkpoints/extended-curriculum-v1/joint/tokenizer.json',REPORTS/'tokenizer.json')
    ids=torch.tensor([[1,261,267,71,2]]);pixels=torch.zeros(1,3,448,448);word_pixels=data.words.batch([word],'cpu')['word_pixels']
    def outputs(m):
        m.eval()
        with torch.no_grad():return dict(text=m(ids)['logits'].clone(),image=m(ids,pixel_values=pixels)['logits'].clone(),word=m.word_logits(word_pixels).clone())
    expected_outputs=outputs(model);phase('trained native text image word cold reload without frozen reference')
    cold,_=load(REPORTS/'model.pt',REPORTS/'plan.json','trained',expected_checkpoint_sha256=file_hash(REPORTS/'model.pt'),expected_plan_sha256=digest,device='cpu')
    actual=outputs(cold)
    if any(not torch.equal(expected_outputs[k],actual[k]) for k in actual):raise ValueError('Actual trained cold outputs must match exactly')
    atomic_json(REPORTS/'complete.json',dict(observed_at=datetime.now(timezone.utc).isoformat(),actual_model_parameters=model.counts()['total_unique'],
        actual_training_only_frozen_reference_parameters=reference.counts()['total_unique'],actual_disposable_logical_updates=2,physical_optimizer_calls=3,
        actual_records=records,actual_Adam_states=len(optimizer.state),
        language_bytes_exact_during_visual_update=True,reference_and_all_unused_heads_exact=True,
        actual_next_model_Adam_CPU_RNG_and_metrics_recovery_exact=True,actual_trained_text_image_word_cold_reload_exact=True,
        full_profile_four_GPU_capacity_and_held_quality_not_established=True,
        source_plan_sha256=digest,seconds=time.monotonic()-started,production_updates=0,goal_achieved=False))
    phase('complete short-source mechanics; no production or quality claim')


if __name__=='__main__':main()
