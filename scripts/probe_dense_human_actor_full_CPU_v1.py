"""Actual under1B policy, complete human-ranking DPO and exact CPU recovery.

Every optimizer update in this probe is disposable. This is not a full human
preference epoch, held alignment gain, PPO training or frontier capability.
"""
import copy
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import random
import time

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports/dense-human-actor-full-CPU-v1'
MAX_STATE=256*1024**2
RESERVE=26*1024**3


def equal(a,b):
    import torch
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal(v,b[k]) for k,v in a.items())
    if isinstance(a,(list,tuple)):return type(a) is type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def signature(parameters):return [(id(p),p._version,p.data_ptr(),p.requires_grad) for p in parameters]


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Actual full policy CPU proof must hide all GPUs')
    if REPORTS.exists():raise ValueError('Preserve each full policy attempt; no automatic restart')
    import numpy as np
    import torch
    from slm_lab.codenet_checkpoint_v1 import FileBindings,atomic_checkpoint,capture_rng
    from slm_lab.corpus import file_hash
    from slm_lab.dense_foundation_inference_v1 import fresh
    from slm_lab.dense_human_preference_v1 import preference_gradients,pair_information
    from slm_lab.diverse_images import atomic_json
    from slm_lab.own_dense_actor_v1 import OwnDenseActor
    from slm_lab.own_dense_actor_checkpoint_v1 import make_state,restore_state
    from slm_lab.point_grounding_data_v1 import PointerData
    from slm_lab.whole_human_data_v2 import WholeHumanData
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    tests=ROOT/'reports/dense-actor-human-preference-v1-tests.log'
    if '5 passed' not in tests.read_text():raise ValueError('Actual parity/gradient/cache/recovery/human-source checks required')
    prior=ROOT/'reports/dense-foundation-full-CPU-v1'
    prior_plan=json.loads((prior/'plan.json').read_text());settings=prior_plan['initial_dense_settings']
    files=dict(prior_plan['source_files_sha256'])
    names=['scripts/probe_dense_human_actor_full_CPU_v1.py','src/slm_lab/own_dense_actor_v1.py',
        'src/slm_lab/own_dense_actor_checkpoint_v1.py','src/slm_lab/dense_human_preference_v1.py',
        'tests/test_own_dense_actor_v1.py','tests/test_dense_human_preference_v1.py',
        'reports/dense-actor-human-preference-v1-tests.log',
        'reports/dense-foundation-full-CPU-v1/plan.json','reports/dense-foundation-full-CPU-v1/complete.json',
        'reports/dense-foundation-full-CPU-v1/independent-complete-readback.json']
    for name in names:files[str(ROOT/name)]=file_hash(ROOT/name)
    guard=FileBindings(files);guard.check(full=True);started=time.monotonic()
    base,metadata=fresh(settings,'cpu');base.eval();original_signature=signature(base.parameters())
    original_digest=tensor_digest(base.named_parameters());initial_rng=torch.get_rng_state().clone()
    model=OwnDenseActor(base,rank=16,seed=949331).eval()
    if (model.counts()['actual_unique']!=999706442 or model.counts()['actor']!=15305216
            or model.counts()['separate_policy_critic']!=1536 or original_signature!=signature(base.parameters())
            or not torch.equal(initial_rng,torch.get_rng_state())):
        raise ValueError('Actual full bounded policy/critic inventory and unchanged original metadata/RNG required')
    data=WholeHumanData(ROOT/'data/whole-human-alignment-v2','preferences','train','optimization')
    position=min(range(len(data)),key=lambda p:sum(data.rows[p]['native'][f]['length'] for f in ('chosen','rejected')))
    information=pair_information(data,position)
    pointer=PointerData(ROOT/'data/localized-pointer-candidate-v1','train','audit')
    pp=json.loads((ROOT/'reports/frozen-base-pointer-full-CPU-v1/complete.json').read_text())['original_source_position']
    photo=pointer.batch([pp],'cpu');ids,pixels=photo['input_ids'],photo['pixel_values']
    chosen=data.batch([position],'chosen','cpu');rejected=data.batch([position],'rejected','cpu')
    REPORTS.mkdir()
    atomic_json(REPORTS/'plan.json',dict(created_at=datetime.now(timezone.utc).isoformat(),source_files_sha256=files,
        initial_dense_settings=settings,actor_settings=model.settings,actual_counts=model.counts(),
        original_human_ranking=information,original_photo_train_position=pp,
        original_photo_id=pointer.rows[pp]['id'],CPU_FP32_threads=2,groups=['actor'],
        all_updates_disposable=True,planned_logical_updates=2,planned_physical_optimizer_calls=3,
        DPO_beta=.1,chosen_CE_weight=.1,learning_rate=1e-5,betas=[.9,.95],weight_decay=.01,
        human_ranked_other_model_responses_not_fresh_own_output_judgments=True,
        original_dense_source_digest=original_digest,critic_new_and_untrained=True,goal_achieved=False))
    with torch.no_grad():
        for batch in (chosen,rejected):
            if not torch.equal(base(**batch)['logits'],model(**batch)['logits']):
                raise ValueError('Both complete original chosen/rejected initial policy outputs must equal the one shared dense reference')
        image_before=base(ids,pixels)['logits'].clone()
        if not torch.equal(image_before,model(ids,pixels)['logits']):raise ValueError('Initial actual full native image logits must equal the dense source')
        base_cache=base.network.prefill(chosen['input_ids'])['cache']
        base_cached_before=base.network.decode(torch.tensor([[4]]),base_cache)['logits'].clone()
        actor_cache=model.policy.prefill(chosen['input_ids'])['cache']
    params=[p for _,p in model.named_selected(('actor',))]
    critic_signature=signature(model.parts['critic'].parameters());critic_weights=model.parts['critic'].weight.clone()
    optimizer=torch.optim.AdamW(params,lr=1e-5,betas=(.9,.95),weight_decay=.01,foreach=False)
    random.seed(949731);np.random.seed(949731);torch.manual_seed(949731)
    def update():
        optimizer.zero_grad(set_to_none=True)
        draws=[random.random(),float(np.random.random()),float(torch.rand(()))]
        record=preference_gradients(model,data,[position],beta=.1,chosen_CE_weight=.1)
        if (any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in params)
                or any(p.grad is not None for p in base.parameters()) or model.parts['critic'].weight.grad is not None):
            raise ValueError('Every policy adapter and no frozen dense/auxiliary/unused critic tensor must receive finite gradients')
        record['gradient_norm_before_clip']=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True))
        record['diagnostic_python_numpy_torch_CPU_draws']=draws
        optimizer.step();optimizer.zero_grad(set_to_none=True);return record
    sources=dict(full_CPU_plan_sha256=file_hash(REPORTS/'plan.json'),initial_dense_sha256=settings['initializer_sha256'],
        human_admission_sha256=file_hash(ROOT/'data/whole-human-alignment-v2/admission.json'))
    first=update();print('actual_full_human_policy_DPO_update',1,first,flush=True)
    cursor=dict(original_human_ranking_position=position,completed_updates=1)
    state=make_state(model,optimizer,groups=('actor',),sources=sources,completed=1,planned=2,cursor=cursor,metrics=[first],device='cpu')
    atomic_checkpoint(REPORTS/'resume-step1.pt',state,MAX_STATE,RESERVE);del state
    expected=update();print('actual_full_human_policy_DPO_update',2,expected,flush=True)
    expected_parts=tensor_digest(model.parts.named_parameters());expected_adam=copy.deepcopy(optimizer.state_dict());rng=capture_rng('cpu')
    saved=torch.load(REPORTS/'resume-step1.pt',map_location='cpu',weights_only=True,mmap=True)
    saved_moments=tensor_digest((str(i)+'.'+k,v[k]) for i,v in saved['optimizer']['state'].items() for k in ('step','exp_avg','exp_avg_sq'))
    restore_state(model,optimizer,saved,groups=('actor',),expected_sources=sources,planned=2,expected_cursor=cursor,device='cpu')
    actual=update()
    if (actual!=expected or expected_parts!=tensor_digest(model.parts.named_parameters()) or not equal(expected_adam,optimizer.state_dict())
            or not equal(rng,capture_rng('cpu')) or saved_moments!=tensor_digest((str(i)+'.'+k,v[k]) for i,v in saved['optimizer']['state'].items() for k in ('step','exp_avg','exp_avg_sq'))):
        raise ValueError('Actual full policy next update/Adam/RNG/source/metrics and pristine input payload must recover exactly')
    state=make_state(model,optimizer,groups=('actor',),sources=sources,completed=2,planned=2,
        cursor={**cursor,'completed_updates':2},metrics=[first,expected],device='cpu')
    atomic_checkpoint(REPORTS/'resume-step2.pt',state,MAX_STATE,RESERVE)
    atomic_checkpoint(REPORTS/'trained-disposable-policy.pt',model.export(sources),96*1024**2,RESERVE)
    del state,saved,expected_adam
    model.eval()
    if (original_signature!=signature(base.parameters()) or original_digest!=tensor_digest(base.named_parameters())
            or critic_signature!=signature(model.parts['critic'].parameters()) or not torch.equal(critic_weights,model.parts['critic'].weight)):
        raise ValueError('Every original dense/head and unused separate critic value/metadata must remain exact')
    with torch.no_grad():
        if (not torch.equal(image_before,base(ids,pixels)['logits'])
                or not torch.equal(base_cached_before,base.network.decode(torch.tensor([[4]]),base_cache)['logits'])):
            raise ValueError('Original native image output and existing original dense cache must stay exact')
        try:model.policy.decode(torch.tensor([[4]]),actor_cache)
        except ValueError:pass
        else:raise ValueError('An actor cache predating actual adapter updates must reject')
        learned=model(**chosen)['logits'].clone();learned_image=model(ids,pixels)['logits'].clone()
        prompt=data.sequence(position,'chosen')[0][:data.rows[position]['native']['chosen']['target_start']]
        rollout=model.generate(torch.tensor([prompt]),max_new_tokens=4,temperature=.7,eos_token_id=2,return_rollout=True)
        if (not bool(torch.isfinite(rollout['logps']).all()) or not bool(torch.isfinite(rollout['values']).all())
                or rollout['values'].count_nonzero()!=0 or rollout['logps'].shape!=rollout['values'].shape):
            raise ValueError('Actual stochastic sampling probabilities and separate untrained critic values required')
    cold_base,_=fresh(settings,'cpu');cold=OwnDenseActor(cold_base,**model.settings).eval()
    cold.import_export(torch.load(REPORTS/'trained-disposable-policy.pt',map_location='cpu',weights_only=True,mmap=True),expected_sources=sources)
    with torch.no_grad():
        if (not torch.equal(learned,cold(**chosen)['logits']) or not torch.equal(learned_image,cold(ids,pixels)['logits'])
                or expected_parts!=tensor_digest(cold.parts.named_parameters())):
            raise ValueError('Actual full trained policy cold text/image reload and every new value must be exact')
    guard.check(full=True)
    atomic_json(REPORTS/'complete.json',dict(actual_counts=model.counts(),actual_actor_Adam_states=len(optimizer.state),
        actual_actor_values=sum(p.numel() for p in params),actual_separate_critic_values=1536,
        complete_original_human_train_ranking=information,complete_source_DPO_metrics=[first,expected],
        own_reference_is_same_frozen_counted_dense_backbone=True,external_teacher_or_extra_reference_backbone=False,
        actual_next_policy_weights_Adam_all_CPU_RNG_cursor_and_metrics_exact=True,
        input_recovery_payload_Adam_unchanged=True,all_original_dense_and_auxiliary_weights_metadata_native_image_and_cache_exact=True,
        unused_separate_critic_values_and_metadata_exact=True,critic_not_trained=True,
        original_initial_actual_text_and_image_logits_bitwise_equal=True,old_policy_cache_rejected_after_update=True,
        actual_stochastic_rollout_tokens=rollout['tokens'].tolist(),actual_stochastic_rollout_logps=rollout['logps'].tolist(),
        actual_separate_critic_values_in_rollout=rollout['values'].tolist(),actual_rollout_response_mask=rollout['response_mask'].tolist(),
        actual_trained_full_size_text_image_cold_reload_exact=True,
        original_dense_sha256=original_digest,retained_step1_sha256=file_hash(REPORTS/'resume-step1.pt'),
        retained_step2_sha256=file_hash(REPORTS/'resume-step2.pt'),trained_policy_sha256=file_hash(REPORTS/'trained-disposable-policy.pt'),
        seconds=time.monotonic()-started,logical_disposable_updates=2,physical_optimizer_calls=3,
        all_probe_updates_discarded_before_production=True,human_ranked_other_model_responses_not_fresh_own_output_judgments=True,
        full_human_epoch_GPU_capacity_held_preference_gain_PPO_and_final_quality_not_claimed=True,candidate_promoted=False,goal_achieved=False))
    print(REPORTS/'complete.json',flush=True)


if __name__=='__main__':main()
