"""Actual full-size own-response PPO/critic/reward-basis CPU mechanics.

Both dense/word and DPO starting artifacts are explicitly disposable. Each
logical update samples two whole original train prompts with a128-action
finite horizon. This is not a full PPO epoch or a useful alignment claim.
"""
import copy
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import random
import time

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports/dense-ppo-cached-full-CPU-v2'
RESERVE=26*1024**3


def equal(a,b):
    import torch
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal(v,b[k]) for k,v in a.items())
    if isinstance(a,(list,tuple)):return type(a) is type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU PPO proof must explicitly hide every GPU')
    if REPORTS.exists():raise ValueError('Preserve prior complete or partial PPO attempts; no automatic retry')
    import numpy as np
    import torch
    from slm_lab.codenet_checkpoint_v1 import FileBindings,atomic_checkpoint,capture_rng
    from slm_lab.corpus import file_hash
    from slm_lab.dense_foundation_inference_v1 import fresh
    from slm_lab.dense_ppo_reward_v1 import load as load_reward,signature
    from slm_lab.dense_ppo_v1 import prompt_source
    from slm_lab.dense_ppo_cached_v2 import DenseCachedPPOStage,SAMPLER
    from slm_lab.diverse_images import atomic_json
    from slm_lab.own_dense_actor_v1 import OwnDenseActor
    from slm_lab.point_grounding_data_v1 import PointerData
    from slm_lab.shared_word_data_v1 import SharedWordData
    from slm_lab.whole_human_data_v2 import WholeHumanData
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    log=ROOT/'reports/dense-ppo-cached-v2-tests.log'
    if '4 passed' not in log.read_text():raise ValueError('Actual independent PPO/GAE gradient, recovery, EOS and feature-basis tests required')
    prior=ROOT/'reports/dense-human-preference-stage-full-CPU-v1';old=json.loads((prior/'plan.json').read_text())
    stage_complete=json.loads((prior/'complete.json').read_text());base_path=Path(old['combined_base_checkpoint'])
    actor_path=prior/'trained-disposable-policy.pt'
    if file_hash(actor_path)!=stage_complete['trained_policy_sha256']:raise ValueError('Exact disposable own trained actor required')
    files=dict(old['source_files_sha256'])
    for name in ('scripts/probe_dense_ppo_cached_full_CPU_v2.py','src/slm_lab/dense_ppo_reward_v1.py','src/slm_lab/dense_ppo_v1.py',
                 'src/slm_lab/dense_ppo_cached_v2.py','tests/test_dense_ppo_cached_v2.py','reports/dense-ppo-cached-v2-tests.log',
                 'tests/test_dense_ppo_v1.py','reports/dense-ppo-v1-tests.log'):
        files[str(ROOT/name)]=file_hash(ROOT/name)
    for name in ('plan.json','complete.json','independent-complete-readback.json','trained-disposable-policy.pt'):
        p=prior/name;files[str(p)]=file_hash(p)
    reward_plan=ROOT/'reports/whole-human-reward-v1/plan.json';completed=ROOT/'reports/whole-human-reward-v1/complete.json'
    reward_summary=json.loads(completed.read_text());arm=reward_summary['selected_arm']
    reward_inputs=dict(checkpoint=str(ROOT/'checkpoints/whole-human-reward-v1'/arm/'model.pt'),plan=str(reward_plan),
        feature_admission=str(ROOT/'data/whole-human-reward-features-v1/admission.json'),arm=arm)
    for k in ('checkpoint','plan','feature_admission'):
        reward_inputs[k+'_sha256']=file_hash(reward_inputs[k]);files[reward_inputs[k]]=reward_inputs[k+'_sha256']
    if reward_inputs['checkpoint_sha256']!=reward_summary['trained_exports'][arm]['checkpoint_sha256']:
        raise ValueError('Exact completed source-bound full-human reward head required')
    for name in ('complete.json','independent-complete-reward-readback-v1.json'):
        p=reward_plan.parent/name;files[str(p)]=file_hash(p)
    guard=FileBindings(files);guard.check(full=True);started=time.monotonic()
    base,_=fresh(old['initial_dense_settings'],'cpu')
    base.import_export(torch.load(base_path,map_location='cpu',weights_only=True,mmap=True)['model'],expected_sources=old['combined_base_sources'])
    model=OwnDenseActor(base,**old['actor_settings']).eval()
    actor_sources=dict(full_stage_CPU_plan_sha256=file_hash(prior/'plan.json'),combined_base_sha256=file_hash(base_path),
        human_admission_sha256=file_hash(ROOT/'data/whole-human-alignment-v2/admission.json'))
    model.import_export(torch.load(actor_path,map_location='cpu',weights_only=True,mmap=True),expected_sources=actor_sources)
    scorer=load_reward(reward_inputs,'cpu');data=WholeHumanData(ROOT/'data/whole-human-alignment-v2','preferences','train','optimization')
    options=dict(seed=951031,steps=3,prompts_per_update=2,max_new_tokens=128,learning_rate=3e-6,betas=[.9,.95],
        weight_decay=.01,gradient_clip=1.,KL_coefficient=.05,gamma=1.,GAE_lambda=.95,clip=.2,value_coefficient=.5)
    REPORTS.mkdir()
    atomic_json(REPORTS/'plan.json',dict(created_at=datetime.now(timezone.utc).isoformat(),source_files_sha256=files,
        combined_base_checkpoint=str(base_path),combined_base_sha256=file_hash(base_path),combined_base_sources=old['combined_base_sources'],
        initial_dense_settings=old['initial_dense_settings'],actor_settings=model.settings,initial_disposable_actor=str(actor_path),
        initial_disposable_actor_sources=actor_sources,reward_inputs=reward_inputs,reward_provenance=scorer.provenance,
        actual_policy_counts=model.counts(),separate_training_only_reward_counts=scorer.counts(),training_options=options,
        CPU_FP32_threads=2,logical_disposable_updates=2,own_generated_response_horizon=128,sampler=SAMPLER,
        no_short_prompt_selection=True,production_GPU_capacity_and_useful_alignment_not_established=True,goal_achieved=False))
    sources=dict(full_PPO_CPU_plan_sha256=file_hash(REPORTS/'plan.json'),initial_actor_sha256=file_hash(actor_path),
        reward_checkpoint_sha256=reward_inputs['checkpoint_sha256'])
    optimizer=torch.optim.AdamW([p for _,p in model.named_selected(('actor','critic'))],lr=options['learning_rate'],
        betas=tuple(options['betas']),weight_decay=options['weight_decay'],foreach=False)
    stage=DenseCachedPPOStage(model,optimizer,data,scorer,options,sources)
    positions=[stage.positions(i) for i in range(2)]
    atomic_json(REPORTS/'whole-original-prompt-controls.json',dict(source_positions=positions,
        whole_original_prompts=[prompt_source(data,p) for ps in positions for p in ps]))
    pointer=PointerData(ROOT/'data/localized-pointer-candidate-v1','train','audit')
    p=json.loads((ROOT/'reports/frozen-base-pointer-full-CPU-v1/complete.json').read_text())['original_source_position'];photo=pointer.batch([p],'cpu')
    words=SharedWordData(ROOT/'data/shared-byte-words-v1','train','optimization');word=words.batch([50000],'cpu')
    ids,pixels=photo['input_ids'],photo['pixel_values'];frozen=tensor_digest(base.named_parameters());fixed=signature(stage.fixed)
    reward_digest=tensor_digest(scorer.named_parameters());reward_signature=signature(scorer.parameters())
    critic_before=model.parts['critic'].weight.detach().clone();actor_before=tensor_digest(model.named_selected(('actor',)))
    with torch.no_grad():
        base_image=base(ids,pixels)['logits'].clone();base_word=base.word_logits(word['word_pixels']).clone()
        cache=base.network.prefill(ids,pixels)['cache'];cached=base.network.decode(torch.tensor([[4]]),cache)['logits'].clone()
    random.seed(951431);np.random.seed(951431);torch.manual_seed(951431)
    first=stage.update();print('actual_own_response_PPO_update',json.dumps(first),flush=True)
    atomic_checkpoint(REPORTS/'resume-step1.pt',stage.state(),256*1024**2,RESERVE)
    def draws():return [random.random(),float(np.random.random()),float(torch.rand(()))]
    expected_draws=draws();expected=stage.update();print('actual_own_response_PPO_update',json.dumps(expected),flush=True)
    learned=tensor_digest(model.parts.named_parameters());adam=copy.deepcopy(optimizer.state_dict());rng=capture_rng('cpu')
    saved=torch.load(REPORTS/'resume-step1.pt',map_location='cpu',weights_only=True,mmap=True)
    pristine=tensor_digest((str(i)+'.'+k,v[k]) for i,v in saved['core']['optimizer']['state'].items() for k in ('step','exp_avg','exp_avg_sq'))
    stage.resume(saved)
    if (draws()!=expected_draws or stage.update()!=expected or learned!=tensor_digest(model.parts.named_parameters())
            or not equal(adam,optimizer.state_dict()) or not equal(rng,capture_rng('cpu'))
            or pristine!=tensor_digest((str(i)+'.'+k,v[k]) for i,v in saved['core']['optimizer']['state'].items() for k in ('step','exp_avg','exp_avg_sq'))):
        raise ValueError('Actual own sampled actions/scores, next actor/critic weights, Adam, all CPU RNG and pristine input moments must recover exactly')
    atomic_checkpoint(REPORTS/'resume-step2.pt',stage.state(),256*1024**2,RESERVE)
    atomic_checkpoint(REPORTS/'trained-disposable-policy.pt',model.export(sources),96*1024**2,RESERVE)
    if (frozen!=tensor_digest(base.named_parameters()) or fixed!=signature(stage.fixed)
            or reward_signature!=signature(scorer.parameters()) or reward_digest!=tensor_digest(scorer.named_parameters())
            or torch.equal(critic_before,model.parts['critic'].weight) or actor_before==tensor_digest(model.named_selected(('actor',)))):
        raise ValueError('Actor and critic must actually change while every frozen base/vision/word/reward value and metadata remains exact')
    model.eval();text_ids=torch.tensor([expected['rollouts'][0]['prompt_ids']+expected['rollouts'][0]['response_ids']])
    with torch.no_grad():
        if (not torch.equal(base_image,base(ids,pixels)['logits']) or not torch.equal(base_word,base.word_logits(word['word_pixels']))
                or not torch.equal(cached,base.network.decode(torch.tensor([[4]]),cache)['logits'])):
            raise ValueError('Actual original fused base image/word and existing cache must remain exact')
        text=model(text_ids);image=model(ids,pixels)
    cold_base,_=fresh(old['initial_dense_settings'],'cpu')
    cold_base.import_export(torch.load(base_path,map_location='cpu',weights_only=True,mmap=True)['model'],expected_sources=old['combined_base_sources'])
    cold=OwnDenseActor(cold_base,**model.settings).eval()
    cold.import_export(torch.load(REPORTS/'trained-disposable-policy.pt',map_location='cpu',weights_only=True,mmap=True),expected_sources=sources)
    with torch.no_grad():
        cold_text=cold(text_ids);cold_image=cold(ids,pixels)
        if (any(not torch.equal(text[k],cold_text[k]) or not torch.equal(image[k],cold_image[k]) for k in ('logits','values'))
                or not torch.equal(base_word,cold_base.word_logits(word['word_pixels']))):
            raise ValueError('Actual learned full-size policy text/image/critic and frozen word cold reload must be exact without loading the training-only reward')
    guard.check(full=True)
    rows=[r for record in stage.metrics for r in record['rollouts']]
    atomic_json(REPORTS/'complete.json',dict(actual_policy_counts=model.counts(),separate_training_only_reward=scorer.counts(),
        actual_actor_and_critic_Adam_states=len(optimizer.state),actual_actor_and_critic_values=sum(p.numel() for p in stage.parameters),
        actual_own_response_updates=stage.metrics,whole_original_prompt_tokens=sum(len(r['prompt_ids']) for r in rows),
        actual_own_sampled_action_tokens=sum(len(r['response_ids']) for r in rows),response_budget=options['max_new_tokens'],sampler=SAMPLER,
        actual_prefill_calls=sum(r['actual_prefill_calls'] for r in rows),actual_decode_calls=sum(r['actual_decode_calls'] for r in rows),
        max_absolute_cache_full_logp_difference=max(r['max_absolute_cache_full_logp_difference'] for r in rows),
        exact_next_own_actions_rewards_actor_critic_Adam_all_CPU_RNG_and_metrics=True,pristine_saved_Adam_payload_unchanged=True,
        actor_and_separate_critic_both_actually_trained=True,all_frozen_dense_word_auxiliary_and_training_reward_values_metadata_exact=True,
        actual_trained_text_image_values_word_cold_reload_exact_without_reward_model=True,
        retained_step1_sha256=file_hash(REPORTS/'resume-step1.pt'),retained_step2_sha256=file_hash(REPORTS/'resume-step2.pt'),
        trained_policy_sha256=file_hash(REPORTS/'trained-disposable-policy.pt'),frozen_dense_sha256=frozen,
        separate_training_reward_sha256=reward_digest,seconds=time.monotonic()-started,logical_disposable_updates=2,physical_optimizer_calls=3,
        source_order_without_short_prompt_selection=True,completed_human_rankings_are_other_model_responses=True,
        own_samples_have_not_been_judged_by_humans=True,full_epoch_GPU_capacity_held_policy_gain_and_high_end_quality_not_established=True,
        all_probe_updates_discarded=True,candidate_promoted=False,goal_achieved=False))
    print(REPORTS/'complete.json',flush=True)


if __name__=='__main__':main()
