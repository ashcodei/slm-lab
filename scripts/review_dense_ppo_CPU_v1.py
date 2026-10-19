"""Replay every small-probe own action and reward; audit both saved PPO states.

This reads all four original prompts and every own response token, reexecutes
the neural sampling distributions and original-encoder reward, and checks all
actor/critic Adam states. It does not replay optimizer gradients or evaluate
held capability. All probe updates remain disposable.
"""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports/dense-ppo-full-CPU-v1'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('The independent CPU reader must hide every GPU')
    destination=REPORTS/'independent-complete-neural-readback.json'
    if destination.exists():raise ValueError('Preserve prior readbacks')
    import torch
    from torch.nn import functional as F
    from slm_lab.codenet_checkpoint_v1 import FileBindings
    from slm_lab.code_residual_checkpoint_v1 import _validate_adam,_validate_rng
    from slm_lab.codenet_stage_v1 import EpochOrder
    from slm_lab.corpus import file_hash
    from slm_lab.dense_ppo_reward_v1 import load as load_reward
    from slm_lab.dense_ppo_v1 import validate_rollouts
    from slm_lab.diverse_images import atomic_json
    from slm_lab.model import ModelConfig
    from slm_lab.own_dense_actor_v1 import OwnDenseActor
    from slm_lab.own_dense_growth_v1 import OwnDenseAssistant,new_backbone
    from slm_lab.whole_human_data_v2 import WholeHumanData
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);started=time.monotonic()
    plan_path=REPORTS/'plan.json';plan=json.loads(plan_path.read_text());complete=json.loads((REPORTS/'complete.json').read_text())
    guard=FileBindings(plan['source_files_sha256']);guard.check(full=True)
    ppo_sources=dict(full_PPO_CPU_plan_sha256=file_hash(plan_path),initial_actor_sha256=file_hash(plan['initial_disposable_actor']),
        reward_checkpoint_sha256=plan['reward_inputs']['checkpoint_sha256'])
    if (complete['logical_disposable_updates']!=2 or complete['physical_optimizer_calls']!=3
            or complete['actual_policy_counts']!=plan['actual_policy_counts'] or complete['goal_achieved'] is not False):
        raise ValueError('Exact bounded original CPU PPO proof and actual model counts required')
    states=[]
    for step in (1,2):
        path=REPORTS/f'resume-step{step}.pt'
        if file_hash(path)!=complete[f'retained_step{step}_sha256']:raise ValueError('Every original saved optimizer byte required')
        state=torch.load(path,map_location='cpu',weights_only=True,mmap=True);core=state['core'];payload=core['model']
        names=[n for n in payload['parts']]
        if (state['options']!=plan['training_options'] or state['reward_provenance']!=plan['reward_provenance']
                or state['reward_counts']!=plan['separate_training_only_reward_counts']
                or core['completed']!=step or core['planned']!=3 or core['groups']!=['actor','critic']
                or core['optimized_names']!=names or core['world_size']!=1
                or core['metrics']!=complete['actual_own_response_updates'][:step]
                or payload['actual_counts']!=plan['actual_policy_counts'] or payload['sources']!=ppo_sources
                or payload['frozen_dense_sha256']!=complete['frozen_dense_sha256']
                or set(payload['parts'])-set(names)):
            raise ValueError('Both complete actual policy/critic states, sources, fixed reward and histories required')
        _validate_adam(core['optimizer'],[payload['parts'][n] for n in names],step);_validate_rng(core['rng'],'cpu')
        if not payload['parts']['critic.weight'].count_nonzero():raise ValueError('The separate critic must really learn')
        actor_signal=any(bool(core['optimizer']['state'][i]['exp_avg'].count_nonzero()) for i,n in enumerate(names) if not n.startswith('critic.'))
        critic_signal=any(bool(core['optimizer']['state'][i]['exp_avg'].count_nonzero()) for i,n in enumerate(names) if n.startswith('critic.'))
        if not actor_signal or not critic_signal:raise ValueError('Actor and critic need actual optimizer gradient signals beyond weight decay')
        states.append(state)
    export_path=REPORTS/'trained-disposable-policy.pt'
    if file_hash(export_path)!=complete['trained_policy_sha256']:raise ValueError('Exact final exported policy bytes required')
    exported=torch.load(export_path,map_location='cpu',weights_only=True,mmap=True)
    if any(not torch.equal(t,states[1]['core']['model']['parts'][n]) for n,t in exported['parts'].items()):
        raise ValueError('Every final actor/critic value must equal the complete second state')
    # Build from the standalone fused checkpoint, not the probe factory.
    raw=torch.load(plan['combined_base_checkpoint'],map_location='cpu',weights_only=True,mmap=True)['model']
    base=OwnDenseAssistant(new_backbone(ModelConfig(**raw['config']),948531),**raw['settings']).eval()
    base.import_export(raw,expected_sources=plan['combined_base_sources']);del raw
    if tensor_digest(base.named_parameters())!=complete['frozen_dense_sha256']:raise ValueError('Every original dense/vision/word/auxiliary value required')
    model=OwnDenseActor(base,**plan['actor_settings']).eval();scorer=load_reward(plan['reward_inputs'],'cpu')
    if model.counts()!=plan['actual_policy_counts'] or scorer.counts()!=plan['separate_training_only_reward_counts']:
        raise ValueError('Actual separate training-only model and deployed model inventories required')
    data=WholeHumanData(ROOT/'data/whole-human-alignment-v2','preferences','train','optimization');setting=plan['training_options']
    for step,state in enumerate(states,1):
        model.validate_export(state['core']['model'],expected_sources=ppo_sources)
        draws=step*setting['prompts_per_update']
        cursor=dict(completed_updates=step,original_prompt_population=len(data),actual_prompt_draws=draws,
            complete_prompt_passes=draws//len(data),next_pass_prompts=draws%len(data),seed=setting['seed'],world_size=1)
        if state['source_population']!=len(data) or state['core']['cursor']!=cursor:
            raise ValueError('Every actual own-response source draw and original source cursor must reconstruct')
    model.validate_export(exported,expected_sources=ppo_sources)
    order=EpochOrder(len(data),setting['seed']);replayed=[]
    for step,record in enumerate(complete['actual_own_response_updates']):
        if step==0:
            payload=torch.load(plan['initial_disposable_actor'],map_location='cpu',weights_only=True,mmap=True)
            model.import_export(payload,expected_sources=plan['initial_disposable_actor_sources'])
            # The bound probe seeds sampling here, after all model loading.
            torch.manual_seed(951431)
        else:
            payload=states[0]['core']['model'];model.import_export(payload,expected_sources=ppo_sources)
            torch.set_rng_state(states[0]['core']['rng']['torch']);torch.rand(()) # explicit intervening RNG recovery control
        positions=[order.at(step*setting['prompts_per_update']+i) for i in range(setting['prompts_per_update'])]
        validate_rollouts(data,positions,setting,record['rollouts'],scorer.provenance,model.config.vocab_size,
            min(model.config.max_text_length,scorer.encoder.config.max_text_length))
        for row in record['rollouts']:
            prompt,actions=row['prompt_ids'],row['response_ids'];sampled=[];actual_logps=[]
            with torch.no_grad():
                for i,action in enumerate(actions):
                    ids=torch.tensor([prompt+actions[:i]]);logits=model(ids)['logits'][0,-1].float()
                    token=int(torch.multinomial(logits.softmax(-1),1));sampled.append(token)
                    actual_logps.append(float(logits.log_softmax(-1)[action]))
                if sampled!=actions or actual_logps!=row['actual_sampling_logps']:
                    raise ValueError('Every original own sampled action and behavior log probability must replay exactly')
                ids=torch.tensor([prompt+actions]);out=model(ids);reference=base(ids)
                if out['values'][0,len(prompt)-1:-1].tolist()!=row['old_values']:
                    raise ValueError('Every actual causal pre-action old critic value must replay exactly')
                reference_logps=reference['logits'][0,len(prompt)-1:-1].float().log_softmax(-1).gather(1,torch.tensor(actions)[:,None]).squeeze(1)
                if reference_logps.tolist()!=row['reference_logps']:raise ValueError('Every original fixed-base reference log probability must replay exactly')
                scoring=torch.tensor([row['scoring_ids']]);hidden=scorer.encoder(scoring,return_hidden=True,return_logits=False)['hidden'][:,-1].float()
                weights=scorer.head.state_dict();norm=hidden*torch.rsqrt(hidden.square().mean(-1,keepdim=True)+scorer.head[0].eps)
                reward=F.linear(F.silu(F.linear(norm*weights['0.weight'],weights['1.weight'])),weights['3.weight']).reshape(())
                if float(reward)!=row['human_reward_score']:raise ValueError('Every original-encoder RMSNorm/MLP human reward must neurally replay exactly')
            replayed.append(dict(source_id=row['id'],original_train_position=row['original_train_position'],
                whole_prompt_tokens=len(prompt),actual_sampled_actions=actions,human_reward_score=float(reward),
                exact_behavior_actions_logps_critic_reference_and_manual_reward=True))
        print('actual_full_neural_sampling_reward_readback_update',step+1,flush=True)
    if sum(r['whole_prompt_tokens'] for r in replayed)!=complete['whole_original_prompt_tokens']:
        raise ValueError('Every complete original prompt must remain counted')
    if sum(len(r['actual_sampled_actions']) for r in replayed)!=complete['actual_own_sampled_action_tokens']:
        raise ValueError('Every actual own sampled token must remain counted')
    scorer.check();guard.check(full=True)
    atomic_json(destination,dict(observed_at=datetime.now(timezone.utc).isoformat(),reader_sha256=file_hash(Path(__file__)),
        complete_sha256=file_hash(REPORTS/'complete.json'),actual_policy_counts=model.counts(),training_only_reward_counts=scorer.counts(),
        actual_replayed_own_rollouts=replayed,all_562_actor_critic_Adam_states_both_steps_validated=len(states[0]['core']['optimizer']['state'])==562,
        actor_and_critic_have_nonzero_optimizer_gradient_signal_beyond_weight_decay=True,
        all_saved_histories_source_masks_reward_GAE_and_joint_normalization_reconstruct=True,
        every_actual_own_sampled_action_behavior_logp_old_critic_and_fixed_reference_neurally_replayed_exactly=True,
        every_human_reward_replayed_with_original_general_encoder_and_direct_RMSNorm_MLP=True,
        every_exported_actor_critic_value_equal_complete_second_snapshot=True,
        no_optimizer_gradient_replay_or_held_quality_evaluation=True,source_responses_ranked_by_humans_but_own_samples_not_human_judged=True,
        all_probe_updates_disposable=True,candidate_promoted=False,goal_achieved=False,seconds=time.monotonic()-started))
    print(destination,flush=True)


if __name__=='__main__':main()
