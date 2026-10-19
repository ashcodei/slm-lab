"""All actual bounded-PPO draws, source targets, actor/critic states and controls."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import torch

from .codenet_stage_v1 import EpochOrder
from .code_residual_checkpoint_v1 import _validate_adam,_validate_rng
from .corpus import file_hash
from .cuda_rng_readback_v1 import review_cuda_rng_layout
from .dense_ppo_control_v1 import FORMAT,validate_controlled
from .dense_ppo_cached_v2 import SAMPLER
from .dense_ppo_v1 import validate_rollouts,prompt_source
from .dense_ppo_inference_v1 import options,sources,FORMAT as EXPORT_FORMAT,INITIAL_FORMAT
from .diverse_images import atomic_json
from .whole_human_sft_evidence_v1 import tensor_digest


def budget(data,setting):
    draws=setting['steps']*setting['prompts_per_update']
    if not 1<=draws<=len(data):raise ValueError('Declared bounded unrepeated original train-source prefix required')
    order=EpochOrder(len(data),setting['seed']);selected=[order.at(i) for i in range(draws)];ledger=hashlib.sha256();tokens=0
    for p in selected:
        row=prompt_source(data,p);tokens+=len(row['prompt_ids'])
        ledger.update(json.dumps(row,sort_keys=True,separators=(',',':')).encode()+b'\n')
    if len(set(selected))!=draws:raise ValueError('No original training source may repeat within the bounded prefix')
    return dict(original_train_population=len(data),original_source_rows_drawn=draws,unique_original_source_positions=len(set(selected)),
        whole_original_prompt_tokens=tokens,ordered_full_prompt_ledger_sha256=ledger.hexdigest(),
        max_action_tokens=draws*setting['max_new_tokens'],response_budget=setting['max_new_tokens'],
        source_rounding_repeats=0,complete_epoch=draws==len(data))


def training_readback(plan,plan_sha,arm,initial,exported,state,receipt,journal,data):
    setting=options(plan,arm);steps=setting['steps'];scale=plan['arms'][arm]['human_reward_scale'];core=state['core'];payload=exported['policy']
    names=list(initial['policy']['parts']);baseline=initial['policy']['parts'];saved=payload['parts'];expected=budget(data,setting)
    if (initial.get('format')!=INITIAL_FORMAT or initial.get('arm')!='initial' or initial.get('run_plan_sha256')!=plan_sha
            or initial.get('actual_own_response_PPO_updates')!=0 or initial['policy']['sources']!=sources(plan,plan_sha)
            or initial['policy']['actual_counts']!=plan['actual_counts'] or tensor_digest(baseline.items())!=plan['initial_parts_sha256']
            or state.get('format')!=FORMAT or state.get('sampler')!=SAMPLER or state['options']!=setting or state['source_population']!=len(data)
            or state['terminal_human_reward_scale']!=scale or state['reward_provenance']!=plan['reward_provenance']
            or state['reward_counts']!=plan['reward_counts'] or core['completed']!=steps or core['planned']!=steps
            or core['groups']!=['actor','critic'] or core['optimized_names']!=names or core['world_size']!=1
            or len(core['metrics'])!=steps or core['metrics']!=journal
            or exported['format']!=EXPORT_FORMAT or exported['arm']!=arm or exported['run_plan_sha256']!=plan_sha
            or exported['actual_own_response_PPO_updates']!=steps or exported['terminal_human_reward_scale']!=scale
            or exported['training_options']!=setting or exported['source_exposures']!=expected or plan['source_exposures'][arm]!=expected
            or payload['sources']!=sources(plan,plan_sha) or core['model']['sources']!=sources(plan,plan_sha)
            or payload['frozen_dense_sha256']!=plan['frozen_dense_sha256'] or core['model']['frozen_dense_sha256']!=plan['frozen_dense_sha256']
            or payload['actual_counts']!=plan['actual_counts'] or saved.keys()!=baseline.keys()
            or any(not torch.equal(t,core['model']['parts'][n]) for n,t in saved.items())
            or any(t.shape!=baseline[n].shape or t.dtype!=baseline[n].dtype or not bool(torch.isfinite(t).all()) for n,t in saved.items())
            or exported.get('separate_training_reward_not_in_policy') is not True
            or exported.get('candidate_promoted') is not False or exported.get('goal_achieved') is not False):
        raise ValueError('Every complete bounded PPO/critic state, fixed reward/source profile and exported tensor must match')
    _validate_adam(core['optimizer'],[saved[n] for n in names],steps);_validate_rng({**core['rng'],'cuda':None},'cpu')
    if plan['training_device']=='cuda':cuda_layout=review_cuda_rng_layout(core['rng']['cuda'])
    elif plan['training_device']=='cpu_fixture' and core['rng']['cuda'] is None:cuda_layout=None
    else:raise ValueError('Exact declared actual CUDA training or explicit small CPU fixture RNG required')
    order=EpochOrder(len(data),setting['seed']);seen=Counter();prompt_tokens=0;action_tokens=0;raw_scores=[]
    for step,record in enumerate(journal):
        width=setting['prompts_per_update'];selected=[order.at(step*width+i) for i in range(width)];rows=record['rollouts']
        validate_rollouts(data,selected,setting,rows,plan['reward_provenance'],plan['vocab_size'],plan['max_text_length'])
        validate_controlled(rows,scale)
        actual=sum(len(r['response_ids']) for r in rows);context=sum(len(r['prompt_ids']) for r in rows)
        if (record['step']!=step+1 or record['source_positions']!=selected or record['actual_action_tokens']!=actual
                or record['actual_complete_prompt_tokens']!=context or record['learning_rate']!=setting['learning_rate']
                or not math.isfinite(record['token_weighted_PPO_loss']) or not math.isfinite(record['gradient_norm_before_clip'])
                or record['gradient_norm_before_clip']<0 or record['policy_unique_parameters']!=plan['actual_counts']['actual_unique']
                or record['separate_training_only_reward']!=plan['reward_counts'] or record['reward_model_not_in_policy_export'] is not True
                or record['own_sampled_responses_not_new_human_judgments'] is not True):
            raise ValueError('Every actual whole-source rollout and finite actor/critic objective metric required')
        seen.update(selected);prompt_tokens+=context;action_tokens+=actual;raw_scores.extend(r['raw_human_reward_score'] for r in rows)
    draws=steps*setting['prompts_per_update'];cursor=dict(completed_updates=steps,original_prompt_population=len(data),actual_prompt_draws=draws,
        complete_prompt_passes=draws//len(data),next_pass_prompts=draws%len(data),seed=setting['seed'],world_size=1)
    if (core['cursor']!=cursor or len(seen)!=draws or set(seen.values())!={1} or prompt_tokens!=expected['whole_original_prompt_tokens']
            or not draws<=action_tokens<=expected['max_action_tokens'] or not receipt['actual_trained_policy_text_image_critic_cold_reload_exact']):
        raise ValueError('Exact complete bounded source cursor, token exposures and actual trained cold reload required')
    return dict(actual_updates=steps,actor_critic_Adam_states=len(core['optimizer']['state']),declared_source_exposures=expected,
        actual_own_action_tokens=action_tokens,every_original_source_draw_and_saved_actor_critic_tensor_checked=True,
        opaque_CUDA_RNG_layout=cuda_layout,terminal_human_reward_scale=scale,raw_human_reward_score_mean=math.fsum(raw_scores)/len(raw_scores),
        raw_reward_mean_is_not_human_judgment_or_policy_quality=True)


def reconcile(root,reports,output,data,held,words,tokenizer):
    root,reports,output=map(Path,(root,reports,output));plan_path=reports/'plan.json';plan=json.loads(plan_path.read_text());sha=file_hash(plan_path)
    initial=torch.load(output/'initial/model.pt',map_location='cpu',weights_only=True,mmap=True);training={};first=[]
    for arm in plan['arms']:
        folder=output/arm;receipt=json.loads((reports/(arm+'-training-complete.json')).read_text())
        if file_hash(folder/'model.pt')!=receipt['checkpoint_sha256'] or file_hash(folder/'resume.pt')!=receipt['resume_sha256']:
            raise ValueError('Exact complete policy and optimizer bytes required')
        exported=torch.load(folder/'model.pt',map_location='cpu',weights_only=True,mmap=True)
        state=torch.load(folder/'resume.pt',map_location='cpu',weights_only=True,mmap=True)
        with (reports/(arm+'-metrics.jsonl')).open() as f:journal=[json.loads(line) for line in f]
        training[arm]=training_readback(plan,sha,arm,initial,exported,state,receipt,journal,data)
        first.append([{k:r[k] for k in ('original_train_position','prompt_ids','response_ids','old_logps','reference_logps','raw_human_reward_score')}
            for r in journal[0]['rollouts']])
    if any(r!=first[0] for r in first[1:]):raise ValueError('All matched initial models/seeds must sample the same first responses and raw rewards')
    from .dense_policy_native_readback_v1 import readback
    controls=readback(root,reports,plan['arms'],held,words,tokenizer)
    result=dict(complete_bounded_PPO_training=training,matched_initial_first_own_samples_exact=True,**controls,
        same_counted_frozen_base_reference=True,separate_training_only_reward_encoder_explicitly_counted=True,
        human_rankings_on_other_model_responses_not_fresh_own_output_judgments=True,
        reused_development_source_qualifications_retained=True,candidate_promoted=False,goal_achieved=False)
    atomic_json(reports/'independent-comparison.json',result);return result
