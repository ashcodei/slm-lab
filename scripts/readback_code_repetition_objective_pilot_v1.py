"""Reencode all original sources and reconcile both actual objective endpoints."""
from collections import Counter,defaultdict
from datetime import datetime,timezone
import gzip
import hashlib
import json
from pathlib import Path
import random

import numpy as np
from tokenizers import Tokenizer
import torch

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.model import ModelConfig,TinyMultimodalLM
from slm_lab.recurrent_ablation import state_digest

ROOT=Path(__file__).resolve().parents[1]


def read(p):return json.loads(p.read_text())


def main():
    torch.set_num_threads(1)
    folder=ROOT/'reports/code-repetition-objective-pilot-v1'
    output=folder/'independent-complete-readback.json'
    if output.exists():raise ValueError('Preserve any previous independent readback')
    receipt=read(ROOT/'reports/code-repetition-objective-pilot-v1-launch.json')
    proc=Path('/proc')/str(receipt['pid'])/'stat'
    if proc.exists():
        fields=proc.read_text().rsplit(')',1)[1].split()
        assert fields[19]!=str(receipt['start_ticks']) or fields[0] in {'Z','X'}, 'Wait for exact training actor to terminate'
    result=read(folder/'results.json');plan=read(folder/'plan.json')
    assert file_hash(folder/'plan.json')==result['plan_sha256']
    assert all(file_hash(Path(p))==h for p,h in result['source_files_sha256'].items())
    assert file_hash(folder/'CE-source-schedule.json')==plan['CE_schedule_sha256']
    assert file_hash(folder/'rollout-source-schedule.json')==plan['rollout_schedule_sha256']
    selected=read(ROOT/'reports/code-residual-lr-pilot-v1/selected-original-rows.json')
    train,validation=selected['train'],selected['validation'];rows={r['id']:r for r in train+validation}
    assert len(train)==512 and len(validation)==128 and len(rows)==640
    assert not ({r['group_id'] for r in train}&{r['group_id'] for r in validation})
    token=Tokenizer.from_file(str(ROOT/'checkpoints/code-residual-lr-pilot-v1/tokenizer.json'))
    inputs,references,prompts={},{},{}
    with gzip.open(ROOT/'data/codenet-durable-pack-v2/original-rows.jsonl.gz','rt') as stream:
        for position,line in enumerate(stream):
            row=json.loads(line)
            if row['id'] not in rows:continue
            expected=rows[row['id']]
            assert row['split']==expected['role'] and row['group_id']==expected['group_id']
            assert position==row['token_pack_index']==expected['global_position']
            assert hashlib.sha256(row['response'].encode()).hexdigest()==row['original_source_body_sha256']
            prompt=row['prompt'] if row['prompt'].rstrip().endswith('Assistant:') else 'User: '+row['prompt']+'\nAssistant:'
            prefix=[1]+token.encode(prompt).ids;reference=token.encode(row['response']).ids+[2]
            ids=prefix+reference
            assert hashlib.sha256(np.asarray(ids,dtype='<u2').tobytes()).hexdigest()==row['complete_input_ids_sha256']==expected['full_input_ids_sha256']
            assert len(ids)==expected['context_tokens']<=4096 and len(reference)==expected['target_tokens']
            inputs[row['id']],references[row['id']],prompts[row['id']]=ids,reference,prefix
    assert len(inputs)==640
    schedule=read(folder/'CE-source-schedule.json')['update_train_ids']
    assert schedule==[[r['id'] for r in train[i:i+4]] for i in range(0,512,4)]
    rollout_rows=read(folder/'rollout-source-schedule.json')['original_train_rows']
    assert len(rollout_rows)==len({r['id'] for r in rollout_rows})==128
    assert len({r['language'] for r in rollout_rows})==54
    assert all(r['role']=='train' and r==rows[r['id']] and len(prompts[r['id']])+128<=4096 for r in rollout_rows)
    base=torch.load(plan['original_base_checkpoint'],map_location='cpu',weights_only=True,mmap=True)
    base_digest=state_digest(base['model'])
    with torch.device('meta'):layout=TinyMultimodalLM(ModelConfig(**base['config']))
    assert set(layout.state_dict())==set(base['model'])
    assert all(layout.state_dict()[k].shape==v.shape for k,v in base['model'].items())
    assert sum(p.numel() for p in layout.parameters())==706012672
    assert layout.lm_head.weight is layout.embedding.weight
    assert torch.equal(base['model']['embedding.weight'],base['model']['lm_head.weight'])
    starting=torch.load(plan['starting_policy_checkpoint'],map_location='cpu',weights_only=True,mmap=True)
    initial_digest=state_digest(starting['residuals']);del starting
    names=[f'{i}.{name}.weight' for i in range(24) for name in ['down','up']]
    all_summaries=[];total_targets=0;rollout_tokens=0;negative_tokens=0
    assert [r['profile'] for r in result['profiles']]==plan['profiles']==[
        dict(name='CE_control',alpha=0.),dict(name='CE_plus_repetition_UL',alpha=.1)]
    for profile in result['profiles']:
        p=profile['profile'];path=Path(profile['checkpoint']);sub=folder/p['name']
        assert file_hash(path)==profile['checkpoint_sha256'] and path.stat().st_size==profile['checkpoint_bytes']<=7*1024**3//4
        frozen=read(sub/'frozen-final-state.json')
        assert frozen['checkpoint_sha256']==profile['checkpoint_sha256'] and frozen['fixed_before_source_generation_or_tests']
        state=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        assert state['format']=='own-code-residual-training-state-v1' and state['stage']=='code_repetition_objective_pilot_v1'
        assert state['steps']==state['planned_steps']==profile['actual_updates']==128
        assert state['objective_profile']==p and state['run_plan_sha256']==result['plan_sha256']
        assert state['base_checkpoint_sha256']==plan['original_base_checkpoint_sha256']
        assert state['source_manifest_sha256']==file_hash(ROOT/'data/codenet-durable-pack-v2/manifest.json')
        assert state['starting_policy_checkpoint_sha256']==plan['starting_policy_sha256']
        assert state['original_selection_sha256']==plan['original_selection_sha256']
        assert state['training_sources']==512 and state['training_rollout_sources']==128
        assert state['rollout_generation_budget']==128 and state['regularizer_ngram']==16
        assert state['reference_repeated_ngrams_exempt'] and state['fresh_optimizer']
        assert state['frozen_backbone_state_sha256']==base_digest==profile['original_base_state_sha256']==profile['final_base_state_sha256']
        assert state['model_counts']==dict(total_unique=857007616,frozen_base=706012672,trainable_code=150994944)
        assert state['residual_parameter_names']==names and set(state['residuals'])==set(names)
        assert sum(t.numel() for t in state['residuals'].values())==150994944
        assert profile['initial_residual_state_sha256']==initial_digest
        assert state_digest(state['residuals'])==profile['final_residual_state_sha256']!=initial_digest
        opt=state['optimizer'];assert opt['param_groups'][0]['params']==list(range(48)) and set(opt['state'])==set(range(48))
        for i,name in enumerate(names):
            tensor=state['residuals'][name];moments=opt['state'][i]
            assert tensor.shape==((2048,1536) if '.down.' in name else (1536,2048)) and tensor.dtype==torch.float32
            assert bool(torch.isfinite(tensor).all())
            assert set(moments)=={'step','exp_avg','exp_avg_sq'} and float(moments['step'])==128
            for key in ['exp_avg','exp_avg_sq']:
                assert moments[key].shape==tensor.shape and moments[key].dtype==tensor.dtype and bool(torch.isfinite(moments[key]).all())
            assert bool((moments['exp_avg_sq']>=0).all())
        rng=state['rng'];assert set(rng)=={'python','numpy','torch','cuda'}
        random.Random().setstate(rng['python']);torch.Generator().set_state(rng['torch'])
        n=rng['numpy'];np.random.RandomState().set_state((n['kind'],np.asarray(n['keys'],dtype=np.uint32),n['position'],n['has_gauss'],n['cached_gaussian']))
        assert rng['cuda'].dtype==torch.uint8 and rng['cuda'].ndim==1 and rng['cuda'].numel()>0
        assert file_hash(sub/'training-rollouts.json')==profile['training_rollouts_sha256']
        rollouts=read(sub/'training-rollouts.json')['records'];metrics=state['metrics']
        assert [m['step'] for m in metrics]==[r['step'] for r in rollouts]==list(range(1,129))
        for metric,rollout,source,group in zip(metrics,rollouts,rollout_rows,schedule):
            assert metric['response_targets']==sum(rows[i]['target_tokens'] for i in group)
            assert metric['UL_alpha']==p['alpha'] and metric['learning_rate']==3e-5*min(1.,metric['step']/16.)
            assert rollout['id']==source['id'] and rollout['role']=='train' and rollout['language']==source['language']
            assert rollout['prompt_tokens']==len(prompts[source['id']])
            assert rollout['prompt_input_ids_sha256']==hashlib.sha256(np.asarray(prompts[source['id']],dtype='<u2').tobytes()).hexdigest()
            generated=rollout['generated_token_ids'];reference=references[source['id']]
            assert 0<len(generated)<=rollout['generation_budget']==128 and 2 not in generated[:-1]
            assert all(type(t) is int and 0<=t<32768 for t in generated)
            # Independent occurrence grouping rather than the online seen-set selector.
            occurrences=defaultdict(list)
            for start in range(len(generated)-15):occurrences[tuple(generated[start:start+16])].append(start+15)
            reference_counts=Counter(tuple(reference[i:i+16]) for i in range(len(reference)-15))
            negatives=sorted(end for gram,ends in occurrences.items() if reference_counts[gram]<2 for end in ends[1:] if generated[end]!=2)
            assert negatives==rollout['negative_generated_token_positions']
            assert metric['negative_positions']==len(negatives)
            assert metric['applied_UL_gradient']==rollout['applied_UL_gradient']==bool(negatives and p['alpha']>0)
            assert metric['mean_UL_before_update']==rollout['mean_UL'] and np.isfinite(rollout['mean_UL'])
            assert np.isfinite(metric['mean_CE_before_update']) and np.isfinite(metric['gradient_norm_before_clipping'])
            raw=token.decode(generated[:-1] if generated[-1]==2 else generated,skip_special_tokens=False)
            assert raw==rollout['raw_text'] and hashlib.sha256(raw.encode()).hexdigest()==rollout['raw_text_sha256']
        targets=sum(m['response_targets'] for m in metrics);total_targets+=targets
        tokens=sum(len(r['generated_token_ids']) for r in rollouts);rollout_tokens+=tokens
        negatives=sum(len(r['negative_generated_token_positions']) for r in rollouts);negative_tokens+=negatives
        assert targets==profile['response_target_exposures']==304575
        assert tokens==profile['own_rollout_tokens'] and profile['own_rollouts']==128
        assert sum(bool(r['negative_generated_token_positions']) for r in rollouts)==profile['rollouts_with_negatives']
        assert sum(m['applied_UL_gradient'] for m in metrics)==profile['actual_UL_gradient_updates']
        for role,selected_rows in [('train',train),('validation',validation)]:
            scores=profile[role]['rows'];assert {r['id'] for r in scores}=={r['id'] for r in selected_rows}
            assert all(r['role']==role and r['targets']==rows[r['id']]['target_tokens'] for r in scores)
            assert profile[role]['token_mean_NLL']==sum(r['targets']*r['mean_NLL'] for r in scores)/sum(r['targets'] for r in scores)
        assert file_hash(sub/'generations.json')==profile['generation_report_sha256']
        generations=read(sub/'generations.json')['records']
        assert len(generations)==len({g['id'] for g in generations})==8
        assert {g['id'] for g in generations}=={r['id'] for r in plan['source_controls']}
        repetition=[]
        for g in generations:
            actual=g['generated_token_ids'];reference=references[g['id']]
            assert 0<len(actual)<=g['generation_budget']==1024 and 2 not in actual[:-1]
            prefix=0
            for x,y in zip(actual,reference):
                if x!=y:break
                prefix+=1
            assert g['matching_reference_prefix_tokens']==prefix
            assert g['exact_complete_reference_token_match']==(actual==reference)
            assert g['first_reference_token_match']==(actual[0]==reference[0])
            assert g['terminal_EOS']==(actual[-1]==2)
            assert g['unique_contiguous_token_4grams']==len({tuple(actual[i:i+4]) for i in range(len(actual)-3)})
            raw=token.decode(actual[:-1] if actual[-1]==2 else actual,skip_special_tokens=False)
            assert raw==g['raw_program_text'] and hashlib.sha256(raw.encode()).hexdigest()==g['raw_program_sha256']
            windows=[tuple(actual[i:i+16]) for i in range(len(actual)-15)]
            repetition.append(dict(id=g['id'],role=g['role'],generated_tokens=len(actual),terminal_EOS=actual[-1]==2,
                repeated16gram_fraction=(len(windows)-len(set(windows)))/len(windows) if windows else 0.))
        for role in ['train','validation']:
            gs=[g for g in generations if g['role']==role]
            assert all(g['role']==rows[g['id']]['role'] for g in gs)
            assert profile['generation_summaries'][role]==dict(sources=4,
                exact_reference_matches=sum(g['exact_complete_reference_token_match'] for g in gs),terminal_EOS=sum(g['terminal_EOS'] for g in gs))
        all_summaries.append(dict(profile=p,checkpoint_sha256=profile['checkpoint_sha256'],
            checkpoint_bytes=profile['checkpoint_bytes'],train_NLL=profile['train']['token_mean_NLL'],
            validation_NLL=profile['validation']['token_mean_NLL'],own_rollout_tokens=tokens,
            selected_repetition_negative_tokens=negatives,actual_UL_gradient_updates=profile['actual_UL_gradient_updates'],
            generation_summaries=profile['generation_summaries'],final_source_repetition_and_lengths=repetition))
        del state,opt
    assert result['actual_Adam_updates_total']==256 and result['complete_original_train_exposures']==1024
    assert total_targets==result['complete_original_response_target_exposures']==609150
    cuda=read(ROOT/'reports/code-repetition-objective-cuda-v1/results.json')
    assert cuda['optimizer_updates']==0 and cuda['full_policy_state_sha256_before']==cuda['full_policy_state_sha256_after']
    assert len(cuda['gradients'])==48 and all(g['present'] and g['finite'] and g['nonzero'] for g in cuda['gradients'])
    assert all(file_hash(Path(p))==h for p,h in cuda['source_files_sha256'].items())
    old=read(ROOT/'reports/code-residual-repeated-source-pilot-v1/results.json')['observations'][1]
    replay_delta=dict(train_NLL=all_summaries[0]['train_NLL']-old['train']['token_mean_NLL'],
        validation_NLL=all_summaries[0]['validation_NLL']-old['validation']['token_mean_NLL'])
    atomic_json(output,dict(verified_at=datetime.now(timezone.utc).isoformat(),source=str(Path(__file__)),
        source_sha256=file_hash(Path(__file__)),original_whole_sources_directly_reencoded=640,
        profiles=all_summaries,actual_Adam_updates_rederived=256,original_CE_programme_exposures=1024,
        original_CE_response_targets_rederived=total_targets,own_train_prompt_rollouts_rederived=256,
        own_rollout_tokens_rederived=rollout_tokens,selected_repetition_negative_positions_rederived=negative_tokens,
        all48_residual_Adam_inventories_per_profile_verified=True,complete_RNG_inventory_verified_without_GPU_restore=True,
        total_unique_parameters_rederived=857007616,original_frozen_base_full_digest_rederived=True,
        actual_CUDA_UL_only_nonzero_gradient_check_verified=True,CE_control_delta_from_earlier_matched_first_pass=replay_delta,
        full_functional_and_QA_assessment_complete=False,candidate_promoted=False,goal_achieved=False,
        scope='Saved sources, candidate masks, raw outputs and complete final state verification, not a replay of every intermediate policy/UL probability or a functional correctness result. Repetition reduction and termination alone do not prove valid or correct code.'))
    print(json.dumps(dict(verified=True,profiles=[{k:v for k,v in r.items() if k!='final_source_repetition_and_lengths'} for r in all_summaries],
        CE_control_replay_delta=replay_delta,report=str(output))))


if __name__=='__main__':main()
