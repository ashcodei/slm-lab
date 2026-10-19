"""Reencode all640 original rows and reconcile all2048 update exposures/state."""
from collections import Counter
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


def read(path):
    return json.loads(path.read_text())


def main():
    torch.set_num_threads(1)
    folder=ROOT/'reports/code-residual-repeated-source-pilot-v1'
    target=folder/'independent-complete-readback.json'
    if target.exists():
        raise ValueError('Preserve previous independent readback')
    launch=read(ROOT/'reports/code-residual-repeated-source-pilot-v1-launch.json')
    proc=Path('/proc')/str(launch['pid'])/'stat'
    if proc.exists():
        fields=proc.read_text().rsplit(')',1)[1].split()
        assert fields[19]!=str(launch['start_ticks']) or fields[0] in {'Z','X'}, 'Wait for exact actor to terminate'
    result=read(folder/'results.json');plan=read(folder/'plan.json')
    bound=result['source_files_sha256']
    assert all(file_hash(Path(n))==h for n,h in bound.items())
    assert file_hash(folder/'plan.json')==result['plan_sha256']
    assert file_hash(folder/'training-source-schedule.json')==plan['schedule_sha256']
    assert file_hash(folder/'generations.json')==result['generation_report_sha256']
    old=ROOT/'reports/code-residual-lr-pilot-v1'
    selected=read(old/'selected-original-rows.json')
    train,validation=selected['train'],selected['validation']
    all_rows={r['id']:r for r in train+validation}
    assert len(train)==512 and len(validation)==128 and len(all_rows)==640
    assert not ({r['group_id'] for r in train}&{r['group_id'] for r in validation})
    assert len({r['language'] for r in train})==54
    token=Tokenizer.from_file(str(ROOT/'checkpoints/code-residual-repeated-source-pilot-v1/tokenizer.json'))
    references,matched={},set()
    with gzip.open(ROOT/'data/codenet-durable-pack-v2/original-rows.jsonl.gz','rt') as stream:
        for position,line in enumerate(stream):
            row=json.loads(line)
            if row['id'] not in all_rows:
                continue
            expected=all_rows[row['id']]
            assert row['split']==expected['role'] and row['group_id']==expected['group_id']
            assert position==row['token_pack_index']==expected['global_position']
            assert hashlib.sha256(row['response'].encode()).hexdigest()==row['original_source_body_sha256']
            prompt=row['prompt'] if row['prompt'].rstrip().endswith('Assistant:') else 'User: '+row['prompt']+'\nAssistant:'
            reference=token.encode(row['response']).ids+[2]
            ids=[1]+token.encode(prompt).ids+reference
            assert hashlib.sha256(np.asarray(ids,dtype='<u2').tobytes()).hexdigest()==expected['full_input_ids_sha256']==row['complete_input_ids_sha256']
            assert len(ids)==expected['context_tokens']<=4096 and len(reference)==expected['target_tokens']
            references[row['id']]=reference;matched.add(row['id'])
    assert matched==set(all_rows)
    schedule=read(folder/'training-source-schedule.json')['update_train_ids']
    assert len(schedule)==512 and all(len(ids)==4 for ids in schedule)
    expected_training_ids={r['id'] for r in train}
    for p in range(4):
        ids=[x for group in schedule[p*128:(p+1)*128] for x in group]
        assert len(ids)==512 and set(ids)==expected_training_ids and len(set(ids))==512
    checkpoint=Path(result['checkpoint'])
    assert file_hash(checkpoint)==result['checkpoint_sha256']
    assert checkpoint.stat().st_size==result['checkpoint_bytes']<=2*1024**3
    state=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
    assert state['format']=='own-code-residual-training-state-v1'
    assert state['stage']=='code_residual_repeated_source_pilot_v1'
    assert state['steps']==state['planned_steps']==result['actual_updates']==512
    assert state['run_plan_sha256']==result['plan_sha256']
    assert state['base_checkpoint_sha256']==plan['original_base_checkpoint_sha256']
    assert state['source_manifest_sha256']==file_hash(ROOT/'data/codenet-durable-pack-v2/manifest.json')
    assert state['starting_policy_checkpoint_sha256']==plan['starting_policy_sha256']
    assert state['original_selection_sha256']==plan['original_selection_sha256']
    assert state['fresh_optimizer'] and state['training_sources']==512 and state['additional_complete_passes']==4
    names=[f'{i}.{name}.weight' for i in range(24) for name in ['down','up']]
    assert state['residual_parameter_names']==names and set(state['residuals'])==set(names)
    assert sum(t.numel() for t in state['residuals'].values())==150994944
    assert state_digest(state['residuals'])==result['final_residual_state_sha256']
    assert result['initial_residual_state_sha256']!=result['final_residual_state_sha256']
    starting=torch.load(plan['starting_policy_checkpoint'],map_location='cpu',weights_only=True,mmap=True)
    assert state_digest(starting['residuals'])==result['initial_residual_state_sha256']
    del starting
    opt=state['optimizer']
    assert opt['param_groups'][0]['params']==list(range(48)) and set(opt['state'])==set(range(48))
    for i,name in enumerate(names):
        tensor=state['residuals'][name];moments=opt['state'][i]
        assert tensor.shape==((2048,1536) if '.down.' in name else (1536,2048))
        assert tensor.dtype==torch.float32 and bool(torch.isfinite(tensor).all())
        assert float(moments['step'])==512 and set(moments)=={'step','exp_avg','exp_avg_sq'}
        for key in ['exp_avg','exp_avg_sq']:
            assert moments[key].shape==tensor.shape and moments[key].dtype==tensor.dtype and bool(torch.isfinite(moments[key]).all())
        assert bool((moments['exp_avg_sq']>=0).all())
    rng=state['rng'];assert set(rng)=={'python','numpy','torch','cuda'}
    random.Random().setstate(rng['python']);torch.Generator().set_state(rng['torch'])
    n=rng['numpy'];np.random.RandomState().set_state((n['kind'],np.asarray(n['keys'],dtype=np.uint32),n['position'],n['has_gauss'],n['cached_gaussian']))
    assert rng['cuda'].dtype==torch.uint8 and rng['cuda'].ndim==1 and rng['cuda'].numel()>0
    metrics=state['metrics'];assert [m['step'] for m in metrics]==list(range(1,513))
    for m,ids in zip(metrics,schedule):
        assert m['response_targets']==sum(all_rows[i]['target_tokens'] for i in ids)
        assert m['pass_number']==(m['step']-1)//128+1
        assert np.isfinite(m['mean_NLL_before_update']) and np.isfinite(m['gradient_norm_before_clipping'])
        assert m['learning_rate']==3e-5*min(1.,m['step']/16.)
    assert sum(m['response_targets'] for m in metrics)==result['response_target_exposures']==1218300
    base=torch.load(plan['original_base_checkpoint'],map_location='cpu',weights_only=True,mmap=True)
    assert state_digest(base['model'])==state['frozen_backbone_state_sha256']==result['original_base_state_sha256']==result['final_base_state_sha256']
    # Serialization may copy the two names of a tied parameter separately.
    # Parameter count follows the actual architecture, not saved tensor addresses.
    with torch.device('meta'):
        layout=TinyMultimodalLM(ModelConfig(**base['config']))
    assert set(layout.state_dict())==set(base['model'])
    assert all(layout.state_dict()[k].shape==v.shape for k,v in base['model'].items())
    assert layout.lm_head.weight is layout.embedding.weight
    assert torch.equal(base['model']['embedding.weight'],base['model']['lm_head.weight'])
    unique_base=sum(t.numel() for t in layout.parameters())
    assert unique_base==706012672
    assert state['model_counts']==result['parameter_counts']==dict(total_unique=857007616,frozen_base=706012672,trainable_code=150994944)
    observations=result['observations'];assert [o['step'] for o in observations]==[0,128,256,512]
    for o in observations:
        for role,selected_rows in [('train',train),('validation',validation)]:
            rows=o[role]['rows'];assert {r['id'] for r in rows}=={r['id'] for r in selected_rows}
            assert all(r['role']==role and r['targets']==all_rows[r['id']]['target_tokens'] for r in rows)
            assert o[role]['token_mean_NLL']==sum(r['targets']*r['mean_NLL'] for r in rows)/sum(r['targets'] for r in rows)
    generations=read(folder/'generations.json')['records']
    assert len(generations)==len({g['id'] for g in generations})==8
    assert {g['id'] for g in generations}=={r['id'] for r in plan['source_controls']}
    for g in generations:
        actual=g['generated_token_ids'];reference=references[g['id']]
        assert 0<len(actual)<=g['generation_budget']==1024 and 2 not in actual[:-1]
        prefix=0
        for x,y in zip(actual,reference):
            if x!=y:break
            prefix+=1
        assert g['matching_reference_prefix_tokens']==prefix
        assert g['exact_complete_reference_token_match']==(actual==reference)
        assert g['terminal_EOS']==(actual[-1]==2)
        assert g['first_reference_token_match']==(actual[0]==reference[0])
        assert g['unique_contiguous_token_4grams']==len({tuple(actual[i:i+4]) for i in range(len(actual)-3)})
        raw=token.decode(actual[:-1] if actual[-1]==2 else actual,skip_special_tokens=False)
        assert raw==g['raw_program_text'] and hashlib.sha256(raw.encode()).hexdigest()==g['raw_program_sha256']
    for role in ['train','validation']:
        rows=[g for g in generations if g['role']==role]
        assert all(g['role']==all_rows[g['id']]['role'] for g in rows)
        assert result['generation_summaries'][role]==dict(sources=4,
            exact_reference_matches=sum(g['exact_complete_reference_token_match'] for g in rows),
            terminal_EOS=sum(g['terminal_EOS'] for g in rows))
    atomic_json(target,dict(verified_at=datetime.now(timezone.utc).isoformat(),source=str(Path(__file__)),
        source_sha256=file_hash(Path(__file__)),complete_source_bindings_rehashed=len(bound),
        original_programs_directly_reencoded=640,training_sources=512,training_languages=54,
        validation_sources=128,validation_languages=53,train_validation_group_overlap=0,
        actual_Adam_updates=512,complete_train_programme_exposures_rederived=2048,
        response_target_exposures_rederived=1218300,all48_residual_Adam_states_verified=True,
        RNG_inventory_verified_without_GPU_restore=True,original_base_full_state_rederived=True,
        total_unique_parameters_rederived=857007616,complete_checkpoint_sha256=result['checkpoint_sha256'],
        complete_checkpoint_bytes=result['checkpoint_bytes'],
        initial_train_NLL=observations[0]['train']['token_mean_NLL'],final_train_NLL=observations[-1]['train']['token_mean_NLL'],
        initial_validation_NLL=observations[0]['validation']['token_mean_NLL'],final_validation_NLL=observations[-1]['validation']['token_mean_NLL'],
        generation_summaries=result['generation_summaries'],functional_correctness_assessed=False,
        candidate_promoted=False,goal_achieved=False,
        scope='Independent saved-source/state readback, not an independent inference rerun or functional/broad-knowledge benchmark. All final capability requirements remain.'))
    print(json.dumps(dict(verified=True,original_sources=640,actual_updates=512,training_exposures=2048,
        generation_summaries=result['generation_summaries'],report=str(target))))


if __name__=='__main__':
    main()
