"""Small own-neural cross-family optimizer integration; no data admission."""
import copy
from datetime import timedelta
import io
import json
import os
from pathlib import Path

import pytest
import torch
from torch import distributed as dist
from torch.multiprocessing import spawn

from test_global_word_microbatch_stage_v2 import fixture, spec as word_spec, subset
from slm_lab.codenet_checkpoint_v1 import capture_rng, restore_rng
from slm_lab.global_visual_matching_gradients_v1 import digest
from slm_lab.joint_visual_objective_update_v1 import update,make_optimizer,named_groups,optimizer_clocks

OPTIONS=dict(precision='fp32',maximum_examples=4,maximum_padded_tokens=450,matching_weight=.1,
    temperature=.7,CTC_weight=.4,native_word_CE_weight=.7,gradient_clip=1.,gradient_bucket_values=4096)


def adam(view):return make_optimizer(view,learning_rates=dict(vision=1e-4,pointing=1e-4,word=1e-4))


def payload(batch,kind,rank=0,world=1):
    slots=[0] if kind=='native' else [0,1] if kind=='photo' else [0,1,2]
    whole=subset(batch,slots);positions=[17+3*i for i in slots]
    source={k:v for k,v in word_spec(whole,positions).items() if kind=='word' or k not in ['global_word_UTF8_bytes','global_word_UTF8_sha256']}
    source['family']='images:additional_commons' if kind=='native' else 'images:additional_photo_captions' if kind=='photo' else 'words'
    if kind=='photo':source.update(pixel_keys=['image0','image1'],captions=[batch['texts'][i] for i in slots])
    local=list(range(len(slots)))[rank::world]
    if not local:return None,source
    current=subset(whole,local)
    native={k:current[k] for k in ['input_ids','attention_mask','labels','pixel_values']}
    word=dict(word_pixels=current['word_pixels'],texts=current['texts']) if kind=='word' else None
    matching=None
    if kind=='photo':
        sequences=[[1]+row[row!=-100].tolist() for row in current['labels']]
        width=max(map(len,sequences));ids=torch.zeros(len(sequences),width,dtype=torch.long);mask=torch.zeros_like(ids,dtype=torch.bool)
        for i,seq in enumerate(sequences):ids[i,:len(seq)]=torch.tensor(seq);mask[i,:len(seq)]=True
        matching=dict(text_ids=ids,text_mask=mask,pixel_keys=source['pixel_keys'][rank::world],captions=source['captions'][rank::world])
    return dict(native=native,word=word,matching=matching),source


def equal_optimizer(left,right):
    a,b=left.state_dict(),right.state_dict();assert a['param_groups']==b['param_groups'] and set(a['state'])==set(b['state'])
    for key,value in a['state'].items():
        for name,tensor in value.items():assert torch.equal(tensor,b['state'][key][name])


@pytest.mark.parametrize('bf16',[False,True])
def test_all_three_objectives_keep_inactive_clocks_and_resume_exact_next_update(bf16):
    view,batch=fixture();optimizer=adam(view);options={**OPTIONS,'precision':'bf16' if bf16 else 'fp32'}
    selected={id(p) for rows in named_groups(view).values() for _,p in rows}
    fixed={name:p.detach().clone() for name,p in view.actor.named_parameters() if id(p) not in selected}
    with torch.no_grad():text=view.actor(batch['input_ids'],attention_mask=batch['attention_mask'])['logits'].clone()
    records=[]
    for kind,expected in [('native',dict(vision=1,pointing=0,word=0)),('photo',dict(vision=2,pointing=1,word=0))]:
        prepared,source=payload(batch,kind);records.append(update(view,optimizer,prepared,source,options))
        assert optimizer_clocks(view,optimizer)==expected
    stream=io.BytesIO();torch.save(dict(model=view.actor.state_dict(),optimizer=optimizer.state_dict(),rng=capture_rng('cpu')),stream);stream.seek(0)
    saved=torch.load(stream,map_location='cpu',weights_only=True)
    prepared,source=payload(batch,'word');actual=update(view,optimizer,prepared,source,options)
    draw=torch.rand(11)
    resumed,_=fixture();resumed.actor.load_state_dict(saved['model'],strict=True);resumed_optimizer=adam(resumed)
    resumed_optimizer.load_state_dict(saved['optimizer']);restore_rng(saved['rng'],'cpu')
    recovered=update(resumed,resumed_optimizer,prepared,source,options)
    assert actual==recovered and torch.equal(draw,torch.rand(11))
    assert optimizer_clocks(view,optimizer)==dict(vision=3,pointing=1,word=1)
    for name,value in view.actor.state_dict().items():
        assert torch.equal(value,resumed.actor.state_dict()[name])
        if name in fixed:assert torch.equal(value,fixed[name])
    equal_optimizer(optimizer,resumed_optimizer)
    with torch.no_grad():assert torch.equal(text,view.actor(batch['input_ids'],attention_mask=batch['attention_mask'])['logits'])
    assert all(p.grad is None for p in view.parameters())
    assert not torch.cuda.is_initialized()


def test_multiple_updates_inside_ambient_BF16_use_current_weights_and_match_no_ambient():
    results=[]
    for ambient in [False,True]:
        view,batch=fixture();optimizer=adam(view);options={**OPTIONS,'precision':'bf16'}
        with torch.autocast('cpu',dtype=torch.bfloat16,enabled=ambient,cache_enabled=True):
            for kind in ['native','photo','word','photo','native']:
                prepared,source=payload(batch,kind);update(view,optimizer,prepared,source,options)
        results.append((view,optimizer))
    for name,value in results[0][0].actor.state_dict().items():assert torch.equal(value,results[1][0].actor.state_dict()[name])
    equal_optimizer(results[0][1],results[1][1])


def test_wrong_payload_and_optimizer_group_rejected_before_forward_or_clock_change():
    view,batch=fixture();optimizer=adam(view);prepared,source=payload(batch,'native')
    original={name:p.detach().clone() for name,p in view.actor.named_parameters()}
    calls=[];hook=view.native.vision.register_forward_hook(lambda *args:calls.append(1))
    bad=copy.deepcopy(prepared);bad['word']=dict(texts=['unrelated'])
    with pytest.raises(ValueError,match='Coordinated joint visual preflight'):update(view,optimizer,bad,source,OPTIONS)
    optimizer.param_groups[1]['name']='word'
    with pytest.raises(ValueError,match='Coordinated joint visual preflight'):update(view,optimizer,prepared,source,OPTIONS)
    hook.remove();assert not calls and not optimizer.state
    assert all(torch.equal(original[name],p) and p.grad is None for name,p in view.actor.named_parameters())


def worker(rank,folder):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and os.nice(0)>=19
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    dist.init_process_group('gloo',init_method='file://'+str(Path(folder)/'rendezvous'),rank=rank,world_size=4,timeout=timedelta(minutes=5))
    try:
        view,batch=fixture();optimizer=adam(view);records=[]
        selected={id(p) for rows in named_groups(view).values() for _,p in rows}
        fixed={name:p.detach().clone() for name,p in view.actor.named_parameters() if id(p) not in selected}
        for kind in ['native','photo']:
            prepared,source=payload(batch,kind,rank,4)
            record=update(view,optimizer,prepared,source,OPTIONS);records.append(record)
        assert optimizer_clocks(view,optimizer)==dict(vision=2,pointing=1,word=0)
        stream=io.BytesIO();torch.save(dict(model=view.actor.state_dict(),optimizer=optimizer.state_dict(),rng=capture_rng('cpu')),stream);stream.seek(0)
        saved=torch.load(stream,map_location='cpu',weights_only=True)
        prepared,source=payload(batch,'word',rank,4);actual=update(view,optimizer,prepared,source,OPTIONS)
        draw=torch.rand(11)
        restored,_=fixture();restored.actor.load_state_dict(saved['model'],strict=True);restored_optimizer=adam(restored)
        restored_optimizer.load_state_dict(saved['optimizer']);restore_rng(saved['rng'],'cpu')
        recovered=update(restored,restored_optimizer,prepared,source,OPTIONS)
        assert actual==recovered and torch.equal(draw,torch.rand(11));records.append(actual)
        equal_optimizer(optimizer,restored_optimizer)
        assert optimizer_clocks(view,optimizer)==dict(vision=3,pointing=1,word=1)
        for name,value in view.actor.state_dict().items():
            assert torch.equal(value,restored.actor.state_dict()[name])
            if name in fixed:assert torch.equal(value,fixed[name])
        hashes=[None]*4;dist.all_gather_object(hashes,digest(torch.cat([p.detach().reshape(-1) for rows in named_groups(view).values() for _,p in rows])))
        assert len(set(hashes))==1
        # Detect disagreement about the objective before ranks enter different
        # nested collective sequences. No source forward or optimizer commit.
        prepared,source=payload(batch,'native',rank,4)
        if rank==1:source['family']='images:additional_real_words'
        calls=[];hook=view.native.vision.register_forward_hook(lambda *args:calls.append(1))
        before=copy.deepcopy(optimizer.state_dict())
        with pytest.raises(ValueError,match='Coordinated joint visual global family/source/settings disagreement'):
            update(view,optimizer,prepared,source,OPTIONS)
        hook.remove();assert not calls and all(p.grad is None for p in view.parameters())
        after=optimizer.state_dict();assert before['param_groups']==after['param_groups']
        for key,value in before['state'].items():
            for name,tensor in value.items():assert torch.equal(tensor,after['state'][key][name])
        assert not torch.cuda.is_initialized()
        (Path(folder)/(str(rank)+'.json')).write_text(json.dumps(dict(complete=True,rank=rank,
            actual_small_own_model_parameters=view.actor.counts()['actual_unique'],actual_selected_values=sum(p.numel() for rows in named_groups(view).values() for _,p in rows),
            complete_cross_family_records=records,exact_next_Adam_and_RNG_recovery=True,replicas_bitexact=True,
            inactive_head_clocks_moments_weights_preserved=True,family_disagreement_rejected_before_any_forward=True,
            no_extra_learned_parameters=True,CUDA_hidden=True,production_optimizer_updates=0,
            full_size_GPU_source_admission_capacity_private_checkpoint_and_quality_pending=True),indent=2)+'\n')
    finally:dist.destroy_process_group()


def test_four_own_replicas_cross_family_clocks_recovery_and_empty_ranks(tmp_path):
    spawn(worker,args=(str(tmp_path),),nprocs=4,join=True)
    rows=[json.loads((tmp_path/(str(rank)+'.json')).read_text()) for rank in range(4)]
    assert all(row['complete'] and row['exact_next_Adam_and_RNG_recovery'] for row in rows)
    assert all(record['local_objective_record']['actual_native_forward_calls']==0
        for record in rows[3]['complete_cross_family_records'] if record['kind']!='photo')
