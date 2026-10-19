"""Small actual own-model two-pass mechanics; not image capability scores."""
from contextlib import nullcontext
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

from test_global_native_visual_stage_kernel_v1 import fixture, subset, specification
import slm_lab.global_photo_gradient_cache_stage_v1 as cached
from slm_lab.global_native_visual_stage_kernel_v1 import native_microbatches
from slm_lab.global_visual_matching_gradients_v1 import digest
from slm_lab.own_actor_vision_alignment_v1 import pair_positives, symmetric_pair_loss
from slm_lab.own_actor_vision_prefix_reuse_v1 import native_from_visual


def photo_fixture():
    view,batch = fixture()
    ids = torch.zeros(9,5,dtype=torch.long)
    mask = torch.ones_like(ids,dtype=torch.bool)
    for i in range(9):
        start = int((batch['labels'][i] != -100).nonzero()[0])
        body = [12,13,14,2] if i in [1,2] else [30+i,40+i,50+i,2]
        batch['input_ids'][i,start:start+4] = torch.tensor(body)
        batch['labels'][i,start:start+4] = torch.tensor(body)
        ids[i] = torch.tensor([1]+body)
    batch['pixel_values'][1] = batch['pixel_values'][0]
    keys = ['same','same','b']+['image'+str(i) for i in range(3,9)]
    captions = ['cat','animal','animal']+['caption'+str(i) for i in range(3,9)]
    return view,batch,ids,mask,keys,captions


def selected(view):
    return [p for _,p in view.named_selected()]


def optimizer(view):
    return torch.optim.AdamW(selected(view),lr=1e-4,eps=1e-5,weight_decay=.01,foreach=False)


def direct_backward(view,batch,ids,mask,keys,captions,groups,amp=None):
    amp = amp or nullcontext
    total = int((batch['labels'][:,1:] != -100).sum())
    parts = []
    images_by_source = {}
    texts_by_source = {}
    for slots in groups:
        length = int(batch['attention_mask'][slots].sum(1).max())
        current = {k:v[slots] if k == 'pixel_values' else v[slots,:length] for k,v in batch.items()}
        with torch.no_grad(),amp():
            hidden = view.actor.policy(ids[slots],attention_mask=mask[slots],return_hidden=True,return_logits=False)['hidden']
            content = mask[slots] & (ids[slots] != 1) & (ids[slots] != 2)
            text = (hidden.float()*content[:,:,None]).sum(1)/content.sum(1,keepdim=True)
        with amp():
            prefix = view.native.vision(current['pixel_values'],True)
            image = view.actor.base.heads['pointing']['patch'](prefix.mean(1))
            text = view.actor.base.heads['pointing']['query'](text.to(prefix.dtype))
            output = native_from_visual(view.native,current['input_ids'],prefix,
                attention_mask=current['attention_mask'],labels=current['labels'])
        count = int((current['labels'][:,1:] != -100).sum())
        parts.append((image,text,output['loss'],count))
        for offset,source in enumerate(slots):
            images_by_source[source] = image[offset]
            texts_by_source[source] = text[offset]
    # Keep all small reference graphs, independently evaluate the complete
    # pair loss in original global order, then apply each separate VJP.
    all_images = torch.stack([images_by_source[i] for i in range(len(ids))])
    all_texts = torch.stack([texts_by_source[i] for i in range(len(ids))])
    loss = symmetric_pair_loss(all_images,all_texts,pair_positives(keys,captions),temperature=.7)
    derivatives = torch.autograd.grad(.1*loss,[v for p in parts for v in p[:2]],retain_graph=True)
    ce_sum = 0.
    for index,(image,text,ce,count) in enumerate(parts):
        torch.autograd.backward([image,text],grad_tensors=derivatives[2*index:2*index+2],inputs=selected(view),retain_graph=True)
        (ce*count/total).backward(inputs=selected(view))
        ce_sum += float(ce.detach())*count
    return float(loss.detach()),ce_sum


@pytest.mark.parametrize('bf16',[False,True])
@pytest.mark.parametrize('outer_autocast',[False,True])
def test_cache_replay_matches_attached_microbatch_reference_with_full_global_negatives(bf16,outer_autocast):
    view,batch,ids,mask,keys,captions = photo_fixture()
    reference = copy.deepcopy(view)
    positions = [17+3*i for i in range(9)]
    spec = specification(batch,positions)
    amp = (lambda:torch.autocast('cpu',dtype=torch.bfloat16)) if bf16 else None
    outer = (lambda:torch.autocast('cpu',dtype=torch.bfloat16,cache_enabled=True)) if bf16 and outer_autocast else nullcontext
    with outer():
        record = cached.accumulate_global_photo_cached(view,batch,ids,mask,local_positions=positions,
            pixel_keys=keys,captions=captions,maximum_padded_tokens=450,temperature=.7,
            feature_dtype=torch.bfloat16 if bf16 else torch.float32,
            native_logits_dtype=torch.bfloat16 if bf16 else torch.float32,amp=amp,**spec)
    groups = native_microbatches(spec['global_context_lengths'],prefix_tokens=196,maximum_padded_tokens=450)
    with outer():
        loss,ce = direct_backward(reference,batch,ids,mask,keys,captions,groups,amp)
    assert abs(record['global_matching']['unweighted_loss']-loss) < 5e-6
    assert abs(record['local_caption_CE_sum']-ce) < 5e-5
    assert record['actual_graph_free_encoder_calls'] == record['actual_replay_encoder_calls'] == len(groups)
    assert record['source_feature_replay_bitexact']
    error = max(float((a.grad-b.grad).abs().max()) for a,b in zip(selected(view),selected(reference)))
    assert error < 5e-6
    chosen = {id(p) for p in selected(view)}
    assert all(p.grad is None for p in view.parameters() if id(p) not in chosen)
    left,right = optimizer(view),optimizer(reference)
    torch.nn.utils.clip_grad_norm_(selected(view),1.,error_if_nonfinite=True)
    torch.nn.utils.clip_grad_norm_(selected(reference),1.,error_if_nonfinite=True)
    left.step();right.step()
    assert max(float((a-b).detach().abs().max()) for a,b in zip(selected(view),selected(reference))) < 5e-6
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('bf16',[False,True])
def test_full_next_Adam_update_recovers_exactly(bf16):
    view,batch,ids,mask,keys,captions = photo_fixture()
    adam = optimizer(view)
    positions = [17+3*i for i in range(9)]
    spec = specification(batch,positions)
    amp = (lambda:torch.autocast('cpu',dtype=torch.bfloat16)) if bf16 else None
    def update(current,opt,source):
        record = cached.accumulate_global_photo_cached(current,source,ids,mask,local_positions=positions,
            pixel_keys=keys,captions=captions,maximum_padded_tokens=450,temperature=.7,
            feature_dtype=torch.bfloat16 if bf16 else torch.float32,
            native_logits_dtype=torch.bfloat16 if bf16 else torch.float32,amp=amp,**spec)
        torch.nn.utils.clip_grad_norm_(selected(current),1.,error_if_nonfinite=True)
        opt.step();opt.zero_grad(set_to_none=True)
        return record
    update(view,adam,batch)
    stream = io.BytesIO()
    torch.save(dict(model=view.actor.state_dict(),optimizer=adam.state_dict()),stream)
    stream.seek(0)
    state = torch.load(stream,map_location='cpu',weights_only=True)
    fresh,*_ = photo_fixture()
    fresh.actor.load_state_dict(state['model'],strict=True)
    resumed = optimizer(fresh);resumed.load_state_dict(state['optimizer'])
    changed = {**batch,'pixel_values':batch['pixel_values']*.8}
    assert update(view,adam,changed) == update(fresh,resumed,changed)
    for name,value in view.actor.state_dict().items():
        assert torch.equal(value,fresh.actor.state_dict()[name])
    for a,b in zip(adam.state.values(),resumed.state.values()):
        for name,value in a.items():assert torch.equal(value,b[name])


def test_stale_replayed_features_reject_before_Adam_and_clear_partial_gradients():
    view,batch,ids,mask,keys,captions = photo_fixture()
    spec = specification(batch,list(range(9)))
    chunks = native_microbatches(spec['global_context_lengths'],prefix_tokens=196,maximum_padded_tokens=450)
    calls = []
    def alter(module,args,output):
        calls.append(1)
        return output+.5 if len(calls) == len(chunks)+1 else output
    hook = view.native.vision.register_forward_hook(alter)
    original = {name:p.detach().clone() for name,p in view.actor.named_parameters()}
    with pytest.raises(ValueError,match='recomputed features differ'):
        cached.accumulate_global_photo_cached(view,batch,ids,mask,local_positions=list(range(9)),
            pixel_keys=keys,captions=captions,maximum_padded_tokens=450,temperature=.7,**spec)
    hook.remove()
    assert all(p.grad is None for p in view.parameters())
    assert all(torch.equal(original[name],p) for name,p in view.actor.named_parameters())


def worker(rank,folder):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and os.nice(0) >= 19
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    dist.init_process_group('gloo',init_method='file://'+str(Path(folder)/'rendezvous'),
        rank=rank,world_size=4,timeout=timedelta(minutes=4))
    try:
        view,batch,ids,mask,keys,captions = photo_fixture()
        reference = copy.deepcopy(view)
        adam,refadam = optimizer(view),optimizer(reference)
        chosen = {id(p) for p in selected(view)}
        fixed = {name:p.detach().clone() for name,p in view.actor.named_parameters() if id(p) not in chosen}
        proofs = []
        for slots in [list(range(9)),[0,2]]:
            whole = subset(batch,slots)
            whole_ids,whole_mask = ids[slots],mask[slots]
            whole_keys,whole_captions = [keys[i] for i in slots],[captions[i] for i in slots]
            positions = [17+3*i for i in slots]
            local = list(range(len(slots)))[rank::4]
            payload = subset(whole,local) if local else None
            spec = specification(whole,positions)
            calls = [];hook = view.native.vision.register_forward_hook(lambda *args:calls.append(1))
            record = cached.accumulate_global_photo_cached(view,payload,whole_ids[local] if local else None,
                whole_mask[local] if local else None,local_positions=positions[rank::4],pixel_keys=whole_keys,
                captions=whole_captions,maximum_padded_tokens=450,temperature=.7,**spec)
            hook.remove()
            assert len(calls) == record['actual_graph_free_encoder_calls']+record['actual_replay_encoder_calls']
            if not local:assert not calls and all(p.grad is None for p in selected(view))
            for p in selected(view):
                if p.grad is None:p.grad = torch.zeros_like(p)
                dist.all_reduce(p.grad,op=dist.ReduceOp.SUM)
            groups = []
            for owner in range(4):
                owned = list(range(len(slots)))[owner::4]
                chunks = native_microbatches([spec['global_context_lengths'][i] for i in owned],prefix_tokens=196,maximum_padded_tokens=450)
                groups.extend([[owned[i] for i in chunk] for chunk in chunks])
            loss,ce = direct_backward(reference,whole,whole_ids,whole_mask,whole_keys,whole_captions,groups)
            gradient_error = max(float((a.grad-b.grad).abs().max()) for a,b in zip(selected(view),selected(reference)))
            assert gradient_error < 5e-6
            assert abs(record['global_matching']['unweighted_loss']-loss) < 5e-6
            sums = [None]*4;dist.all_gather_object(sums,record['local_caption_CE_sum'])
            assert abs(sum(sums)-ce) < 5e-5
            torch.nn.utils.clip_grad_norm_(selected(view),1.,error_if_nonfinite=True)
            torch.nn.utils.clip_grad_norm_(selected(reference),1.,error_if_nonfinite=True)
            adam.step();refadam.step();adam.zero_grad(set_to_none=True);refadam.zero_grad(set_to_none=True)
            error = max(float((a-b).detach().abs().max()) for a,b in zip(selected(view),selected(reference)))
            assert error < 5e-6
            for name,p in view.actor.named_parameters():
                if name in fixed:assert torch.equal(fixed[name],p) and p.grad is None
            hashes = [None]*4
            dist.all_gather_object(hashes,digest(torch.cat([p.detach().reshape(-1) for p in selected(view)])))
            assert len(set(hashes)) == 1
            proofs.append(dict(rank=rank,global_sources=len(slots),local_sources=len(local),
                actual_own_unique_parameters=view.actor.counts()['actual_unique'],
                actual_selected_values=sum(p.numel() for p in selected(view)),
                serial_shard_max_gradient_absolute_difference=gradient_error,
                serial_shard_max_Adam_weight_absolute_difference=error,
                all_replicas_bitexact=True,all_fixed_weights_bitexact=True,record=record))
        # One rank's incomplete matching EOS must reject everyone before any
        # graph-free or gradient-bearing image encoder call.
        source = subset(batch,[0,1,2])
        spec = specification(source,[17,20,23])
        current = subset(source,[rank]) if rank < 3 else None
        current_ids = ids[rank:rank+1].clone() if rank < 3 else None
        if rank == 1:current_ids[0,-1] = 4
        calls = [];hook = view.native.vision.register_forward_hook(lambda *args:calls.append(1))
        with pytest.raises(ValueError,match='Coordinated cached photo source/native preflight'):
            cached.accumulate_global_photo_cached(view,current,current_ids,mask[rank:rank+1] if rank < 3 else None,
                local_positions=[17,20,23][rank::4],pixel_keys=keys[:3],captions=captions[:3],
                maximum_padded_tokens=450,temperature=.7,**spec)
        hook.remove();assert not calls and all(p.grad is None for p in view.parameters())
        assert not torch.cuda.is_initialized()
        (Path(folder)/(str(rank)+'.json')).write_text(json.dumps(dict(complete=True,proofs=proofs,
            one_rank_incomplete_caption_rejected_everywhere_before_any_encoder_call=True,
            CUDA_hidden=True,production_optimizer_updates=0,full_size_GPU_capacity_and_quality_pending=True),indent=2)+'\n')
    finally:dist.destroy_process_group()


def test_four_actual_own_replicas_global_pairs_microbatch_replay_and_empty_tail(tmp_path):
    spawn(worker,args=(str(tmp_path),),nprocs=4,join=True)
    rows = [json.loads((tmp_path/(str(rank)+'.json')).read_text()) for rank in range(4)]
    assert all(row['complete'] and len(row['proofs']) == 2 for row in rows)
    assert all(row['proofs'][1]['record']['actual_replay_encoder_calls'] == 0 for row in rows[2:])
