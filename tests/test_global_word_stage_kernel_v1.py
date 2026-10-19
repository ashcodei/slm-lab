"""Small own-neural whole-word mechanics; no OCR or model-quality benchmark."""
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

from slm_lab.model import ModelConfig
from slm_lab.own_dense_growth_v1 import OwnDenseAssistant,new_backbone
from slm_lab.own_dense_actor_v1 import OwnDenseActor
from slm_lab.own_actor_vision_prefix_split_backward_v1 import OwnActorVisionPrefixSplitBackward
from slm_lab.global_word_stage_kernel_v1 import accumulate_global_word,named_word_selected
from slm_lab.shared_word_stage_v1 import ctc_word_sum
from slm_lab.global_visual_matching_gradients_v1 import digest


def fixture():
    torch.set_num_threads(1)
    c = ModelConfig(vocab_size=64,hidden_size=32,intermediate_size=64,num_layers=2,num_heads=4,num_kv_heads=2,
        max_text_length=128,vision_hidden_size=32,vision_intermediate_size=64,vision_layers=1,vision_heads=4,
        image_size=224,patch_size=16)
    model = OwnDenseActor(OwnDenseAssistant(new_backbone(c,994117),reward_hidden=8,word_temporal=4,
        pointing_rank=8,router_hidden=8,detector_hidden=4,word_expansion=5),rank=2,seed=994217)
    view = OwnActorVisionPrefixSplitBackward(model).train();view.native.gradient_checkpointing = True
    view.actor.base.train();view.actor.base.network.gradient_checkpointing = True;view.native.vision.train()
    texts = ['cat','Aé','a'*64];bodies = [[20,21,22],[23,24],[25]*64]
    sequences = [[1,5,6]+body+[2] for body in bodies];width = max(map(len,sequences))
    ids = torch.zeros(3,width,dtype=torch.long);labels = torch.full_like(ids,-100);mask = torch.zeros_like(ids,dtype=torch.bool)
    for i,seq in enumerate(sequences):ids[i,:len(seq)] = torch.tensor(seq);labels[i,3:len(seq)] = torch.tensor(seq[3:]);mask[i,:len(seq)] = True
    g = torch.Generator().manual_seed(994317)
    batch = dict(input_ids=ids,labels=labels,attention_mask=mask,pixel_values=torch.rand(3,3,448,448,generator=g)*2-1,
        word_pixels=torch.rand(3,3,64,448,generator=g)*2-1,texts=texts)
    return view,batch


def subset(batch,slots):
    return {k:[v[p] for p in slots] if k == 'texts' else v[slots] for k,v in batch.items()}


def optimizer(view):return torch.optim.AdamW([p for _,p in named_word_selected(view)],lr=1e-4,eps=1e-5,weight_decay=.01,foreach=False)


def expected_backward(view,batch,amp=None):
    amp = amp or nullcontext;selected = [p for _,p in named_word_selected(view)]
    with amp():logits = view.actor.base.word_logits(batch['word_pixels'])
    ctc,byte_count = ctc_word_sum(logits,batch['texts']);(ctc*.4/len(batch['texts'])).backward(inputs=selected)
    with amp():logits = view.native(**{k:batch[k] for k in ['input_ids','attention_mask','pixel_values']})['logits']
    targets = batch['labels'][:,1:];valid = targets != -100
    ce = torch.nn.functional.cross_entropy(logits[:,:-1].float()[valid],targets[valid],reduction='sum')
    (ce*.7/int(valid.sum())).backward(inputs=selected)
    return float(ctc.detach()),float(ce.detach()),byte_count


def test_actual_serial_whole_word_gradients_match_existing_CTC_and_native_objective():
    view,batch = fixture();reference = copy.deepcopy(view)
    positions = [17,23,29];targets = [int((row[1:] != -100).sum()) for row in batch['labels']];byte_lengths = [len(s.encode()) for s in batch['texts']]
    record = accumulate_global_word(view,batch,global_positions=positions,local_positions=positions,
        global_response_EOS_targets=targets,global_word_UTF8_bytes=byte_lengths,CTC_weight=.4,native_CE_weight=.7)
    expected = expected_backward(reference,batch)
    assert record['local_byte_normalized_CTC_sum'] == expected[0] and record['local_native_CE_sum'] == expected[1]
    assert record['local_UTF8_bytes'] == expected[2] == 70 and record['actual_CTC_frames'] == 140
    for (_,a),(_,b) in zip(named_word_selected(view),named_word_selected(reference)):assert torch.equal(a.grad,b.grad)


@pytest.mark.parametrize('bf16',[False,True])
def test_actual_word_step_next_Adam_update_recovers_exactly_and_keeps_FP32_word_readout(bf16):
    view,batch = fixture();opt = optimizer(view);amp = (lambda:torch.autocast('cpu',dtype=torch.bfloat16)) if bf16 else None
    positions = [17,23,29];targets = [int((row[1:] != -100).sum()) for row in batch['labels']];byte_lengths = [len(s.encode()) for s in batch['texts']]
    def step(current,adam,payload):
        record = accumulate_global_word(current,payload,global_positions=positions,local_positions=positions,
            global_response_EOS_targets=targets,global_word_UTF8_bytes=byte_lengths,CTC_weight=.4,native_CE_weight=.7,
            native_logits_dtype=torch.bfloat16 if bf16 else torch.float32,amp=amp)
        assert record['actual_word_readout_dtype'] == 'torch.float32' and record['actual_CTC_frames'] == 140
        torch.nn.utils.clip_grad_norm_([p for _,p in named_word_selected(current)],1.,error_if_nonfinite=True)
        adam.step();adam.zero_grad(set_to_none=True);return record
    step(view,opt,batch)
    stream = io.BytesIO();torch.save(dict(model=view.actor.state_dict(),optimizer=opt.state_dict()),stream);stream.seek(0)
    saved = torch.load(stream,map_location='cpu',weights_only=True);fresh,_ = fixture();fresh.actor.load_state_dict(saved['model'],strict=True)
    resumed = optimizer(fresh);resumed.load_state_dict(saved['optimizer'])
    changed = {**batch,'word_pixels':batch['word_pixels']*.8,'pixel_values':batch['pixel_values']*.9}
    assert step(view,opt,changed) == step(fresh,resumed,changed)
    for name,value in view.actor.state_dict().items():assert torch.equal(value,fresh.actor.state_dict()[name])
    for left,right in zip(opt.state.values(),resumed.state.values()):
        for name,value in left.items():assert torch.equal(value,right[name])


def distributed_worker(rank,folder):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and os.nice(0) >= 19
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    dist.init_process_group('gloo',init_method='file://'+str(Path(folder)/'rendezvous'),rank=rank,world_size=4,timeout=timedelta(minutes=4))
    try:
        view,batch = fixture();reference = copy.deepcopy(view);opt = optimizer(view);refopt = optimizer(reference)
        selected = [p for _,p in named_word_selected(view)];gold = [p for _,p in named_word_selected(reference)];chosen = {id(p) for p in selected}
        fixed = {n:p.detach().clone() for n,p in view.actor.named_parameters() if id(p) not in chosen}
        with torch.no_grad():text_before = view.actor(batch['input_ids'],attention_mask=batch['attention_mask'])['logits'].clone()
        observations = []
        for slots,positions in [([0,1,2],[17,23,29]),([2],[29])]:
            whole = subset(batch,slots);local_slots = slots[rank::4];local = subset(batch,local_slots) if local_slots else None
            targets = [int((row[1:] != -100).sum()) for row in whole['labels']];byte_lengths = [len(s.encode()) for s in whole['texts']]
            observed = [];hook = view.actor.base.heads['word'].register_forward_hook(lambda *args:observed.append(1))
            record = accumulate_global_word(view,local,global_positions=positions,local_positions=positions[rank::4],
                global_response_EOS_targets=targets,global_word_UTF8_bytes=byte_lengths,CTC_weight=.4,native_CE_weight=.7)
            hook.remove();assert len(observed) == bool(local_slots)
            if not local_slots:assert all(p.grad is None for p in selected)
            for p in selected:
                if p.grad is None:p.grad = torch.zeros_like(p)
                dist.all_reduce(p.grad,op=dist.ReduceOp.SUM)
            expected = expected_backward(reference,whole)
            maximum_gradient = max(float((a.grad-b.grad).abs().max()) for a,b in zip(selected,gold))
            assert maximum_gradient < 5e-6
            torch.nn.utils.clip_grad_norm_(selected,1.,error_if_nonfinite=True);torch.nn.utils.clip_grad_norm_(gold,1.,error_if_nonfinite=True)
            opt.step();refopt.step();opt.zero_grad(set_to_none=True);refopt.zero_grad(set_to_none=True)
            maximum_weight = max(float((a-b).detach().abs().max()) for a,b in zip(selected,gold));assert maximum_weight < 5e-6
            identities = [None]*4;dist.all_gather_object(identities,digest(torch.cat([p.detach().reshape(-1) for p in selected])));assert len(set(identities)) == 1
            assert all(torch.equal(p,fixed[n]) and p.grad is None for n,p in view.actor.named_parameters() if n in fixed)
            with torch.no_grad():assert torch.equal(text_before,view.actor(batch['input_ids'],attention_mask=batch['attention_mask'])['logits'])
            records = [None]*4;dist.all_gather_object(records,record)
            assert sum(r['local_UTF8_bytes'] for r in records) == expected[2] == sum(byte_lengths)
            assert abs(sum(r['local_byte_normalized_CTC_sum'] for r in records)-expected[0]) < 1e-4
            observations.append(dict(rank=rank,global_words=len(positions),local_words=len(local_slots),
                actual_own_unique_parameters=view.actor.counts()['actual_unique'],actual_selected_values=sum(p.numel() for p in selected),
                actual_maximum_word_UTF8_bytes=max(byte_lengths),actual_maximum_required_CTC_steps=max(len(s.encode())+sum(a == b for a,b in zip(s.encode(),s.encode()[1:])) for s in whole['texts']),
                actual_CTC_frames_on_nonempty_ranks=140,serial_max_gradient_absolute_difference=maximum_gradient,
                serial_max_Adam_weight_absolute_difference=maximum_weight,all_four_replicas_bitexact=True,
                all_fixed_text_actor_critic_pointing_other_heads_unchanged=True,record=record))
        # Removing one original UTF8 byte on the owner must reject every rank
        # before any forward, gradient or optimizer mutation.
        payload = subset(batch,[2]) if rank == 0 else None
        if rank == 0:payload['texts'] = ['a'*63]
        calls = [];hook = view.actor.base.heads['word'].register_forward_hook(lambda *args:calls.append(1))
        before = [p.detach().clone() for p in selected];old_state = copy.deepcopy(opt.state_dict())
        with pytest.raises(ValueError,match='Coordinated word source/native preflight'):
            accumulate_global_word(view,payload,global_positions=[29],local_positions=[29] if rank == 0 else [],
                global_response_EOS_targets=[65],global_word_UTF8_bytes=[64])
        hook.remove();assert not calls and all(torch.equal(a,b) for a,b in zip(before,selected)) and all(p.grad is None for p in selected)
        for key,value in opt.state_dict()['state'].items():
            for name,tensor in value.items():assert torch.equal(tensor,old_state['state'][key][name])
        assert not torch.cuda.is_initialized()
        (Path(folder)/(str(rank)+'.json')).write_text(json.dumps(dict(complete=True,observations=observations,
            truncated_UTF8_on_one_rank_rejected_before_forward_or_Adam=True,production_updates=0),indent=2)+'\n')
    finally:dist.destroy_process_group()


def test_actual_four_word_replicas_complete_CTC_CE_and_singleton_tail(tmp_path):
    spawn(distributed_worker,args=(str(tmp_path),),nprocs=4,join=True)
    rows = [json.loads((tmp_path/(str(rank)+'.json')).read_text()) for rank in range(4)]
    assert all(row['complete'] and len(row['observations']) == 2 for row in rows)
    assert all(row['observations'][1]['local_words'] == 0 for row in rows[1:])
    assert all(r['actual_maximum_word_UTF8_bytes'] == 64 and r['actual_maximum_required_CTC_steps'] == 127 for row in rows for r in row['observations'])
