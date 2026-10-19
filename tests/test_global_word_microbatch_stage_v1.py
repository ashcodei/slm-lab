"""Small actual own-model whole-word microbatch mechanics, not OCR scores."""
from contextlib import nullcontext
import copy
from datetime import timedelta
import hashlib
import io
import json
import os
from pathlib import Path

import pytest
import torch
from torch import distributed as dist
from torch.multiprocessing import spawn

from test_global_word_stage_kernel_v1 import fixture as original_fixture, subset
from slm_lab.deterministic_positions_v1 import install
from slm_lab.global_visual_matching_gradients_v1 import digest
from slm_lab.global_word_microbatch_stage_v1 import accumulate_global_word_microbatched
from slm_lab.global_word_stage_kernel_v1 import named_word_selected
from slm_lab.shared_word_stage_v1 import ctc_word_sum


def fixture():
    view,batch=original_fixture()
    install(view.actor.base)
    return view,subset(batch,[0,1,2]*3)


def selected(view):return [p for _,p in named_word_selected(view)]


def optimizer(view):
    return torch.optim.AdamW(selected(view),lr=1e-4,eps=1e-5,weight_decay=.01,foreach=False)


def spec(batch,positions):
    return dict(global_positions=positions,global_context_lengths=batch['attention_mask'].sum(1).tolist(),
        global_response_EOS_targets=(batch['labels'][:,1:]!=-100).sum(1).tolist(),
        global_word_UTF8_bytes=[len(text.encode('utf8')) for text in batch['texts']],
        global_word_UTF8_sha256=[hashlib.sha256(text.encode('utf8')).hexdigest() for text in batch['texts']])


def independent_chunks(lengths,limit=450):
    remaining=list(range(len(lengths)));result=[]
    while remaining:
        n=max(n for n in range(1,min(4,len(remaining))+1) if n*(196+max(lengths[i] for i in remaining[:n]))<=limit)
        result.append(remaining[:n]);remaining=remaining[n:]
    return result


def independent_backward(view,batch,chunks,global_words,global_targets,amp=None):
    amp=amp or nullcontext;ctc_total=ce_total=0.;byte_count=0
    with torch.autocast('cpu',enabled=False):
        for slots in chunks:
            width=int(batch['attention_mask'][slots].sum(1).max())
            native={k:batch[k][slots,:width] for k in ['input_ids','attention_mask','labels']}
            native['pixel_values']=batch['pixel_values'][slots]
            with amp():logits=view.actor.base.word_logits(batch['word_pixels'][slots])
            ctc,actual_bytes=ctc_word_sum(logits,[batch['texts'][i] for i in slots])
            (ctc*.4/global_words).backward(inputs=selected(view));ctc_total+=float(ctc.detach());byte_count+=actual_bytes
            with amp():output=view.native(**native)
            targets=int((native['labels'][:,1:]!=-100).sum())
            (output['loss']*.7*targets/global_targets).backward(inputs=selected(view));ce_total+=float(output['loss'].detach())*targets
    return ctc_total,ce_total,byte_count


@pytest.mark.parametrize('bf16',[False,True])
@pytest.mark.parametrize('ambient',[False,True])
def test_complete_microbatches_match_independent_CTC_and_existing_native_mean_CE(bf16,ambient):
    view,batch=fixture();reference=copy.deepcopy(view);positions=[17+3*i for i in range(9)];settings=spec(batch,positions)
    amp=(lambda:torch.autocast('cpu',dtype=torch.bfloat16)) if bf16 else None
    outer=(lambda:torch.autocast('cpu',dtype=torch.bfloat16)) if bf16 and ambient else nullcontext
    with outer():
        record=accumulate_global_word_microbatched(view,batch,local_positions=positions,maximum_padded_tokens=450,
            CTC_weight=.4,native_CE_weight=.7,native_logits_dtype=torch.bfloat16 if bf16 else torch.float32,amp=amp,**settings)
    chunks=independent_chunks(settings['global_context_lengths'])
    with outer():expected=independent_backward(reference,batch,chunks,9,sum(settings['global_response_EOS_targets']),amp)
    assert abs(record['local_byte_normalized_CTC_sum']-expected[0])<1e-6
    assert abs(record['local_native_CE_sum']-expected[1])<5e-5
    assert record['local_UTF8_bytes']==expected[2]==210
    assert record['actual_word_forward_calls']==record['actual_native_forward_calls']==len(chunks)
    assert all(row['actual_CTC_frames']==140 and row['actual_word_logits_dtype']=='torch.float32' for row in record['actual_microbatches'])
    assert max(float((a.grad-b.grad).abs().max()) for a,b in zip(selected(view),selected(reference)))<5e-6
    chosen={id(p) for p in selected(view)}
    assert all(p.grad is None for p in view.parameters() if id(p) not in chosen)
    left,right=optimizer(view),optimizer(reference)
    for current in [view,reference]:torch.nn.utils.clip_grad_norm_(selected(current),1.,error_if_nonfinite=True)
    left.step();right.step()
    assert max(float((a-b).detach().abs().max()) for a,b in zip(selected(view),selected(reference)))<5e-6


@pytest.mark.parametrize('bf16',[False,True])
def test_whole_word_microbatch_next_Adam_update_recovers_exactly(bf16):
    view,batch=fixture();adam=optimizer(view);positions=list(range(9));settings=spec(batch,positions)
    amp=(lambda:torch.autocast('cpu',dtype=torch.bfloat16)) if bf16 else None
    def update(current,opt,payload):
        record=accumulate_global_word_microbatched(current,payload,local_positions=positions,maximum_padded_tokens=450,
            CTC_weight=.4,native_CE_weight=.7,native_logits_dtype=torch.bfloat16 if bf16 else torch.float32,amp=amp,**settings)
        torch.nn.utils.clip_grad_norm_(selected(current),1.,error_if_nonfinite=True)
        opt.step();opt.zero_grad(set_to_none=True);return record
    fixed_ids={id(p) for p in selected(view)}
    fixed={name:p.detach().clone() for name,p in view.actor.named_parameters() if id(p) not in fixed_ids}
    update(view,adam,batch)
    stream=io.BytesIO();torch.save(dict(model=view.actor.state_dict(),optimizer=adam.state_dict()),stream);stream.seek(0)
    saved=torch.load(stream,map_location='cpu',weights_only=True)
    resumed,_=fixture();resumed.actor.load_state_dict(saved['model'],strict=True);resumed_adam=optimizer(resumed)
    resumed_adam.load_state_dict(saved['optimizer'])
    changed={**batch,'pixel_values':batch['pixel_values']*.8,'word_pixels':batch['word_pixels']*.9}
    assert update(view,adam,changed)==update(resumed,resumed_adam,changed)
    for name,value in view.actor.state_dict().items():
        assert torch.equal(value,resumed.actor.state_dict()[name])
        if name in fixed:assert torch.equal(value,fixed[name])
    for a,b in zip(adam.state.values(),resumed_adam.state.values()):
        for name,value in a.items():assert torch.equal(value,b[name])


def test_ambient_BF16_does_not_change_whole_word_gradients():
    gradients=[]
    for ambient in [False,True]:
        view,batch=fixture();settings=spec(batch,list(range(9)))
        with torch.autocast('cpu',dtype=torch.bfloat16,enabled=ambient):
            accumulate_global_word_microbatched(view,batch,local_positions=list(range(9)),maximum_padded_tokens=450,
                CTC_weight=.4,native_CE_weight=.7,native_logits_dtype=torch.bfloat16,
                amp=lambda:torch.autocast('cpu',dtype=torch.bfloat16),**settings)
        gradients.append([p.grad.clone() for p in selected(view)])
    assert all(torch.equal(a,b) for a,b in zip(*gradients))


def worker(rank,folder):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and os.nice(0)>=19
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    dist.init_process_group('gloo',init_method='file://'+str(Path(folder)/'rendezvous'),rank=rank,world_size=4,timeout=timedelta(minutes=4))
    try:
        view,batch=fixture();reference=copy.deepcopy(view);adam,refadam=optimizer(view),optimizer(reference)
        chosen={id(p) for p in selected(view)}
        fixed={name:p.detach().clone() for name,p in view.actor.named_parameters() if id(p) not in chosen}
        with torch.no_grad():original_text=view.actor(batch['input_ids'],attention_mask=batch['attention_mask'])['logits'].clone()
        proofs=[]
        for slots in [list(range(9)),[2]]:
            whole=subset(batch,slots);positions=[17+3*i for i in slots];settings=spec(whole,positions)
            local=list(range(len(slots)))[rank::4];payload=subset(whole,local) if local else None
            calls=[];hook=view.actor.base.heads['word'].register_forward_hook(lambda *args:calls.append(1))
            record=accumulate_global_word_microbatched(view,payload,local_positions=positions[rank::4],maximum_padded_tokens=450,
                CTC_weight=.4,native_CE_weight=.7,**settings)
            hook.remove();assert len(calls)==record['actual_word_forward_calls']
            if not local:assert not calls and all(p.grad is None for p in selected(view))
            for p in selected(view):
                if p.grad is None:p.grad=torch.zeros_like(p)
                dist.all_reduce(p.grad,op=dist.ReduceOp.SUM)
            reference_ctc=reference_ce=0.;reference_bytes=0
            for owner in range(4):
                owned=list(range(len(slots)))[owner::4]
                if not owned:continue
                source=subset(whole,owned);chunks=independent_chunks(source['attention_mask'].sum(1).tolist())
                ctc,ce,actual_bytes=independent_backward(reference,source,chunks,len(slots),sum(settings['global_response_EOS_targets']))
                reference_ctc+=ctc;reference_ce+=ce;reference_bytes+=actual_bytes
            error=max(float((a.grad-b.grad).abs().max()) for a,b in zip(selected(view),selected(reference)))
            assert error<5e-6
            records=[None]*4;dist.all_gather_object(records,record)
            assert sum(v['local_UTF8_bytes'] for v in records)==reference_bytes==sum(settings['global_word_UTF8_bytes'])
            assert abs(sum(v['local_byte_normalized_CTC_sum'] for v in records)-reference_ctc)<1e-6
            assert abs(sum(v['local_native_CE_sum'] for v in records)-reference_ce)<5e-5
            for current in [view,reference]:torch.nn.utils.clip_grad_norm_(selected(current),1.,error_if_nonfinite=True)
            adam.step();refadam.step();adam.zero_grad(set_to_none=True);refadam.zero_grad(set_to_none=True)
            weight_error=max(float((a-b).detach().abs().max()) for a,b in zip(selected(view),selected(reference)))
            assert weight_error<5e-6
            for name,p in view.actor.named_parameters():
                if name in fixed:assert torch.equal(p,fixed[name]) and p.grad is None
            with torch.no_grad():assert torch.equal(original_text,view.actor(batch['input_ids'],attention_mask=batch['attention_mask'])['logits'])
            hashes=[None]*4;dist.all_gather_object(hashes,digest(torch.cat([p.detach().reshape(-1) for p in selected(view)])))
            assert len(set(hashes))==1
            proofs.append(dict(rank=rank,global_words=len(slots),local_words=len(local),actual_own_unique_parameters=view.actor.counts()['actual_unique'],
                actual_selected_values=sum(p.numel() for p in selected(view)),maximum_original_UTF8_bytes=max(settings['global_word_UTF8_bytes']),
                longest_repeated_byte_minimum_CTC_steps=127,actual_nonempty_CTC_frames=140,
                serial_matching_microbatch_max_gradient_difference=error,serial_matching_microbatch_max_Adam_weight_difference=weight_error,
                every_replica_bitexact=True,all_text_actor_critic_pointing_other_heads_fixed=True,record=record,
                old_fullbatch_shape_raw_gradient_gate_still_failed_unwaived=True))
        rejected=[]
        for fault in ['one_byte_missing','same_byte_length_changed_transcript','missing_native_EOS','global_hash_disagreement']:
            payload=subset(batch,[1]) if rank==0 else None;settings=spec(subset(batch,[1]),[20])
            if rank==0:
                if fault=='one_byte_missing':payload['texts']=['A']
                elif fault=='same_byte_length_changed_transcript':payload['texts']=['Bé']
                elif fault=='missing_native_EOS':payload['labels'][payload['labels']==2]=-100
            if rank==1 and fault=='global_hash_disagreement':settings['global_word_UTF8_sha256'][0]='f'*64
            calls=[];hook=view.actor.base.heads['word'].register_forward_hook(lambda *args:calls.append(1))
            before=[p.detach().clone() for p in selected(view)];state=copy.deepcopy(adam.state_dict())
            with pytest.raises(ValueError,match='Coordinated microbatched word'):
                accumulate_global_word_microbatched(view,payload,local_positions=[20] if rank==0 else [],
                    maximum_padded_tokens=450,CTC_weight=.4,native_CE_weight=.7,**settings)
            hook.remove();assert not calls and all(p.grad is None for p in selected(view))
            assert all(torch.equal(a,b) for a,b in zip(before,selected(view)))
            for key,value in adam.state_dict()['state'].items():
                for name,tensor in value.items():assert torch.equal(tensor,state['state'][key][name])
            rejected.append(fault)
        assert not torch.cuda.is_initialized()
        (Path(folder)/(str(rank)+'.json')).write_text(json.dumps(dict(complete=True,proofs=proofs,
            coordinated_invalid_sources_rejected_before_forward_and_Adam=rejected,CUDA_hidden=True,
            production_optimizer_updates=0,full_size_GPU_capacity_and_quality_pending=True),indent=2)+'\n')
    finally:dist.destroy_process_group()


def test_four_own_word_replicas_complete_microbatches_empty_tail_and_source_hash_rejections(tmp_path):
    spawn(worker,args=(str(tmp_path),),nprocs=4,join=True)
    rows=[json.loads((tmp_path/(str(rank)+'.json')).read_text()) for rank in range(4)]
    assert all(row['complete'] and len(row['proofs'])==2 for row in rows)
    assert all(row['proofs'][1]['record']['actual_word_forward_calls']==0 for row in rows[1:])
