"""Four actual small own text/image models, complete caption CE plus matching."""
import copy
from datetime import timedelta
import json
import os
from pathlib import Path

import pytest
import torch
from torch import distributed as dist
from torch.multiprocessing import spawn

from test_own_actor_vision_prefix_reuse_v1 import actor,data
from slm_lab.global_photo_prefix_stage_kernel_v1 import accumulate_global_photo
from slm_lab.global_visual_matching_gradients_v1 import digest
from slm_lab.own_actor_vision_alignment_v1 import pair_positives
from slm_lab.own_actor_vision_prefix_split_backward_v1 import OwnActorVisionPrefixSplitBackward


def worker(rank,folder):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and os.nice(0) >= 19
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    dist.init_process_group('gloo',init_method='file://'+str(Path(folder)/'rendezvous'),rank=rank,world_size=4,timeout=timedelta(minutes=3))
    proofs = []
    try:
        model = actor();reference = copy.deepcopy(model)
        view = OwnActorVisionPrefixSplitBackward(model).train();gold = OwnActorVisionPrefixSplitBackward(reference).train()
        for current in [view,gold]:current.native.gradient_checkpointing = True;current.native.vision.train()
        selected = [p for _,p in view.named_selected()];gold_selected = [p for _,p in gold.named_selected()]
        optimizer = torch.optim.AdamW(selected,lr=1e-3,weight_decay=.01,foreach=False)
        refopt = torch.optim.AdamW(gold_selected,lr=1e-3,weight_decay=.01,foreach=False)
        batch,ids,mask,_ = data()
        with torch.no_grad():text_before = model(ids,attention_mask=mask)['logits'].clone()
        original = {n:p.detach().clone() for n,p in model.named_parameters()}
        selected_ids = {id(p) for p in selected}
        for count in [3,2]:
            whole = {k:v[:count] for k,v in batch.items()};source_ids = ids[:count];source_mask = mask[:count]
            positions = [19+5*p for p in range(count)];local = list(range(count))[rank::4]
            targets = [int((row[1:] != -100).sum()) for row in whole['labels']]
            keys = ['image-'+str(p) for p in positions];captions = ['caption-'+str(p) for p in positions]
            current = {k:v[local] for k,v in whole.items()} if local else None
            local_ids = source_ids[local] if local else None;local_mask = source_mask[local] if local else None
            observed = [];hook = view.native.vision.register_forward_hook(lambda *args:observed.append(1))
            record = accumulate_global_photo(view,current,local_ids,local_mask,global_positions=positions,
                local_positions=positions[rank::4],global_response_EOS_targets=targets,pixel_keys=keys,captions=captions,
                matching_weight=.1,temperature=.7)
            hook.remove();assert len(observed) == (1 if local else 0)
            if not local:assert all(p.grad is None for p in selected)
            for p in selected:
                if p.grad is None:p.grad = torch.zeros_like(p)
                dist.all_reduce(p.grad,op=dist.ReduceOp.SUM)
            objective = gold.objective(whole,source_ids,source_mask,pair_positives(keys,captions),matching_weight=.1,temperature=.7)
            gold.backward(objective)
            gradient_error = max(float((a.grad-b.grad).abs().max()) for a,b in zip(selected,gold_selected))
            assert gradient_error < 5e-6
            ce_sums = [None]*4;dist.all_gather_object(ce_sums,record['local_caption_CE_sum'])
            assert abs(sum(ce_sums)/sum(targets)-float(objective['caption_CE'].detach())) < 5e-6
            assert abs(record['global_matching']['unweighted_loss']-float(objective['matching_loss'].detach())) < 5e-6
            torch.nn.utils.clip_grad_norm_(selected,1.,error_if_nonfinite=True)
            torch.nn.utils.clip_grad_norm_(gold_selected,1.,error_if_nonfinite=True)
            optimizer.step();refopt.step();optimizer.zero_grad(set_to_none=True);refopt.zero_grad(set_to_none=True)
            error = max(float((a-b).detach().abs().max()) for a,b in zip(selected,gold_selected))
            assert error < 5e-6
            for name,p in model.named_parameters():
                if id(p) not in selected_ids:assert torch.equal(original[name],p) and p.grad is None
            with torch.no_grad():assert torch.equal(text_before,model(ids,attention_mask=mask)['logits'])
            identities = [None]*4;dist.all_gather_object(identities,digest(torch.cat([p.detach().reshape(-1) for p in selected])))
            assert len(set(identities)) == 1
            proofs.append(dict(rank=rank,global_sources=count,local_sources=len(local),actual_own_unique_parameters=model.counts()['actual_unique'],
                actual_selected_values=sum(p.numel() for p in selected),actual_image_encoder_calls=len(observed),
                complete_original_response_EOS_targets=sum(targets),serial_max_gradient_absolute_difference=gradient_error,
                serial_max_Adam_weight_absolute_difference=error,replicas_bitexact=True,text_logits_and_fixed_weights_bitexact=True,
                record=record,full_size_GPU_and_quality_not_proved=True))
        # An incomplete native target on one actual rank must reject all ranks
        # before any encoder forward or optimizer state mutation.
        optimizer.zero_grad(set_to_none=True)
        current = {k:v[rank:rank+1].clone() for k,v in batch.items()} if rank < 3 else None
        if rank == 1:current['labels'][current['labels'] == 2] = -100
        before = [p.detach().clone() for p in selected];state_before = copy.deepcopy(optimizer.state_dict())
        observed = [];hook = view.native.vision.register_forward_hook(lambda *args:observed.append(1))
        with pytest.raises(ValueError,match='Coordinated photo source/native preflight'):
            accumulate_global_photo(view,current,ids[rank:rank+1] if rank < 3 else None,mask[rank:rank+1] if rank < 3 else None,
                global_positions=[19,24,29],local_positions=[19,24,29][rank::4],
                global_response_EOS_targets=[int((row[1:] != -100).sum()) for row in batch['labels']],
                pixel_keys=['a','b','c'],captions=['caption a','caption b','caption c'])
        hook.remove();assert not observed and all(torch.equal(a,b) for a,b in zip(before,selected))
        assert all(p.grad is None for p in selected)
        for key,state in optimizer.state_dict()['state'].items():
            for name,tensor in state.items():assert torch.equal(tensor,state_before['state'][key][name])
        assert not torch.cuda.is_initialized()
        (Path(folder)/(str(rank)+'.json')).write_text(json.dumps(dict(complete=True,proofs=proofs,
            missing_EOS_on_one_rank_rejected_everywhere_before_forward_and_Adam=True,
            CUDA_hidden=True,production_optimizer_updates=0),indent=2)+'\n')
    finally:dist.destroy_process_group()


def test_actual_four_own_replicas_caption_matching_gradients_and_empty_tail(tmp_path):
    spawn(worker,args=(str(tmp_path),),nprocs=4,join=True)
    rows = [json.loads((tmp_path/(str(rank)+'.json')).read_text()) for rank in range(4)]
    assert all(r['complete'] and len(r['proofs']) == 2 for r in rows)
    assert all(p['actual_image_encoder_calls'] == 0 for p in rows[3]['proofs'])
    assert rows[2]['proofs'][1]['actual_image_encoder_calls'] == 0
