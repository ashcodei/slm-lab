"""Actual CPU feature derivatives and four-process matching/Adam mechanics."""
import copy
from datetime import timedelta
import json
import os
from pathlib import Path

import pytest
import torch
from torch import distributed as dist
from torch.multiprocessing import spawn

from slm_lab.global_visual_matching_gradients_v1 import global_matching_gradients,backward_global_matching,digest
from slm_lab.own_actor_vision_alignment_v1 import pair_positives,symmetric_pair_loss


def fixture(count):
    generator = torch.Generator().manual_seed(991117+count)
    images = torch.randn(count,5,generator=generator)
    texts = torch.randn(count,5,generator=generator)
    positions = [3+7*p for p in range(count)]
    keys = ['image-'+str(p) for p in range(count)];captions = ['caption-'+str(p) for p in range(count)]
    if count >= 7:
        keys[1] = keys[0];captions[2] = captions[1];captions[4] = captions[1]
    return images,texts,positions,keys,captions


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_serial_global_derivatives_match_canonical_full_pair_loss_and_weight(dtype):
    torch.set_num_threads(1)
    raw_images,raw_texts,positions,keys,captions = fixture(7)
    images = torch.nn.Parameter(raw_images.to(dtype));texts = torch.nn.Parameter(raw_texts.to(dtype))
    result = global_matching_gradients(images,texts,global_positions=positions,local_positions=positions,
        pixel_keys=keys,captions=captions,matching_weight=.3,temperature=.7)
    a = images.detach().clone().requires_grad_();b = texts.detach().clone().requires_grad_()
    loss = .3*symmetric_pair_loss(a,b,pair_positives(keys,captions),temperature=.7)
    loss.backward()
    assert result.weighted_loss == float(loss.detach())
    assert result.actual_global_positive_cells == int(pair_positives(keys,captions).sum())
    assert images.grad is None and texts.grad is None
    backward_global_matching(images,texts,result,parameters=[images,texts])
    assert torch.equal(images.grad,a.grad) and torch.equal(texts.grad,b.grad)


@pytest.mark.parametrize('fault',['wrong_positions','duplicate_positions','no_negatives','nan_weight','oversize_features'])
def test_invalid_global_sources_and_settings_rejected_without_creating_leaf_gradients(fault):
    torch.set_num_threads(1)
    raw_images,raw_texts,positions,keys,captions = fixture(7)
    images = torch.nn.Parameter(raw_images);texts = torch.nn.Parameter(raw_texts)
    settings = dict(global_positions=positions,local_positions=positions,pixel_keys=keys,captions=captions)
    if fault == 'wrong_positions': settings['local_positions'] = positions[::-1]
    elif fault == 'duplicate_positions': settings['global_positions'] = [positions[0]]*7
    elif fault == 'no_negatives': settings['captions'] = ['same']*7
    elif fault == 'nan_weight': settings['matching_weight'] = float('nan')
    else: images = torch.nn.Parameter(torch.ones(7,1025));texts = torch.nn.Parameter(torch.ones(7,1025))
    with pytest.raises(ValueError,match='Coordinated'):global_matching_gradients(images,texts,**settings)
    assert images.grad is None and texts.grad is None


@pytest.mark.parametrize('fault',['new_features','mutated_features','mutated_gradients'])
def test_backward_plan_cannot_attach_to_changed_features_or_changed_derivatives(fault):
    torch.set_num_threads(1)
    a,b,positions,keys,captions = fixture(7)
    images = torch.nn.Parameter(a);texts = torch.nn.Parameter(b)
    plan = global_matching_gradients(images,texts,global_positions=positions,local_positions=positions,pixel_keys=keys,captions=captions)
    if fault == 'new_features':images = torch.nn.Parameter(a.clone())
    elif fault == 'mutated_features':
        with torch.no_grad():images.add_(.1)
    else:plan.image_gradients.add_(.1)
    with pytest.raises(ValueError,match='untouched'):backward_global_matching(images,texts,plan,parameters=[images,texts])
    assert images.grad is None and texts.grad is None


def distributed_worker(rank,folder):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19:
        raise ValueError('Only newly spawned low-priority CUDA-hidden mechanical workers allowed')
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    dist.init_process_group('gloo',init_method='file://'+str(Path(folder)/'rendezvous'),rank=rank,world_size=4,timeout=timedelta(minutes=3))
    observations = []
    try:
        for count in [7,2,64]:
            a,b,positions,keys,captions = fixture(count)
            for dtype in [torch.float32,torch.bfloat16]:
                images = torch.nn.Parameter(a[rank::4].to(dtype));texts = torch.nn.Parameter(b[rank::4].to(dtype))
                plan = global_matching_gradients(images,texts,global_positions=positions,local_positions=positions[rank::4],
                    pixel_keys=keys,captions=captions,matching_weight=.3,temperature=.7)
                gold_images = a.to(dtype).detach().clone().requires_grad_();gold_texts = b.to(dtype).detach().clone().requires_grad_()
                gold = .3*symmetric_pair_loss(gold_images,gold_texts,pair_positives(keys,captions),temperature=.7)
                gold.backward();assert plan.weighted_loss == float(gold.detach())
                assert torch.equal(plan.image_gradients.to(dtype),gold_images.grad[rank::4])
                assert torch.equal(plan.text_gradients.to(dtype),gold_texts.grad[rank::4])
                record = backward_global_matching(images,texts,plan,parameters=[images,texts])
                if positions[rank::4]:
                    assert torch.equal(images.grad,gold_images.grad[rank::4]) and torch.equal(texts.grad,gold_texts.grad[rank::4])
                else:assert images.grad is None and texts.grad is None and record['local_backward_calls'] == 0
                identities = [None]*4;dist.all_gather_object(identities,plan.global_features_sha256)
                assert len(set(identities)) == 1
                observations.append(dict(kind='exact_complete_feature_derivatives',source_count=count,dtype=str(dtype),
                    rank=rank,empty_rank=not bool(positions[rank::4]),record=plan.record(),actual_backward=record))
            # Real tiny learned projection replicas verify global normalization,
            # one parameter-gradient SUM and the actual resulting Adam update.
            torch.manual_seed(991217)
            model = torch.nn.ModuleDict(dict(image=torch.nn.Linear(6,5,bias=False),text=torch.nn.Linear(6,5,bias=False),
                fixed=torch.nn.Linear(6,3,bias=False)))
            reference = copy.deepcopy(model);selected = [model['image'].weight,model['text'].weight]
            fixed = model['fixed'].weight.detach().clone();optimizer = torch.optim.AdamW(selected,lr=1e-3,weight_decay=.01,foreach=False)
            refopt = torch.optim.AdamW([reference['image'].weight,reference['text'].weight],lr=1e-3,weight_decay=.01,foreach=False)
            x = torch.randn(count,6,generator=torch.Generator().manual_seed(991317+count))
            y = torch.randn(count,6,generator=torch.Generator().manual_seed(991417+count))
            if positions[rank::4]:
                images = model['image'](x[rank::4]);texts = model['text'](y[rank::4]);forwards = 2
            else:images = torch.empty(0,5);texts = torch.empty(0,5);forwards = 0
            plan = global_matching_gradients(images,texts,global_positions=positions,local_positions=positions[rank::4],
                pixel_keys=keys,captions=captions,matching_weight=.3,temperature=.7)
            backward_global_matching(images,texts,plan,parameters=selected)
            for p in selected:
                if p.grad is None:p.grad = torch.zeros_like(p)
                dist.all_reduce(p.grad,op=dist.ReduceOp.SUM)
            expected = .3*symmetric_pair_loss(reference['image'](x),reference['text'](y),pair_positives(keys,captions),temperature=.7)
            expected.backward()
            gradient_error = max(float((p.grad-q.grad).abs().max()) for p,q in zip(selected,[reference['image'].weight,reference['text'].weight]))
            assert gradient_error < 1e-6
            torch.nn.utils.clip_grad_norm_(selected,1.,error_if_nonfinite=True)
            torch.nn.utils.clip_grad_norm_([reference['image'].weight,reference['text'].weight],1.,error_if_nonfinite=True)
            optimizer.step();refopt.step()
            weight_error = max(float((p-q).detach().abs().max()) for p,q in zip(selected,[reference['image'].weight,reference['text'].weight]))
            assert weight_error < 1e-6
            assert torch.equal(fixed,model['fixed'].weight) and model['fixed'].weight.grad is None
            identities = [None]*4;dist.all_gather_object(identities,digest(torch.cat([p.detach().reshape(-1) for p in selected])))
            assert len(set(identities)) == 1
            observations.append(dict(kind='actual_projection_gradient_SUM_and_Adam',source_count=count,rank=rank,
                actual_total_parameters=sum(p.numel() for p in model.parameters()),selected_parameter_values=60,
                actual_local_neural_projection_calls=forwards,serial_max_gradient_absolute_difference=gradient_error,
                serial_max_Adam_parameter_absolute_difference=weight_error,replicas_bitexact=True,fixed_head_unchanged=True))
        # One rank's bad source/order/precision/value must be rejected by every
        # rank before any tensor collectives or optimizer commit can diverge.
        for fault in ['source_order','caption_identity','feature_precision','nonfinite','no_negatives','zero_norm']:
            a,b,positions,keys,captions = fixture(7)
            images = torch.nn.Parameter(a[rank::4]);texts = torch.nn.Parameter(b[rank::4]);local = positions[rank::4]
            if fault == 'source_order' and rank == 1:local = local[::-1]
            if fault == 'caption_identity' and rank == 1:captions[0] = 'different full source caption'
            if fault == 'feature_precision' and rank == 1:images = torch.nn.Parameter(images.bfloat16());texts = torch.nn.Parameter(texts.bfloat16())
            if fault == 'nonfinite' and rank == 1:
                with torch.no_grad():images[0,0] = float('nan')
            if fault == 'zero_norm' and rank == 1:
                with torch.no_grad():images[0].zero_()
            if fault == 'no_negatives':captions = ['same']*7
            with pytest.raises(ValueError):
                global_matching_gradients(images,texts,global_positions=positions,local_positions=local,pixel_keys=keys,captions=captions)
            assert images.grad is None and texts.grad is None
            dist.barrier();observations.append(dict(kind='all_ranks_rejected',fault=fault,rank=rank))
        assert not torch.cuda.is_initialized()
        (Path(folder)/(str(rank)+'.json')).write_text(json.dumps(observations,indent=2)+'\n')
    finally:dist.destroy_process_group()


def test_actual_four_process_global_pairs_gradients_empty_ranks_and_Adam(tmp_path):
    spawn(distributed_worker,args=(str(tmp_path),),nprocs=4,join=True)
    observations = [json.loads((tmp_path/(str(rank)+'.json')).read_text()) for rank in range(4)]
    assert all(len(rows) == 15 for rows in observations)
    assert all(sum(r['kind'] == 'exact_complete_feature_derivatives' for r in rows) == 6 for rows in observations)
    assert all(sum(r['kind'] == 'actual_projection_gradient_SUM_and_Adam' for r in rows) == 3 for rows in observations)
    assert all(sum(r['kind'] == 'all_ranks_rejected' for r in rows) == 6 for rows in observations)
    assert all(r['actual_local_neural_projection_calls'] == 0 for rank in [2,3] for r in observations[rank]
        if r['kind'] == 'actual_projection_gradient_SUM_and_Adam' and r['source_count'] == 2)
