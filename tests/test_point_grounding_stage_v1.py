import copy
import json
import math
import os
from pathlib import Path

import pytest
import torch
from torch import distributed as dist
from torch.multiprocessing import spawn

from slm_lab.model import ModelConfig,TinyMultimodalLM
from slm_lab.point_grounding_head_v1 import PointGroundingHead
from slm_lab.point_grounding_stage_v1 import PointerStage,make_optimizer,frozen_digest


class OriginalFixture:
    role='train';purpose='optimization'

    def __init__(self,n=37):self.n=n
    def __len__(self):return self.n
    def batch(self,positions,device):
        ids=torch.tensor([[1,4+p%8,6,2] for p in positions],device=device)
        pixels=torch.stack([torch.linspace(-1,1,3*16*16).reshape(3,16,16)+(p%3)*.1 for p in positions]).to(device)
        target=torch.zeros(len(positions),4,device=device)
        for i,p in enumerate(positions):target[i,p%4]=4+p%3
        return dict(input_ids=ids,attention_mask=torch.ones_like(ids,dtype=torch.bool),pixel_values=pixels,
            targets=target,valid_cells=torch.ones_like(target,dtype=torch.bool))


def model():
    torch.manual_seed(1093)
    config=ModelConfig(vocab_size=32,hidden_size=16,intermediate_size=32,num_layers=1,num_heads=4,num_kv_heads=2,
        max_text_length=64,vision_hidden_size=8,vision_intermediate_size=16,vision_layers=1,vision_heads=2,
        image_size=16,patch_size=4,pool_size=2)
    return PointGroundingHead(TinyMultimodalLM(config),rank=8)


def options(world=1,micro=2,accum=4,n=37):
    global_count=world*micro*accum;steps=math.ceil(n/global_count)
    return dict(seed=13977,world_size=world,microbatch_per_rank=micro,accumulation=accum,
        global_queries_per_update=global_count,epochs=1,steps=steps,planned_complete_query_exposures=steps*global_count,
        rounding_repeated_queries=steps*global_count-n,vision_learning_rate=2e-4,head_learning_rate=2e-3,
        betas=[.9,.95],weight_decay=.01,gradient_norm_clip=1.,warmup_fraction=.1,cosine_floor_fraction=.1,
        image_size=16,maximum_whole_query_tokens=34)


def test_complete_query_epoch_rounding_and_bad_resume_rejected_before_live_mutation(tmp_path):
    torch.set_num_threads(1);data=OriginalFixture();opts=options();head=model()
    optimizer=make_optimizer(head,opts);kernel=PointerStage(head,optimizer,data,opts,{'run_plan_sha256':'a'*64},'cpu')
    order=[]
    for step in range(kernel.steps):
        for micro in range(opts['accumulation']):order.extend(kernel.positions(step,0,micro))
    assert len(order)==40 and len(set(order[:37]))==37 and order[37:]==order[:3]
    before=frozen_digest(head);kernel.update();kernel.save(tmp_path/'resume.pt',20*1024**2,0)
    saved=torch.load(tmp_path/'resume.pt',weights_only=True);kernel.update()
    live={n:t.clone() for n,t in head.state_dict().items()};cursor=kernel.completed
    for mutate in [lambda s:s['sampler_state'].update(source_exposures=0),
                   lambda s:s['worker_rng_states'].pop(),
                   lambda s:s['optimizer']['state'][next(iter(s['optimizer']['state']))]['exp_avg'].fill_(float('nan'))]:
        bad=copy.deepcopy(saved);mutate(bad)
        with pytest.raises(ValueError):kernel.resume(bad)
        assert kernel.completed==cursor and all(torch.equal(t,head.state_dict()[n]) for n,t in live.items())
    assert frozen_digest(head)==before
    data.purpose='evaluation'
    with pytest.raises(ValueError,match='optimization role'):PointerStage(head,optimizer,data,opts,{},'cpu')


def worker(rank,folder):
    os.environ['CUDA_VISIBLE_DEVICES']='';torch.set_num_threads(1)
    folder=Path(folder);dist.init_process_group('gloo',init_method='file://'+str(folder/'rendezvous'),rank=rank,world_size=4)
    try:
        opts=options(world=4);data=OriginalFixture();head=model();frozen=frozen_digest(head)
        kernel=PointerStage(head,make_optimizer(head,opts),data,opts,{'run_plan_sha256':'b'*64},'cpu')
        first=kernel.update();kernel.save(folder/'resume.pt',20*1024**2,0)
        uninterrupted=kernel.update();expected={n:t.clone() for n,t in head.state_dict().items()}
        fresh=model();resumed=PointerStage(fresh,make_optimizer(fresh,opts),data,opts,{'run_plan_sha256':'b'*64},'cpu')
        resumed.resume(torch.load(folder/'resume.pt',map_location='cpu',weights_only=True))
        again=resumed.update()
        assert again==uninterrupted and all(torch.equal(t,fresh.state_dict()[n]) for n,t in expected.items())
        assert frozen_digest(fresh)==frozen
        if rank==0:
            torch.save(expected,folder/'distributed-final.pt')
            (folder/'evidence.json').write_text(json.dumps(dict(actual_CPU_workers=4,
                exact_native_resume_next_metric_and_all_tensors=True,frozen_language_critic_unchanged=True,
                first=first,second=uninterrupted)))
    finally:dist.destroy_process_group()


def test_actual_four_worker_mean_objective_and_exact_collective_resume(tmp_path):
    torch.set_num_threads(1)
    spawn(worker,args=(str(tmp_path),),nprocs=4,join=True)
    opts=options(world=1,micro=8,accum=4);head=model();kernel=PointerStage(head,make_optimizer(head,opts),OriginalFixture(),opts,{'run_plan_sha256':'c'*64},'cpu')
    for _ in range(kernel.steps):kernel.update()
    actual=torch.load(tmp_path/'distributed-final.pt',weights_only=True)
    for name,tensor in head.state_dict().items():
        assert torch.allclose(tensor,actual[name],atol=2e-6,rtol=2e-5),name
    evidence=json.loads((tmp_path/'evidence.json').read_text())
    assert evidence['actual_CPU_workers']==4 and evidence['exact_native_resume_next_metric_and_all_tensors']
    draws=[p for metric in [evidence['first'],evidence['second']] for micro in range(4)
        for row in metric['workers'] for p in row['micros'][micro]['positions']]
    assert len(draws)==64 and len(set(draws[:37]))==37 and draws[37:]==draws[:27]
