"""Four real CPU ranks on a tiny own replica; never real-model training metrics."""
from contextlib import nullcontext
from collections import OrderedDict
import copy
from datetime import timedelta
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch
from torch import distributed as dist
import torch.multiprocessing as mp

from slm_lab.code_residual_branch_v1 import RoutedCodeResidual
from slm_lab.code_residual_collective_checkpoint_v2 import (fingerprint,
    save_collective_residual,restore_collective_residual)
from slm_lab.codenet_checkpoint_v1 import capture_rng
from slm_lab.codenet_runner_v1 import FixedReductionModel,average_gradients
from slm_lab.codenet_stage_v1 import Objective,objective_gradients
from slm_lab.corpus import file_hash
from slm_lab.model import ModelConfig,TinyMultimodalLM
from slm_lab.recurrent_ablation import state_digest


SOURCES=dict(base_checkpoint_sha256='a'*64,tokenizer_sha256='b'*64,run_plan_sha256='c'*64)


def model_and_optimizer():
    torch.manual_seed(31)
    cfg=ModelConfig(vocab_size=32,hidden_size=16,intermediate_size=32,num_layers=2,num_heads=4,num_kv_heads=2,
        max_text_length=64,vision_hidden_size=8,vision_intermediate_size=16,vision_layers=1,vision_heads=2,
        image_size=8,patch_size=4,pool_size=2)
    model=RoutedCodeResidual(TinyMultimodalLM(cfg),width=12);model.code.gradient_checkpointing=True
    optimizer=torch.optim.AdamW(model.adapter_parameters(),lr=.001,foreach=False)
    return model,optimizer,FixedReductionModel(model.code)


def sampler(step):return dict(updates=step,code_example_exposures=8*step,replay_example_exposures=4*step)


class Rows:
    def batch(self,rows,field,device):
        assert len(rows)==1
        row=rows[0];ids=torch.tensor([row['ids']],device=device)
        labels=ids.clone();labels[:,:2]=-100
        pixels=row['pixels']
        return dict(input_ids=ids,labels=labels,pixel_values=pixels.to(device) if pixels is not None else None)


def stochastic_rows(rank,step):
    draws=(random.random(),float(np.random.random()),float(torch.rand(())))
    offset=int(sum(draws)*5)%10
    def row(length,image=False):
        return dict(ids=[1,3]+[4+(i+offset)%24 for i in range(length)]+[2],
            pixels=torch.rand(1,3,8,8) if image else None)
    return dict(code=[row(rank+1),row(7-rank)],replay=[row(rank+2,rank%2==0)],draws=draws)


def worker(rank,rendezvous,folder):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=4,timeout=timedelta(seconds=60))
    try:
        folder=Path(folder);model,opt,wrapped=model_and_optimizer();base_before=state_digest(model.base.state_dict())
        torch.manual_seed(101+rank);random.seed(201+rank);np.random.seed(301+rank)
        encoder=Rows();metrics=[]
        def update(step,compare_global=False):
            model.train();opt.zero_grad(set_to_none=True);data=stochastic_rows(rank,step)
            result=objective_gradients(wrapped,[Objective('code',encoder,[[r] for r in data['code']],'response',1.),
                Objective('replay',encoder,[[r] for r in data['replay']],'response',.5)],'cpu',nullcontext,world=4)
            average_gradients(wrapped,'cpu',4)
            if compare_global:
                all_data=[None]*4;dist.all_gather_object(all_data,data)
                if rank==0:
                    reference=copy.deepcopy(model);reference.zero_grad(set_to_none=True)
                    for key,coefficient in [('code',1.),('replay',.5)]:
                        rows=[r for d in all_data for r in d[key]]
                        targets=sum(len(r['ids'])-2 for r in rows)
                        for row in rows:
                            batch=encoder.batch([row],'response','cpu')
                            (reference.code(**batch)['loss']*coefficient*(len(row['ids'])-2)/targets).backward()
                    for actual,expected in zip(model.adapter_parameters(),reference.adapter_parameters()):
                        torch.testing.assert_close(actual.grad,expected.grad,rtol=3e-5,atol=1e-7)
                dist.barrier()
            assert all(p.grad is None for p in model.base.parameters())
            torch.nn.utils.clip_grad_norm_(model.adapter_parameters(),1.,error_if_nonfinite=True);opt.step();opt.zero_grad(set_to_none=True)
            assert state_digest(model.base.state_dict())==base_before
            metrics.append(dict(step=step+1,objective=result['summed_objective']))
        def before_state():return fingerprint(dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=capture_rng('cpu')))
        def restore(path,sha,**extra):
            kwargs=dict(expected_checkpoint_sha256=sha,**SOURCES,planned_steps=2,
                sampler_state_for_step=sampler,device='cpu');kwargs.update(extra)
            return restore_collective_residual(path,model,opt,**kwargs)
        update(0,compare_global=True)
        path=folder/'resume.pt'
        saved=save_collective_residual(path,model,opt,**SOURCES,completed_steps=1,planned_steps=2,
            sampler_state=sampler(1),metrics=metrics,device='cpu',maximum_bytes=1_000_000,minimum_free_bytes=0)
        payload=torch.load(path,weights_only=True,mmap=True)
        assert len({fingerprint(r) for r in payload['worker_rng_states']})==4
        validation_cases=0
        for fault in ['one_rank_hash','one_rank_plan','one_rank_sampler']:
            before=before_state();extra={};sha=saved['checkpoint_sha256']
            if fault=='one_rank_hash' and rank==1:sha='0'*64
            if fault=='one_rank_plan' and rank==2:extra['run_plan_sha256']='d'*64
            if fault=='one_rank_sampler' and rank==3:
                extra['sampler_state_for_step']=lambda step:dict(sampler(step),code_example_exposures=999)
            with pytest.raises(ValueError,match='Collective preflight failed'):restore(path,sha,**extra)
            assert before_state()==before;validation_cases+=1
        for fault in ['world','missing_rng','finite_weight_corruption','nonfinite_Adam','parameter_order','typed_progress']:
            bad_path=folder/f'{fault}.pt'
            if rank==0:
                bad=copy.deepcopy(payload)
                if fault=='world':bad['world_size']=3
                if fault=='missing_rng':bad['worker_rng_states'].pop()
                if fault=='finite_weight_corruption':next(iter(bad['residuals'].values())).flatten()[0]+=.01
                if fault=='nonfinite_Adam':bad['optimizer']['state'][0]['exp_avg'].flatten()[0]=float('nan')
                if fault=='parameter_order':bad['residual_parameter_names'].reverse()
                if fault=='typed_progress':bad['sampler_state']['updates']=True
                torch.save(bad,bad_path)
            dist.barrier();before=before_state()
            with pytest.raises(ValueError,match='Collective preflight failed'):restore(bad_path,file_hash(bad_path))
            assert before_state()==before;validation_cases+=1
        # A capacity failure must reach every rank and preserve the previous file.
        before=before_state();old_hash=file_hash(path)
        with pytest.raises(ValueError,match='Collective snapshot failed'):
            save_collective_residual(path,model,opt,**SOURCES,completed_steps=1,planned_steps=2,
                sampler_state=sampler(1),metrics=metrics,device='cpu',maximum_bytes=128,minimum_free_bytes=0)
        assert file_hash(path)==old_hash and before_state()==before and not path.with_suffix('.pt.pending').exists()
        # A finite one-rank weight drift must never create a supposedly shared state.
        parameter=model.adapter_parameters()[0];old=parameter.detach().clone()
        if rank==1:
            with torch.no_grad():parameter.flatten()[0]+=.01
        before=before_state()
        with pytest.raises(ValueError,match='Workers disagree'):
            save_collective_residual(folder/'drift.pt',model,opt,**SOURCES,completed_steps=1,planned_steps=2,
                sampler_state=sampler(1),metrics=metrics,device='cpu',maximum_bytes=1_000_000,minimum_free_bytes=0)
        assert before_state()==before and not (folder/'drift.pt').exists()
        with torch.no_grad():parameter.copy_(old)
        update(1);expected=before_state()
        expected_next=(random.random(),float(np.random.random()),float(torch.rand(())))
        model,opt,wrapped=model_and_optimizer()
        step,progress,restored_metrics=restore(path,saved['checkpoint_sha256'])
        assert step==1 and progress==sampler(1);metrics[:]=restored_metrics
        update(1)
        assert before_state()==expected
        assert (random.random(),float(np.random.random()),float(torch.rand(())))==expected_next
        complete=save_collective_residual(folder/'complete.pt',model,opt,**SOURCES,completed_steps=2,planned_steps=2,
            sampler_state=sampler(2),metrics=metrics,device='cpu',maximum_bytes=1_000_000,minimum_free_bytes=0)
        before=before_state()
        with pytest.raises(ValueError,match='Collective preflight failed'):restore(folder/'complete.pt',complete['checkpoint_sha256'])
        assert before_state()==before
        assert restore(folder/'complete.pt',complete['checkpoint_sha256'],allow_completed=True)[0]==2
        assert before_state()==before
        (folder/f'rank-{rank}.json').write_text(json.dumps(dict(rank=rank,world=4,
            own_tiny_replica_parameters=model.parameter_counts(),unequal_global_target_mean_verified=True,
            text_and_image_replay_executed=True,frozen_base_unchanged=True,
            exact_residual_Adam_and_all_rank_RNG_restart=True,rank_distinct_rng_verified=True,
            malformed_resume_cases_rejected_without_mutation=validation_cases,
            rank_zero_capacity_failure_broadcast_and_previous_checkpoint_preserved=True,
            one_rank_weight_drift_rejected=True,completed_resume_without_extra_updates=True)))
    finally:dist.destroy_process_group()


def test_context_must_be_initialized():
    with pytest.raises(ValueError,match='initialized'):
        save_collective_residual('unused',None,None,**SOURCES,completed_steps=1,planned_steps=2,
            sampler_state=sampler(1),metrics=[dict(step=1)],device='cpu',maximum_bytes=1000,minimum_free_bytes=0)


def test_fingerprint_distinguishes_nesting_types_scalars_and_tensor_bytes():
    assert fingerprint({'a':{'b':1},'c':2})!=fingerprint({'a':{'b':1,'c':2}})
    assert fingerprint(1)!=fingerprint(True)
    assert fingerprint(torch.tensor(1.))!=fingerprint(torch.tensor(2.))
    assert fingerprint(OrderedDict(a=torch.tensor(1.)))==fingerprint({'a':torch.tensor(1.)})
    with pytest.raises(ValueError):fingerprint(float('nan'))


def test_four_workers_exact_resume_collective_rejection_and_global_objective(tmp_path):
    mp.spawn(worker,args=(str(tmp_path/'rendezvous'),str(tmp_path)),nprocs=4,join=True)
    for rank in range(4):
        result=json.loads((tmp_path/f'rank-{rank}.json').read_text())
        assert result['exact_residual_Adam_and_all_rank_RNG_restart']
        assert result['malformed_resume_cases_rejected_without_mutation']==9
