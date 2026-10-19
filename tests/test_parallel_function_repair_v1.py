"""Actual small neural replica/gradient mechanics, never a quality benchmark."""
import copy
from datetime import timedelta
import json
from pathlib import Path

import pytest
import torch
from torch import distributed as dist
from torch.multiprocessing import spawn

from slm_lab.full_function_repair_stage_v1 import gradients
from slm_lab.parallel_function_repair_v1 import update
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
from test_full_function_repair_stage_v1 import fixture

PROFILES=['plain_repeat','semantic_repair','weighted_semantic_repair']


def optimizer(model,options):
    return torch.optim.AdamW([p for _,p in model.named_selected(('actor',))],lr=options['learning_rate'],
        betas=(.9,.95),weight_decay=.01,foreach=False)


def serial_step(model,opt,data,positions,profile,options,weights,amp=None):
    opt.zero_grad(set_to_none=True);record=gradients(model,data,positions,profile,options,weights,amp=amp)
    torch.nn.utils.clip_grad_norm_([p for _,p in model.named_selected(('actor',))],options['gradient_clip'],error_if_nonfinite=True)
    opt.step();opt.zero_grad(set_to_none=True);return record


@pytest.mark.parametrize('profile',PROFILES)
@pytest.mark.parametrize('precision',['fp32','bf16'])
def test_one_actual_replica_matches_existing_complete_source_global_objective_and_Adam(profile,precision):
    model,data,weights,options=fixture();other=copy.deepcopy(model)
    opt=optimizer(model,options);reference=optimizer(other,options);frozen=tensor_digest(model.base.named_parameters())
    amp=(lambda:torch.autocast('cpu',dtype=torch.bfloat16)) if precision=='bf16' else None
    for positions in [[0,1,2],[1]]:
        actual=update(model,opt,data,positions,profile,options,weights,amp=amp)
        expected=serial_step(other,reference,data,positions,profile,options,weights,amp)
        assert actual['response_EOS_targets']==expected['response_EOS_targets']
        assert actual['original_token_mean_NLL']==expected['original_token_mean_NLL']
        assert actual['weighted_token_mean_NLL']==expected['weighted_token_mean_NLL']
        assert all(torch.equal(a,b) for a,b in zip(model.parts.parameters(),other.parts.parameters()))
    assert frozen==tensor_digest(model.base.named_parameters())


def worker(rank,folder):
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    dist.init_process_group('gloo',init_method='file://'+str(Path(folder)/'rendezvous'),rank=rank,world_size=4,timeout=timedelta(minutes=3))
    proofs=[]
    try:
        for profile,precision in [(p,q) for q in ['fp32','bf16'] for p in PROFILES]:
            torch.manual_seed(982311);model,data,weights,options=fixture();other=copy.deepcopy(model)
            opt=optimizer(model,options);reference=optimizer(other,options)
            amp=(lambda:torch.autocast('cpu',dtype=torch.bfloat16)) if precision=='bf16' else None
            frozen=tensor_digest(model.base.named_parameters());critic=model.parts['critic'].weight.detach().clone()
            calls=[0];allowed=[False];original=model.forward
            def forward(*args,**kwargs):
                if not allowed[0]:raise AssertionError('An empty tail replica performed a fabricated neural forward')
                calls[0]+=1;return original(*args,**kwargs)
            model.forward=forward;maximum=0.;records=[]
            for positions in [[0,1,2],[1]]:
                allowed[0]=bool(positions[rank::4]);before=calls[0]
                actual=update(model,opt,data,positions,profile,options,weights,rank=rank,world=4,amp=amp)
                expected=serial_step(other,reference,data,positions,profile,options,weights,amp)
                assert calls[0]-before==actual['actual_rank_observations'][rank]['actual_microbatches']
                observed=actual['actual_rank_observations'][rank]['actual_precision_per_forward']
                assert all(v['autocast_enabled']==(precision=='bf16') and v['actual_logits_dtype']==('torch.bfloat16' if precision=='bf16' else 'torch.float32') for v in observed)
                assert actual['response_EOS_targets']==expected['response_EOS_targets']
                assert abs(actual['original_token_mean_NLL']-expected['original_token_mean_NLL'])<2e-6
                assert abs(actual['weighted_token_mean_NLL']-expected['weighted_token_mean_NLL'])<2e-6
                maximum=max(maximum,max(float((a-b).abs().max()) for a,b in zip(model.parts.parameters(),other.parts.parameters())))
                assert maximum<5e-6
                digests=[None]*4;dist.all_gather_object(digests,tensor_digest(model.parts.named_parameters()))
                assert len(set(digests))==1;records.append(actual)
            before=tensor_digest(model.parts.named_parameters());adam=copy.deepcopy(opt.state_dict());rng=torch.get_rng_state().clone()
            if rank==1:data.purpose='source_preparation'
            with pytest.raises(ValueError,match='Coordinated preflight'):update(model,opt,data,[0],'semantic_repair',options,weights,rank=rank,world=4)
            assert tensor_digest(model.parts.named_parameters())==before and torch.equal(rng,torch.get_rng_state())
            for key,value in opt.state_dict()['state'].items():
                for name,tensor in value.items():assert torch.equal(tensor,adam['state'][key][name])
            assert tensor_digest(model.base.named_parameters())==frozen and torch.equal(model.parts['critic'].weight,critic)
            assert all(p.grad is None for p in model.base.parameters()) and not torch.cuda.is_initialized()
            proofs.append(dict(profile=profile,precision=precision,actual_rank=rank,actual_neural_forwards=calls[0],actual_global_updates=2,
                source_positions=[[0,1,2],[1]],serial_max_absolute_actor_parameter_error=maximum,
                all_replicas_actor_values_identical=True,invalid_single_replica_source_role_rejected_before_mutation=True,
                full_model_GPU_capacity_and_recovery_not_proved=True,records=records))
        (Path(folder)/(str(rank)+'.json')).write_text(json.dumps(proofs))
    finally:dist.destroy_process_group()


def test_actual_four_replicas_match_serial_global_targets_and_zero_forward_uneven_tails(tmp_path):
    spawn(worker,args=(str(tmp_path),),nprocs=4,join=True)
    proof=[json.loads((tmp_path/(str(rank)+'.json')).read_text()) for rank in range(4)]
    assert all(len(rows)==6 and all(row['actual_global_updates']==2 for row in rows) for rows in proof)
    assert all(row['actual_neural_forwards']==0 for row in proof[3])
    assert all(row['records'][1]['actual_rank_source_shards']==[[1],[],[],[]] for rows in proof for row in rows)
