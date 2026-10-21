import copy
from pathlib import Path
import random

import numpy as np
import pytest
import torch
from torch import distributed as dist
from torch import multiprocessing as mp

from slm_lab.codenet_checkpoint_v1 import capture_rng, atomic_checkpoint
from slm_lab.distributed_dense_code_step_v1 import update, gather
from slm_lab.distributed_dense_code_state_v1 import make, restore
from slm_lab.parallel_function_repair_stage_v1 import value_digest
from slm_lab.public_reasoned_code_train_data_v1 import PublicReasonedCodeSFTData
from slm_lab.serial_dense_code_bf16_step_v2 import update as serial
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
from test_own_dense_growth_v1 import model as own_model
from test_public_reasoned_code_train_data_v1 import packet

SOURCES = {'controlled_fixture': 'c'*64}


def adam(model):
    return torch.optim.AdamW([p for _, p in model.language_named_parameters()], lr=1e-5,
        betas=(.9,.95), eps=1e-8, weight_decay=.01, foreach=False, fused=False)


def worker(rank, root, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://'+rendezvous, rank=rank, world_size=4)
    try:
        random.seed(900+rank);np.random.seed(900+rank);torch.manual_seed(900+rank)
        data=PublicReasonedCodeSFTData(root)
        model,_=own_model();optimizer=adam(model)
        original_fixed=tensor_digest((n,p) for n,p in model.named_parameters()
            if id(p) not in {id(p) for _,p in model.language_named_parameters()})
        oracle=copy.deepcopy(model);reference=adam(oracle)
        expected=serial(oracle,reference,data,[0,1,2],'plain_tokens',allow_mechanics=True)
        first=update(model,optimizer,data,[0,1,2],allow_mechanics=True)
        assert first['rank_source_counts']==[1,1,1,0]
        assert first['original_token_mean_NLL']==expected['original_token_mean_NLL']
        for (_,p),(_,q) in zip(model.language_named_parameters(),oracle.language_named_parameters()):
            torch.testing.assert_close(p,q,atol=2e-6,rtol=3e-5)
        assert len(set(gather(tensor_digest(model.named_parameters()))))==1
        state=make(model,optimizer,sources=SOURCES,completed=1,planned=3,metrics=[first],device='cpu')
        if rank==0:
            atomic_checkpoint(Path(root)/'state.pt',state,10*1024**2,0)
        dist.barrier()
        second=update(model,optimizer,data,[2],allow_mechanics=True)
        target=(tensor_digest(model.named_parameters()),value_digest(optimizer.state_dict()),value_digest(capture_rng('cpu')))
        fresh,_=own_model();fresh_adam=adam(fresh)
        saved=torch.load(Path(root)/'state.pt',weights_only=True)
        done,rows=restore(fresh,fresh_adam,saved,sources=SOURCES,planned=3,device='cpu')
        assert done==1 and rows==[first]
        assert update(fresh,fresh_adam,data,[2],allow_mechanics=True)==second
        assert target==(tensor_digest(fresh.named_parameters()),value_digest(fresh_adam.state_dict()),value_digest(capture_rng('cpu')))
        assert original_fixed==tensor_digest((n,p) for n,p in fresh.named_parameters()
            if id(p) not in {id(p) for _,p in fresh.language_named_parameters()})
        pristine=(tensor_digest(fresh.named_parameters()),value_digest(fresh_adam.state_dict()),value_digest(capture_rng('cpu')))
        corrupted=copy.deepcopy(saved)
        if rank==2:corrupted['common']['optimizer']['state'][0]['exp_avg'].fill_(float('nan'))
        with pytest.raises(ValueError):restore(fresh,fresh_adam,corrupted,sources=SOURCES,planned=3,device='cpu')
        assert pristine==(tensor_digest(fresh.named_parameters()),value_digest(fresh_adam.state_dict()),value_digest(capture_rng('cpu')))
        with pytest.raises(ValueError):
            update(fresh,fresh_adam,data,[0,0] if rank==1 else [0,1,2],allow_mechanics=True)
        assert pristine==(tensor_digest(fresh.named_parameters()),value_digest(fresh_adam.state_dict()),value_digest(capture_rng('cpu')))
        (Path(root)/('rank-'+str(rank)+'.passed')).write_text('actual four-process gradients, tail and recovery passed\n')
    finally:
        dist.destroy_process_group()


def test_four_actual_processes_match_serial_oracle_and_fresh_all_rank_Adam_RNG_recovery(tmp_path):
    packet(tmp_path)
    mp.spawn(worker,args=(str(tmp_path),str(tmp_path/'rendezvous')),nprocs=4,join=True)
    assert len(list(tmp_path.glob('rank-*.passed')))==4
