import copy
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from slm_lab.codenet_checkpoint_v1 import capture_rng
from slm_lab.fast_joint_stage_v1 import make_optimizer
from slm_lab.public_pretraining_checkpoint_v1 import make_state, restore_state
from slm_lab.public_pretraining_data_v1 import RAW, TokenBudgetSchedule
from slm_lab.public_pretraining_update_v1 import update
from test_fast_joint_v1 import batch, options
from test_public_pretraining_update_v1 import raw_batch
from test_own_dense_growth_v1 import model as make_model
from test_own_alignment_checkpoint_v1 import assert_nested_equal


def fixture():
    model,_ = make_model();setting = options();optimizer = make_optimizer(model,setting)
    batches = {RAW:raw_batch(),'image':batch(image=True),'words':batch(image=True,word=True)}
    d = SimpleNamespace(role='train',purpose='optimization',retention_identity_sha256='a'*64,
        lengths={n:b['attention_mask'].sum(1).tolist() for n,b in batches.items()},
        targets={n:(b['labels'][:,1:]!=-100).sum(1).tolist() for n,b in batches.items()},
        kinds={RAW:'text','image':'image','words':'word'},retention_entries=[('image',[0,1]),('words',[0,1])])
    def batch_at(name,positions,device):
        return {k:(v[positions].to(device) if isinstance(v,torch.Tensor) else [v[p] for p in positions]) for k,v in batches[name].items()}
    d.batch = batch_at
    schedule = TokenBudgetSchedule(d,image_prefix=16,context_budget=100,maximum_rows=1,seed=21)
    plan = dict(initial_dense_settings=dict(initializer_sha256='a'*64),actual_counts=model.counts(),
        planned_steps=len(schedule),training_options=setting)
    return model,optimizer,d,schedule,plan,setting


def draws():return [random.random(),float(np.random.random()),float(torch.rand(()))]


def test_real_source_modality_Adam_and_private_RNG_recovery_reproduces_the_exact_next_update():
    model,opt,data,schedule,plan,setting = fixture();records = []
    for i in range(len(schedule)-1):records.append(update(model,opt,data,schedule,i,setting))
    state = make_state(model,opt,plan,'b'*64,len(records),records,[capture_rng('cpu')],'capacity',schedule=schedule,data=data)
    expected_draws = draws();expected = update(model,opt,data,schedule,len(records),setting)
    weights = {n:p.detach().clone() for n,p in model.named_parameters()};moments = copy.deepcopy(opt.state_dict())
    count,actual = restore_state(model,opt,state,plan,'b'*64,'cpu',0,1,'capacity',schedule=schedule,data=data)
    assert count == len(records) and actual == records and draws() == expected_draws
    assert update(model,opt,data,schedule,count,setting) == expected
    assert all(torch.equal(p,weights[n]) for n,p in model.named_parameters())
    assert_nested_equal(moments,opt.state_dict())
    for corrupt in ['position','rate','clock','missing_leaf','RNG']:
        bad = copy.deepcopy(state);core = bad['core']
        if corrupt == 'position':core['metrics'][0]['global_original_positions'] = [-1]
        elif corrupt == 'rate':core['metrics'][0]['learning_rates']['language'] *= 2
        elif corrupt == 'clock':next(iter(core['optimizer']['state'].values()))['step'] += 1
        elif corrupt == 'missing_leaf':core['optimizer']['state'].pop(next(iter(core['optimizer']['state'])))
        else:core['rngs'][0]['torch'] = torch.zeros(4,dtype=torch.uint8)
        rng = capture_rng('cpu')
        with pytest.raises((ValueError,RuntimeError)):
            restore_state(model,opt,bad,plan,'b'*64,'cpu',0,1,'capacity',schedule=schedule,data=data)
        assert all(torch.equal(p,weights[n]) for n,p in model.named_parameters())
        assert_nested_equal(moments,opt.state_dict());assert_nested_equal(rng,capture_rng('cpu'))
