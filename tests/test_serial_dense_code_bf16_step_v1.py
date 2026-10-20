import copy

import pytest
import torch

from slm_lab.codenet_checkpoint_v1 import capture_rng
from slm_lab.own_dense_checkpoint_v1 import make_state,restore_state
from slm_lab.parallel_function_repair_stage_v1 import value_digest
from slm_lab.public_reasoned_code_train_data_v1 import PublicReasonedCodeSFTData
from slm_lab.serial_dense_code_bf16_step_v1 import update
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
from test_own_dense_growth_v1 import model as own_model
from test_public_reasoned_code_train_data_v1 import packet

SOURCES={'synthetic_test_source':'d'*64}


def setup(root):
    torch.set_num_threads(1);packet(root);data=PublicReasonedCodeSFTData(root,purpose='capacity');model,_=own_model()
    optimizer=torch.optim.AdamW([p for _,p in model.language_named_parameters()],lr=.001,weight_decay=0.,foreach=False,fused=False)
    return model,optimizer,data


@pytest.mark.parametrize('profile,factor',[('plain_tokens',1),('eos_weight4',4)])
def test_complete_full_language_update_matches_independent_token_logsoftmax_and_Adam_oracle(tmp_path,profile,factor):
    model,optimizer,data=setup(tmp_path);oracle=copy.deepcopy(model)
    adam=torch.optim.AdamW([p for _,p in oracle.language_named_parameters()],lr=.001,weight_decay=0.,foreach=False,fused=False)
    positions=[0,1,2];denominator=sum(data.information(p)['response_EOS_targets']+factor-1 for p in positions)
    oracle.train();oracle.network.gradient_checkpointing=True;losses=[]
    for position in positions:
        ids,labels=data.sequence(position)
        with torch.autocast('cpu',dtype=torch.bfloat16):logits=oracle(torch.tensor([ids]))['logits']
        for i,target in enumerate(labels[1:],1):
            if target!=-100:losses.append(-logits[0,i-1].float().log_softmax(-1)[target]*(factor if target==2 else 1)/denominator)
    parameters=[p for _,p in oracle.language_named_parameters()]
    torch.stack(losses).sum().backward(inputs=parameters)
    expected_norm=float(torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True));adam.step();adam.zero_grad(set_to_none=True)
    names={id(p) for _,p in model.language_named_parameters()}
    fixed=tensor_digest((n,p) for n,p in model.named_parameters() if id(p) not in names);rng=value_digest(capture_rng('cpu'))
    record=update(model,optimizer,data,positions,profile,allow_mechanics=True)
    assert record['source_metadata']['actual_global_weighted_targets']==denominator
    assert record['gradient_norm_before_clip']==pytest.approx(expected_norm,abs=3e-6,rel=3e-5)
    for (_,a),(_,b) in zip(model.language_named_parameters(),oracle.language_named_parameters()):assert torch.allclose(a,b,atol=2e-6,rtol=3e-5)
    assert tensor_digest((n,p) for n,p in model.named_parameters() if id(p) not in names)==fixed
    assert all(p.grad is None for p in model.parameters()) and value_digest(capture_rng('cpu'))==rng


def test_fresh_full_native_Adam_and_RNG_recovery_reproduces_next_original_source(tmp_path):
    model,optimizer,data=setup(tmp_path);first=update(model,optimizer,data,[0,1,2],'plain_tokens',allow_mechanics=True)
    state=make_state(model,optimizer,sources=SOURCES,completed=1,planned=2,cursor={'next':1},metrics=[first],device='cpu')
    expected=update(model,optimizer,data,[2],'plain_tokens',allow_mechanics=True)
    weights=tensor_digest(model.named_parameters());moments=value_digest(optimizer.state_dict());rng=value_digest(capture_rng('cpu'))
    fresh,_=own_model();adam=torch.optim.AdamW([p for _,p in fresh.language_named_parameters()],lr=.001,weight_decay=0.,foreach=False,fused=False)
    restore_state(fresh,adam,state,expected_sources=SOURCES,planned=2,expected_cursor={'next':1},device='cpu')
    assert update(fresh,adam,data,[2],'plain_tokens',allow_mechanics=True)==expected
    assert tensor_digest(fresh.named_parameters())==weights and value_digest(adam.state_dict())==moments
    assert value_digest(capture_rng('cpu'))==rng


def test_bad_or_nonproduction_scope_rejects_before_full_native_mutation(tmp_path):
    model,optimizer,data=setup(tmp_path);before=tensor_digest(model.named_parameters())
    for positions in [[0,0],[len(data)]]:
        with pytest.raises(ValueError):update(model,optimizer,data,positions,'plain_tokens',allow_mechanics=True)
        assert not optimizer.state and tensor_digest(model.named_parameters())==before
    with pytest.raises(ValueError):update(model,optimizer,data,[0],'plain_tokens')
    assert not optimizer.state and all(p.grad is None for p in model.parameters())
