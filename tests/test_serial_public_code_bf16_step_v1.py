import copy

import pytest
import torch

from slm_lab.codenet_checkpoint_v1 import capture_rng
from slm_lab.own_dense_actor_checkpoint_v1 import make_state, restore_state
from slm_lab.parallel_function_repair_stage_v1 import value_digest
from slm_lab.public_reasoned_code_train_data_v1 import PublicReasonedCodeSFTData
from slm_lab.serial_public_code_bf16_step_v1 import update
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
from test_parallel_public_reasoned_code_SFT_stage_v1 import own_model, SOURCES
from test_public_reasoned_code_train_data_v1 import packet


def setup(root):
    torch.set_num_threads(1); packet(root); data=PublicReasonedCodeSFTData(root,purpose='capacity')
    model=own_model(); adam=torch.optim.AdamW([p for _,p in model.named_selected(('actor',))],lr=.001,weight_decay=0.,foreach=False)
    return model,adam,data


@pytest.mark.parametrize('profile,factor',[('plain_tokens',1),('eos_weight4',4)])
def test_BF16_complete_batch_matches_independent_logsoftmax_gradient_and_single_Adam_oracle(tmp_path,profile,factor):
    model,adam,data=setup(tmp_path); oracle=copy.deepcopy(model)
    ref=torch.optim.AdamW([p for _,p in oracle.named_selected(('actor',))],lr=.001,weight_decay=0.,foreach=False)
    positions=[0,1,2];denominator=sum(data.information(p)['response_EOS_targets']+factor-1 for p in positions)
    oracle.train();oracle.policy.gradient_checkpointing=True;losses=[]
    for position in positions:
        ids,labels=data.sequence(position)
        with torch.autocast('cpu',dtype=torch.bfloat16):scores=oracle(torch.tensor([ids]))['logits']
        for i,target in enumerate(labels[1:],1):
            if target!=-100:losses.append(-scores[0,i-1].float().log_softmax(-1)[target]*(factor if target==2 else 1)/denominator)
    torch.stack(losses).sum().backward()
    expected_norm=float(torch.nn.utils.clip_grad_norm_([p for _,p in oracle.named_selected(('actor',))],1.,error_if_nonfinite=True))
    ref.step();ref.zero_grad(set_to_none=True)
    base=tensor_digest(model.base.named_parameters());critic=model.parts['critic'].weight.detach().clone();rng=value_digest(capture_rng('cpu'))
    row=update(model,adam,data,positions,profile,allow_mechanics=True)
    assert row['source_metadata']['actual_global_weighted_targets']==denominator
    assert row['gradient_norm_before_clip']==pytest.approx(expected_norm,abs=3e-6,rel=3e-5)
    for (_,a),(_,b) in zip(model.named_selected(('actor',)),oracle.named_selected(('actor',))):
        assert torch.allclose(a,b,atol=2e-6,rtol=3e-5)
    assert tensor_digest(model.base.named_parameters())==base and torch.equal(critic,model.parts['critic'].weight)
    assert value_digest(capture_rng('cpu'))==rng
    assert all(m['forward_dtype']=='torch.bfloat16' and m['loss_dtype']=='torch.float32' for m in row['actual_microbatches'])


def test_actual_two_update_selected_Adam_RNG_recovery_replays_the_same_next_complete_source(tmp_path):
    model,adam,data=setup(tmp_path)
    first=update(model,adam,data,[0,1,2],'plain_tokens',allow_mechanics=True)
    state=make_state(model,adam,groups=('actor',),sources=SOURCES,completed=1,planned=2,cursor={'next':1},metrics=[first],device='cpu')
    expected=update(model,adam,data,[2],'plain_tokens',allow_mechanics=True)
    weights=tensor_digest(model.parts.named_parameters());moments=value_digest(adam.state_dict());rng=value_digest(capture_rng('cpu'))
    restore_state(model,adam,state,groups=('actor',),expected_sources=SOURCES,planned=2,expected_cursor={'next':1},device='cpu')
    assert update(model,adam,data,[2],'plain_tokens',allow_mechanics=True)==expected
    assert tensor_digest(model.parts.named_parameters())==weights and value_digest(adam.state_dict())==moments
    assert value_digest(capture_rng('cpu'))==rng


def test_bad_duplicate_reserved_or_unadmitted_scope_rejects_before_an_optimizer_call(tmp_path):
    model,adam,data=setup(tmp_path);before=tensor_digest(model.named_parameters())
    for positions in [[0,0],[len(data)]]:
        with pytest.raises(ValueError):update(model,adam,data,positions,'plain_tokens',allow_mechanics=True)
        assert not adam.state and tensor_digest(model.named_parameters())==before
    with pytest.raises(ValueError):update(model,adam,data,[0],'plain_tokens')
    assert not adam.state and all(p.grad is None for p in model.parameters())
