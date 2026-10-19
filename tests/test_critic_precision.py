import torch
from torch.nn import functional as F

from test_model import tiny


def test_small_reward_differences_survive_bf16_autocast():
    model=tiny()
    with torch.no_grad():
        model.value_head.weight.zero_();model.value_head.weight[0,0]=1
    hidden=torch.zeros(2,1,model.config.hidden_size);hidden[0,0,0]=1;hidden[1,0,0]=1.001
    with torch.autocast('cpu',dtype=torch.bfloat16):
        rounded=model.value_head(hidden).squeeze(-1)
        actual=model.critic(hidden)
    assert rounded[0]==rounded[1]
    assert actual.dtype==torch.float32 and actual[1]>actual[0]


def test_critic_projection_matches_fp32_reference_and_keeps_training_gradients():
    model=tiny().train()
    with torch.no_grad():model.value_head.weight.normal_()
    ids=torch.randint(0,model.config.vocab_size,(2,8))
    with torch.autocast('cpu',dtype=torch.bfloat16):
        out=model(ids,return_hidden=True)
    expected=F.linear(out['hidden'].float(),model.value_head.weight.float()).squeeze(-1)
    torch.testing.assert_close(out['values'],expected,rtol=0,atol=0)
    out['values'].sum().backward()
    assert model.value_head.weight.grad.abs().sum()>0
    assert model.embedding.weight.grad.abs().sum()>0
