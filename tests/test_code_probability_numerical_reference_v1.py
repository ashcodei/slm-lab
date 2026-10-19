import torch

from slm_lab import model as model_module
from slm_lab.code_probability_numerical_reference_v1 import cached_actions, compare
from test_own_dense_actor_v1 import actor


def test_same_weight_FP64_cached_full_oracle_keeps_original_FP32_student_and_RNG_exact():
    torch.set_num_threads(1)
    model = actor().eval()
    prefix,actions = [1,4,7],[8,5,2]
    with torch.no_grad():
        logits = model(torch.tensor([prefix+actions]))['logits'][0,len(prefix)-1:-1]
        full = logits.log_softmax(-1).gather(1,torch.tensor(actions)[:,None]).squeeze(1)
        cached = cached_actions(model,prefix,actions)
    observation = dict(complete_prompt_ids=prefix,response_ids=actions,
        cached_behavior_logps=cached.tolist(),full_behavior_logps=full.tolist())
    before = {n:p.detach().clone() for n,p in model.named_parameters()}
    RNG = torch.get_rng_state()
    rotary = model_module.rotary
    result = compare(model,observation)
    assert result['recorded_cached_FP32_replay_exact']
    assert result['FP64_full_vs_cached_max_absolute_difference'] < 1e-9
    assert result['actual_counts'] == model.counts()
    assert model_module.rotary is rotary and torch.equal(RNG,torch.get_rng_state())
    assert all(torch.equal(p,before[n]) and p.dtype == torch.float32 for n,p in model.named_parameters())
