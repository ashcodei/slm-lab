from pathlib import Path
from types import SimpleNamespace

import torch
from tokenizers import Tokenizer

from slm_lab.execution_group_controlled_v2 import ControlledExecutionGroup
from slm_lab.execution_group_prefix_reuse_v5 import sample_group
from slm_lab.model import ModelConfig
from slm_lab.own_dense_growth_v1 import OwnDenseAssistant,new_backbone
from slm_lab.own_dense_actor_v1 import OwnDenseActor

ROOT=Path(__file__).resolve().parents[1]


def test_shared_prefix_preserves_every_sample_probability_RNG_draw_and_original_cache():
    torch.set_num_threads(1)
    c=ModelConfig(vocab_size=32768,hidden_size=32,intermediate_size=64,num_layers=2,num_heads=4,num_kv_heads=2,
        max_text_length=64,vision_hidden_size=8,vision_intermediate_size=16,vision_layers=1,vision_heads=2,image_size=16,patch_size=4,pool_size=2)
    model=OwnDenseActor(OwnDenseAssistant(new_backbone(c,958713),reward_hidden=8,word_temporal=4,pointing_rank=8,router_hidden=8,detector_hidden=4),rank=2,seed=958713)
    class FixtureReward:
        def score(self,key,response):return dict(reward=0.,test_fixture_not_production_reward=True)
    stage=SimpleNamespace(model=model,group_size=2,max_new_tokens=6,reward=FixtureReward(),tokenizer=Tokenizer.from_file(str(ROOT/'data/tokenizer-corpus/tokenizer.json')))
    original_prefill=model.policy.prefill;count=[];initial_caches=[]
    def prefill(*args,**kwargs):
        result=original_prefill(*args,**kwargs);count.append(1);initial_caches.append((result['cache'],[t.clone() for p in result['cache'].layers for t in p]));return result
    model.policy.prefill=prefill
    prefix=[1,261,267,71];torch.manual_seed(958714);RNG=torch.get_rng_state();old=ControlledExecutionGroup.sample(stage,prefix,'fixture');after=torch.get_rng_state()
    assert len(count)==2
    count.clear();initial_caches.clear();torch.set_rng_state(RNG);new=sample_group(stage,prefix,'fixture')
    assert len(count)==1 and torch.equal(torch.get_rng_state(),after)
    for a,b in zip(old,new):assert a=={k:b[k] for k in a}
    for cache,tensors in initial_caches:
        assert all(torch.equal(a,b) for a,b in zip([t for p in cache.layers for t in p],tensors))
    with torch.no_grad():
        for row in new:
            ids=torch.tensor([prefix+row['response_ids']]);logits=model(ids)['logits'][0,len(prefix)-1:-1]
            actual=logits.log_softmax(-1)[torch.arange(len(row['response_ids'])),row['response_ids']]
            assert torch.allclose(actual,torch.tensor(row['actual_cached_behavior_logps']),atol=1e-4,rtol=1e-5)
    model.policy.prefill=original_prefill
