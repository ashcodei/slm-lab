from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer

from slm_lab.dense_foundation_inference_v1 import DenseNativeView
from slm_lab.model import ModelConfig
from slm_lab.native_prefix_rollout_CPU_v1 import sample
from slm_lab.own_dense_growth_v1 import OwnDenseAssistant,new_backbone


def test_actual_private_generator_rollout_replays_without_weight_or_global_rng_mutation():
    torch.set_num_threads(1)
    tokenizer=Tokenizer.from_file(str(Path(__file__).resolve().parents[1]/'data/tokenizer-corpus/tokenizer.json'))
    config=ModelConfig(vocab_size=32768,hidden_size=32,num_layers=2,num_heads=4,num_kv_heads=2,
        intermediate_size=64,max_text_length=256,vision_hidden_size=32,vision_intermediate_size=64,vision_layers=1,vision_heads=4,image_size=32,patch_size=16)
    assistant=OwnDenseAssistant(new_backbone(config,963717),pointing_rank=16,reward_hidden=8,router_hidden=8,detector_hidden=8,word_temporal=8,word_expansion=1).eval()
    model=DenseNativeView(assistant).eval()
    # This unit fixture is a small architecture mechanics test, not a public
    # training example or evidence of full-size model quality.
    prefix=[1,*tokenizer.encode('User: Complete the code.\nAssistant:\ndef f(x):\n').ids]
    hint=tokenizer.encode('def f(x):\n').ids
    # Tokenize prompt and reference prefix separately, as the real pack does.
    prefix=[1,*tokenizer.encode('User: Complete the code.\nAssistant:').ids,*hint]
    before={n:p.detach().clone() for n,p in model.named_parameters()};rng=torch.get_rng_state().clone()
    first=sample(model,tokenizer,prefix,hint,horizon=6,sampler='full_vocabulary_temperature1',seed=17)
    second=sample(model,tokenizer,prefix,hint,horizon=6,sampler='full_vocabulary_temperature1',seed=17)
    assert first==second and len(first['actual_generated_action_ids'])>=1
    assert first['complete_candidate_response']==tokenizer.decode(hint+first['actual_generated_action_ids'])
    assert first['cached_full_max_absolute_difference']<1e-4
    assert all(torch.equal(p,before[n]) and p.grad is None for n,p in model.named_parameters())
    assert torch.equal(rng,torch.get_rng_state())
    with pytest.raises(ValueError):sample(model,tokenizer,prefix,hint,horizon=256,sampler='greedy',seed=17)
    with pytest.raises(ValueError):sample(model,tokenizer,prefix,[3,4],horizon=6,sampler='greedy',seed=17)
