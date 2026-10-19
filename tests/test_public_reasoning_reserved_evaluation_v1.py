"""Complete native reserved transport with an explicit empty-output test model."""
import json
from pathlib import Path
import shutil

import pytest
import torch

from slm_lab.codenet_token_pack_v1 import CompleteCodePack
from slm_lab.model import ModelConfig
from slm_lab.own_dense_actor_v1 import OwnDenseActor
from slm_lab.own_dense_growth_v1 import OwnDenseAssistant,new_backbone
from slm_lab.public_reasoning_reserved_evaluation_v1 import PublicReasoningReserved,evaluate
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest


@pytest.fixture(scope='module')
def population():
    return PublicReasoningReserved('data/public-reasoning-curriculum-v1',
        'reports/public-reasoning-curriculum-v1-readback-admission/complete.json')


def test_every_real_reserved_case_target_budget_and_unfiltered_termination(population):
    assert len(population.cases)==289 and len({row['task'] for row in population.cases})==12
    assert {row['split'] for row in population.rows}=={'validation'}
    for case in population.cases:
        scored=population.grade(case['position'],case['complete_answer_EOS_ids'])
        assert scored['correct'] and len(case['original_prompt_ids'])+case['native_generation_budget']<=4096
        assert not population.grade(case['position'],[0]+case['complete_answer_EOS_ids'])['correct']
    with pytest.raises(ValueError):CompleteCodePack(population.folder,'validation',purpose='training')
    with pytest.raises(ValueError):population.grade(0,[2,2])


def test_complete289_native_deliveries_are_observed_unchanged_without_any_training(population,tmp_path):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    config=ModelConfig(vocab_size=32768,hidden_size=8,intermediate_size=16,num_layers=1,num_heads=2,num_kv_heads=1,
        max_text_length=4096,vision_hidden_size=8,vision_intermediate_size=16,vision_layers=1,vision_heads=2,
        image_size=16,patch_size=4,pool_size=2)
    base=OwnDenseAssistant(new_backbone(config,983613),pointing_rank=4,reward_hidden=4,word_temporal=4,
        router_hidden=4,detector_hidden=4,word_expansion=1)
    # A deliberately initialized transport fixture. This tests complete native
    # delivery and strict empty-response scoring, never learned model quality.
    with torch.no_grad():
        base.network.embedding.weight.fill_(1.);base.network.embedding.weight[2].fill_(10.)
        for block in base.network.blocks:
            block.attn.o_proj.weight.zero_();block.down.weight.zero_()
    model=OwnDenseActor(base,rank=2).eval();sources=dict(test_fixture_sha256='e'*64)
    checkpoint=tmp_path/'actor.pt';torch.save(model.export(sources),checkpoint)
    shutil.copyfile(population.folder/'tokenizer.json',tmp_path/'tokenizer.json')
    before=tensor_digest(model.named_parameters())
    result=evaluate(model.policy,checkpoint,tmp_path/'evaluation',population)
    assert result['examples']==289 and result['correct']==0 and result['actual_CPU_threads']==1
    assert result['all_model_inputs_exclude_reference_answer_tokens']
    with (tmp_path/'evaluation/native-actions.jsonl').open() as stream:actions=list(map(json.loads,stream))
    with (tmp_path/'evaluation/actual-native-results.jsonl').open() as stream:rows=list(map(json.loads,stream))
    assert len(actions)==len(rows)==289
    for case,event,row in zip(population.cases,actions,rows):
        assert event['actual_generated_token_ids']==[2] and event['actual_stopped_on_EOS']
        assert event['original_input_token_ids']==case['original_prompt_ids']
        assert row['actual_generated_token_ids']==[2] and row['score']['response']=='' and not row['score']['correct']
        assert event['requested_native_action_budget']==case['native_generation_budget']
    assert tensor_digest(model.named_parameters())==before and all(parameter.grad is None for parameter in model.parameters())
