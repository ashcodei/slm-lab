from pathlib import Path
import torch

from slm_lab.execution_group_controlled_v2 import ControlledExecutionGroup
from slm_lab.execution_group_capacity_v1 import maximum_prompt,stress
from slm_lab.model import ModelConfig
from slm_lab.own_dense_growth_v1 import OwnDenseAssistant,new_backbone
from slm_lab.own_dense_actor_v1 import OwnDenseActor

ROOT=Path(__file__).resolve().parents[1]


def test_full_fixed_horizon_uses_actual_kernel_without_optimizer_or_source_mutation():
    torch.set_num_threads(1)
    config=ModelConfig(vocab_size=32768,hidden_size=32,intermediate_size=64,num_layers=2,num_heads=4,num_kv_heads=2,
        max_text_length=4096,vision_hidden_size=8,vision_intermediate_size=16,vision_layers=1,vision_heads=2,
        image_size=16,patch_size=4,pool_size=2)
    base=OwnDenseAssistant(new_backbone(config,957319),reward_hidden=8,word_temporal=4,pointing_rank=8,router_hidden=8,detector_hidden=4)
    model=OwnDenseActor(base,rank=2,seed=957319)
    optimizer=torch.optim.AdamW([p for _,p in model.named_selected(('actor',))],lr=.001,foreach=False)
    stage=ControlledExecutionGroup(model,optimizer,ROOT,method='execution_hybrid',group_size=2,max_new_tokens=3)
    longest=maximum_prompt(stage)
    assert longest['maximum_prompt_and_full_horizon_fit']
    original_step=optimizer.step;position=stage.order[0];report=stress(stage,position)
    assert report['complete_global_fixed_actions']==6
    assert report['no_optimizer_commit'] and report['unchanged_weights_metadata_Adam_private_RNG_and_source_cursor']
    assert stage.completed==0 and not stage.metrics and not optimizer.state and optimizer.step==original_step
