import json
from pathlib import Path

import pytest
import torch

from slm_lab.expanded_joint_sources_v1 import ExpandedJointSources
from slm_lab.model import ModelConfig
from slm_lab.native_training_text_projection_v1 import NativeTrainingTextProjection


@pytest.fixture(scope='module')
def data():
    torch.set_num_threads(1)
    root=Path(__file__).resolve().parents[1]
    plan=json.loads((root/'reports/phase-joint-v3/plan.json').read_text())
    raw=torch.load(plan['initial_dense_settings']['initializer'],map_location='cpu',weights_only=True,mmap=True)
    config=ModelConfig(**raw['config']);del raw
    return ExpandedJointSources(root,config)


def test_each_family_longest_projection_equals_actual_native_batch_without_padding_or_target_loss(data):
    projection=NativeTrainingTextProjection(data)
    assert len(data.populations)==28
    for family in data.populations:
        position=max(range(data.populations[family]),key=lambda p:data.lengths[family][p])
        ids,labels=projection.sequence(family,position)
        batch=data.batch(family,[position],'cpu');real=int(batch['attention_mask'][0].sum())
        assert ids==batch['input_ids'][0,:real].tolist()
        assert labels==batch['labels'][0,:real].tolist()
        assert len(ids)==data.lengths[family][position]


def test_held_unknown_and_out_of_range_inputs_are_not_accepted(data):
    projection=NativeTrainingTextProjection(data)
    with pytest.raises(ValueError):projection.sequence('held',0)
    with pytest.raises(ValueError):projection.sequence('human',data.populations['human'])
    with pytest.raises(ValueError):NativeTrainingTextProjection(object())
