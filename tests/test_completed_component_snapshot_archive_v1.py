import importlib.util
from pathlib import Path

import pytest
import torch


spec = importlib.util.spec_from_file_location('component_archive', Path(__file__).parents[1] / 'scripts/archive_completed_component_snapshots_v1.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


def states():
    plan = {'steps': 3, 'corpus': 'original-source'}
    final = {'steps': 3, 'plan': plan, 'model': {'encoder': torch.tensor([1., 2.])}}
    snapshot = {**final, 'optimizer': {'retained': 'exact'}}
    return snapshot, final, plan


def test_exact_completed_state_is_accepted():
    module.assert_completed(*states(), 3)


@pytest.mark.parametrize('change', ['incomplete', 'changed_plan', 'changed_weight', 'missing_optimizer'])
def test_incomplete_or_changed_state_cannot_be_archived_as_completed(change):
    snapshot, final, plan = states()
    if change == 'incomplete': snapshot['steps'] = 2
    elif change == 'changed_plan': snapshot['plan'] = {'steps': 3, 'corpus': 'different-source'}
    elif change == 'changed_weight': snapshot['model'] = {'encoder': torch.tensor([1., 3.])}
    else: snapshot.pop('optimizer')
    with pytest.raises(ValueError): module.assert_completed(snapshot, final, plan, 3)
