import copy
import json
import os

import pytest

from slm_lab import complete_joint_owned_recipe_v1 as recipe
from slm_lab.full_experiment_launch_queue_v1 import identity


def original_queue(tmp_path):
    folder = tmp_path / 'queue'; folder.mkdir()
    owner = identity(os.getpid())
    (folder / 'plan.json').write_text(json.dumps(dict(owner=owner, declared_stages=list(recipe.STAGES))))
    return folder, owner


def test_actual_live_exact_original_queue_waits_without_inspecting_a_model_or_funding_a_GPU(tmp_path, monkeypatch):
    queue, owner = original_queue(tmp_path)
    def forbidden(*args): raise AssertionError('No new initializer inspection before the exact original full experiments finish')
    monkeypatch.setattr(recipe, 'completed_source', forbidden)
    assert recipe.prepare(tmp_path, queue=queue, expected_queue_owner=owner,
        initializer=tmp_path / 'not-created', measurement=tmp_path / 'not-created', runtime_sources={}) is None


@pytest.mark.parametrize('kind', ['different_birth', 'shortened_scope', 'terminal_without_complete'])
def test_changed_birth_shortened_original_scope_and_incomplete_predecessor_cannot_prepare_GPU_work(tmp_path, monkeypatch, kind):
    queue, owner = original_queue(tmp_path)
    if kind == 'different_birth': owner = dict(owner, ticks='0')
    elif kind == 'shortened_scope':
        path = queue / 'plan.json'; record = json.loads(path.read_text()); record['declared_stages'][-1]['scope'].pop()
        path.write_text(json.dumps(record))
    else: monkeypatch.setattr(recipe, 'terminal', lambda expected: True)
    with pytest.raises(ValueError): recipe.prepare(tmp_path, queue=queue, expected_queue_owner=owner,
        initializer=tmp_path / 'not-created', measurement=tmp_path / 'not-created', runtime_sources={})
