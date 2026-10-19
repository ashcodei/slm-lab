import json

import pytest

from slm_lab.corpus import file_hash
from slm_lab.pretraining_operational_recovery_v1 import math_for_runtime, validate_interruption


def test_runtime_owner_change_cannot_change_math_plan_or_import_unbound_math(tmp_path):
    path = tmp_path/'math.json'
    math = dict(owner=dict(pid=100, ticks='200'), planned_steps=4447,
                training_options=dict(maximum_microbatch_tokens=4096), source_schedule_sha256='full17families')
    path.write_text(json.dumps(math))
    runtime = dict(owner=dict(pid=101, ticks='201'), original_neural_plan=str(path),
                   original_neural_plan_sha256=file_hash(path), neural_options_unchanged=True)
    assert math_for_runtime(runtime) == math
    math['planned_steps'] = 20; path.write_text(json.dumps(math))
    with pytest.raises(ValueError, match='immutable'): math_for_runtime(runtime)


def test_dead_parent_is_insufficient_when_a_rank_lives_or_is_missing_from_inventory():
    parent = dict(pid=100, ticks='200'); child = dict(pid=101, ticks='201')
    receipt = dict(complete=True, all_old_registered_births_terminal=True, old_registered_births=[parent, child])
    validate_interruption(dict(owner=parent), [child], receipt, lambda row: False)
    with pytest.raises(ValueError, match='Every exact'):
        validate_interruption(dict(owner=parent), [child], receipt, lambda row: row == child)
    receipt['old_registered_births'] = [parent]
    with pytest.raises(ValueError, match='Every exact'):
        validate_interruption(dict(owner=parent), [child], receipt, lambda row: False)
