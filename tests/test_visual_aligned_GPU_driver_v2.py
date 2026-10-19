"""New launch/paired-scope controls; never actual GPU or neural-score evidence."""
import copy
import json
from types import SimpleNamespace

import pytest

import train_joint_visual_aligned_continuation_GPU_v2 as driver
from compare_visual_native_endpoints_v1 import paired_words
from slm_lab import joint_visual_runtime_contract_v1 as contract
from slm_lab import exact_waiting_process_retirement_v1 as retirement


def words():
    rows = [dict(source_position=p, id=str(p), group_id='fixture', family='fixture', expected='example',
        wrong_image_id=str((p + 1) % 3429), source_image_sha256='a' * 64, native_prompt_ids=[1, 5],
        conditions={c: dict(native=dict(generated_ids=[6, 2])) for c in ['proper', 'blank', 'wrong']}) for p in range(3429)]
    return dict(results=rows, generation_budget=64, complete_held_source_words=True, held_source_gradients=False,
        overall=dict(synthetic_schema_control_not_a_neural_score=True))


def test_every_original_word_and_all_three_controls_are_paired_without_a_subset_gain_claim():
    before = words()
    after = copy.deepcopy(before)
    after['results'][9]['conditions']['proper']['native']['generated_ids'] = [7, 2]
    result = paired_words(before, after)
    assert result['original_words'] == 3429
    assert result['all_three_condition_native_sequences_identical'] == 3428
    assert result['same_original_sources_wrong_donors_and_full_native_budgets']
    assert not result['candidate_promoted']


@pytest.mark.parametrize('case', ['partial_words', 'missing_wrong_condition', 'wrong_donor', 'changed_prompt', 'wrong_native_budget'])
def test_changed_held_word_population_or_input_or_budget_cannot_be_called_a_matched_gain(case):
    before, after = words(), words()
    if case == 'partial_words':
        after['results'].pop()
    elif case == 'missing_wrong_condition':
        after['results'][9]['conditions'].pop('wrong')
    elif case == 'wrong_donor':
        after['results'][9]['wrong_image_id'] = 'another'
    elif case == 'changed_prompt':
        after['results'][9]['native_prompt_ids'] = [1, 9]
    else:
        after['generation_budget'] = 32
    with pytest.raises(ValueError):
        paired_words(before, after)


@pytest.mark.parametrize('case', ['idle_inherited', 'preparation_owner_live', 'not_enough_disk', 'missing_exact_predecessor'])
def test_launch_rejects_before_any_model_directory_or_GPU_lease_when_a_required_gate_fails(case, tmp_path, monkeypatch):
    prep = tmp_path / 'prep'
    prep.mkdir()
    (prep / 'ready.json').write_text(json.dumps(dict(exact_predecessors=dict(pipeline_owner='fixed'), required_free_bytes_before_launch=100)))
    (prep / 'plan.json').write_text(json.dumps(dict(owner=dict(pid=1, ticks='100'))))
    monkeypatch.setattr(driver, 'PREP', prep)
    monkeypatch.setattr(driver, 'OUT', tmp_path / 'GPU-output')
    monkeypatch.setattr(driver, 'DEST', tmp_path / 'GPU-checkpoints')
    monkeypatch.setattr(driver.os, 'sched_getscheduler', lambda pid: driver.os.SCHED_IDLE if case == 'idle_inherited' else driver.os.SCHED_OTHER)
    monkeypatch.setattr(driver.os, 'nice', lambda delta: 0)
    monkeypatch.setattr(retirement, 'terminal', lambda owner: case != 'preparation_owner_live')
    calls = []
    def ready(root, **predecessors):
        calls.append(predecessors)
        return case != 'missing_exact_predecessor'
    monkeypatch.setattr(contract, 'ready', ready)
    monkeypatch.setattr(driver.shutil, 'disk_usage', lambda path: SimpleNamespace(free=99))
    with pytest.raises(ValueError):
        driver.parent()
    assert not driver.OUT.exists() and not driver.DEST.exists()
    assert len(calls) == (0 if case in ['idle_inherited', 'preparation_owner_live'] else 1)
