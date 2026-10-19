import copy
import json
from pathlib import Path
import shutil
import sys

import pytest

from slm_lab import metadata_observer_recovery_v1 as recovery
from slm_lab import full_experiment_launch_queue_v1 as original
from slm_lab import full_experiment_launch_queue_v2 as queue
from test_full_experiment_launch_queue_v1 import fixture as handoff_fixture

ROOT = Path(__file__).resolve().parents[1]


def old_copy(tmp_path):
    source = ROOT / 'reports/full-owned-experiment-launch-queue-v1'
    if not source.is_dir():
        pytest.skip('The actual closed metadata-failure evidence is local and excluded from Git')
    folder = tmp_path / 'retired'; folder.mkdir()
    for name in ['plan.json', 'progress.json']:
        shutil.copyfile(source / name, folder / name)
    log = tmp_path / 'retired.log'; shutil.copyfile(source.with_suffix('.log'), log)
    return folder, log


def test_actual_closed_original_failure_and_all_source_bytes_are_read_only_and_complete(tmp_path):
    folder, log = old_copy(tmp_path)
    before = {p.name: recovery.sha(p) for p in [folder / 'plan.json', folder / 'progress.json', log]}
    result = recovery.capture(folder, log, kind='serial')
    assert result['terminal_verified'] and result['original_metadata_failure_preserved']
    assert result['original_owner_not_relabelled_successful'] and not result['neural_retry_or_training_admission']
    assert before == {p.name: recovery.sha(p) for p in [folder / 'plan.json', folder / 'progress.json', log]}


@pytest.mark.parametrize('kind', ['live', 'launched', 'unrelated_failure'])
def test_live_owned_neural_or_unrelated_failure_cannot_be_adopted_as_metadata_recovery(tmp_path, monkeypatch, kind):
    folder, log = old_copy(tmp_path)
    if kind == 'live':
        monkeypatch.setattr(recovery, 'terminal', lambda expected: False)
    elif kind == 'launched':
        (folder / 'visual-launched.json').write_text('{}')
    else:
        log.write_text('ValueError: an actual failed neural stage\n')
    with pytest.raises(ValueError):
        recovery.capture(folder, log, kind='serial')


def test_all_four_actual_CPU_handoffs_preserve_original_order_scopes_and_restore_module_bindings(tmp_path, monkeypatch):
    captures, directory, publish, _ = handoff_fixture(tmp_path)
    previous = original.STAGES
    # These are explicitly metadata-only fixture drivers; no GPU is queried
    # or model constructed. Only the native metadata location/selector moves.
    indexed = tmp_path / 'recovered-native'; source = tmp_path / 'reports' / previous[3]['preparation']
    shutil.copytree(source, indexed)
    specs = (*queue.STAGES[:3], dict(queue.STAGES[3], preparation=str(indexed)))
    monkeypatch.setattr(queue, 'STAGES', specs)
    shutil.copyfile(tmp_path / 'scripts' / previous[3]['driver'], tmp_path / 'scripts' / specs[3]['driver'])
    result = queue.run(tmp_path, directory, captures, guard=lambda: None, publish=publish,
        resources=lambda *args: dict(ready=True), pause=lambda seconds: __import__('time').sleep(.01))
    assert result['complete'] and (tmp_path / 'actual-child-order').read_text().splitlines() == [r['name'] for r in previous]
    assert original.STAGES is previous
    assert all(a['scope'] == b['scope'] and a['output'] == b['output'] for a, b in zip(previous, specs))


def test_exception_restores_unchanged_original_stage_binding(tmp_path, monkeypatch):
    captures, directory, publish, _ = handoff_fixture(tmp_path, fail='visual')
    previous = original.STAGES
    with pytest.raises(ValueError, match='owned driver failed'):
        queue.run(tmp_path, directory, captures, guard=lambda: None, publish=publish,
            resources=lambda *args: dict(ready=True), pause=lambda seconds: __import__('time').sleep(.01))
    assert original.STAGES is previous


@pytest.mark.parametrize('failure_at', [None, 'configure', 'parent'])
def test_native_parent_adapter_restores_recipe_and_admission_even_after_failure(monkeypatch, failure_at):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    import train_native_code_suffix_aligned_GPU_v4 as driver
    import train_native_code_suffix_aligned_GPU_v5 as adapter
    from slm_lab import native_suffix_owned_runtime_contract_v4 as contract
    from slm_lab.native_suffix_owned_runtime_contract_v5 import require_complete_preparation
    previous_recipe, previous_admission = driver.PREP, contract.require_complete_preparation
    original_neural_worker = driver.distributed
    actual_calls = []
    def call(name):
        assert driver.PREP == recovery.INDEXED_OUT
        assert contract.require_complete_preparation is require_complete_preparation
        assert driver.distributed is original_neural_worker
        actual_calls.append(name)
        if name == failure_at:
            raise ValueError('controlled parent adapter failure')
    monkeypatch.setattr(driver, 'configure', lambda: call('configure'))
    monkeypatch.setattr(driver, 'parent', lambda: call('parent'))
    if failure_at is None:
        adapter.main()
    else:
        with pytest.raises(ValueError, match='controlled parent adapter failure'):
            adapter.main()
    assert actual_calls == (['configure'] if failure_at == 'configure' else ['configure', 'parent'])
    assert driver.PREP == previous_recipe and contract.require_complete_preparation is previous_admission
    assert 'torch' not in sys.modules
