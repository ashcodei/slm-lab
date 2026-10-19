"""Exact predecessor birth and terminal physical-evidence admission."""
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    spec = importlib.util.spec_from_file_location('phase_snapshot_queue', ROOT/'scripts/archive_inactive_phase_snapshots_CPU_v1.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def save(path, value): path.write_text(json.dumps(value))


def test_actual_live_process_birth_stays_waiting_even_if_a_complete_file_exists(tmp_path, monkeypatch):
    module = load(monkeypatch); owner = dict(pid=os.getpid(), ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19])
    save(tmp_path/'plan.json', dict(owner=owner)); save(tmp_path/'complete.json', dict(complete=True))
    assert module.predecessor(tmp_path, owner) is None


def test_missing_process_requires_exact_complete_physical_receipt(tmp_path, monkeypatch):
    module = load(monkeypatch); owner = dict(pid=99999975, ticks='1'); save(tmp_path/'plan.json', dict(owner=owner))
    with pytest.raises(ValueError, match='without a complete physical receipt'): module.predecessor(tmp_path, owner)
    save(tmp_path/'complete.json', dict(complete=True, owner=owner, actual_archive_manifest=dict(original_retired=True, physical_restore_checked=True)))
    with pytest.raises(ValueError, match='actual complete physical restore'): module.predecessor(tmp_path, owner)
    value=json.loads((tmp_path/'complete.json').read_text()); value['actual_archive_manifest']['physical_restore']=dict(independent_physical_file_rehash_passed=True)
    save(tmp_path/'complete.json',value)
    assert module.predecessor(tmp_path, owner)['terminal_complete_physical_archive']


def test_failed_predecessor_must_close_only_its_unadmitted_duplicate_and_keep_original(tmp_path, monkeypatch):
    module=load(monkeypatch);owner=dict(pid=99999975,ticks='1');save(tmp_path/'plan.json',dict(owner=owner))
    failure=dict(owner=owner,original_retained=True);save(tmp_path/'failure.json',failure)
    with pytest.raises(ValueError,match='retain its exact original'):module.predecessor(tmp_path,owner)
    failure['discarded_unadmitted_duplicate']=dict(complete_original_and_anchor_still_exact=True,only_this_new_unadmitted_duplicate_discarded=True)
    save(tmp_path/'failure.json',failure)
    assert module.predecessor(tmp_path,owner)['terminal_failed_attempt_preserved']
    with pytest.raises(ValueError,match='exact original archival owner'):module.predecessor(tmp_path,dict(pid=owner['pid'],ticks='2'))
