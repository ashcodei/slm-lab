import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import own_training_gpu_lease_v2 as owner
from slm_lab.corpus import file_hash


@pytest.fixture
def setup(tmp_path, monkeypatch):
    reports = tmp_path / 'reports'; reports.mkdir()
    receipt = tmp_path / 'legacy.json'; status = tmp_path / 'status.json'; status.write_text('{"status":"preserved failure"}')
    # The numeric PID exists, but this old birth does not. It must never receive
    # a stop/resume signal just because the original birth is terminal.
    old = dict(pid=os.getpid(), ticks='-1')
    receipt.write_text(json.dumps(dict(complete=True, original_waiter=old, original_waiter_terminal=True,
        legacy_status_file=str(status), legacy_status_sha256=file_hash(status))))
    lease = tmp_path / 'GPU.lock'; lease.touch()
    monkeypatch.setattr(owner, 'REPORTS', reports); monkeypatch.setattr(owner, 'RECEIPT', receipt)
    monkeypatch.setattr(owner, 'LEASE', lease); monkeypatch.setattr(owner, 'WAITER', old)
    plan = dict(owner=owner.resources.identity(os.getpid()), waiter=old,
        terminal_waiter_receipt_sha256=file_hash(receipt), source_files_sha256={str(receipt):file_hash(receipt)},
        expires_at_unix=time.time()+60)
    return reports, receipt, status, lease, plan


def test_actual_kernel_lock_and_exact_live_parent_required_no_dummy_waiter(setup):
    reports, receipt, status, lease, plan = setup
    with lease.open('a+') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert owner.lock_owner() == os.getpid()
        owner.idle_original(); owner.hold_original(plan); owner.owned(plan)
        bad = dict(plan, owner=dict(plan['owner'], ticks='wrong'))
        with pytest.raises(ValueError): owner.owned(bad)
    with pytest.raises(ValueError): owner.owned(plan)


def test_changed_preserved_failure_or_live_original_birth_rejects(setup, monkeypatch):
    reports, receipt, status, lease, plan = setup
    status.write_text('{"status":"changed"}')
    with pytest.raises(ValueError): owner.idle_original()
    actual = owner.resources.identity(os.getpid()); monkeypatch.setattr(owner, 'WAITER', actual)
    data = json.loads(receipt.read_text()); data.update(original_waiter=actual, legacy_status_sha256=file_hash(status))
    receipt.write_text(json.dumps(data))
    with pytest.raises(ValueError): owner.idle_original()


def test_pidfd_cleanup_stops_only_registered_exact_child_preserving_other_process(setup):
    reports, receipt, status, lease, plan = setup
    args = [sys.executable, '-c', 'import signal; signal.pause()']
    first = subprocess.Popen(args); other = subprocess.Popen(args)
    try:
        actor = owner.resources.identity(first.pid)
        owner.stop_registered(dict(owner.resources.identity(other.pid), ticks='not-the-current-birth'))
        assert other.poll() is None
        (reports/'children.json').write_text(json.dumps([dict(**actor,subject='owned-CPU-control')]))
        owner.restore_owned(plan, 'actual CPU cleanup control')
        assert first.wait(timeout=5) != 0 and other.poll() is None
        restored = json.loads((reports/'restored.json').read_text())
        assert restored['waiter'] is None and restored['original_waiter_not_signalled_or_restarted']
        assert restored['all_registered_point_actors_terminal']
    finally:
        for process in [first, other]:
            if process.poll() is None: process.terminate()
            process.wait(timeout=5)
