"""Real disposable process suspension, hard deadline and killed-parent rescue."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from slm_lab.bounded_waiter_pause_v1 import pause,state
from slm_lab.exact_waiting_process_retirement_v1 import observe,terminal


def identity(pid):
    fields=Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()
    return dict(pid=pid,ticks=fields[19])


def until(test,seconds=4):
    deadline=time.monotonic()+seconds
    while not test():
        if time.monotonic()>deadline:raise AssertionError('Expected actual process transition not observed')
        time.sleep(.01)


@pytest.fixture
def target():
    child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
    value=identity(child.pid)
    until(lambda:Path(f'/proc/{child.pid}/cmdline').read_bytes().startswith(sys.executable.encode()+b'\0-c\0'))
    actual=observe(value)
    try:yield child,actual
    finally:
        child.send_signal(signal.SIGCONT)
        child.terminate();child.wait()


def test_actual_waiter_is_restored_after_normal_and_exceptional_exit(target,tmp_path):
    child,value=target
    script='''import json
from pathlib import Path
from slm_lab.bounded_waiter_pause_v1 import pause,state
row=json.loads(ROW)
folder=Path(FOLDER)
with pause([row],folder/'normal',maximum_seconds=3):
    assert state(row) in {'T','t'}
assert state(row) in {'R','S'}
try:
    with pause([row],folder/'exception',maximum_seconds=3):
        assert state(row) in {'T','t'}
        raise RuntimeError('owned failure')
except RuntimeError:pass
else:raise AssertionError('Owned exception must propagate')
assert state(row) in {'R','S'}
'''.replace('ROW',repr(json.dumps(value))).replace('FOLDER',repr(str(tmp_path)))
    subprocess.run([sys.executable,'-c',script],env={**os.environ,'CUDA_VISIBLE_DEVICES':''},check=True,timeout=8)
    assert state(value) in {'R','S'}
    assert json.loads((tmp_path/'normal/rescue-complete.json').read_text())['complete']
    assert not terminal(value)


def test_hard_deadline_resumes_target_while_coordinator_is_still_alive(target,tmp_path):
    _,value=target
    script='''import json,time
from pathlib import Path
from slm_lab.bounded_waiter_pause_v1 import pause,state
row=json.loads(ROW)
with pause([row],Path(FOLDER)/'deadline',maximum_seconds=1):
    assert state(row) in {'T','t'}
    deadline=time.monotonic()+3
    while state(row) in {'T','t'}:
        if time.monotonic()>deadline:raise AssertionError('Hard deadline did not resume target')
        time.sleep(.01)
'''.replace('ROW',repr(json.dumps(value))).replace('FOLDER',repr(str(tmp_path)))
    subprocess.run([sys.executable,'-c',script],env={**os.environ,'CUDA_VISIBLE_DEVICES':''},check=True,timeout=8)
    receipt=json.loads((tmp_path/'deadline/rescue-complete.json').read_text())
    assert receipt['reason']=='hard_monotonic_deadline' and not terminal(value)


def test_separate_guard_resumes_original_birth_after_coordinator_is_killed(target,tmp_path):
    _,value=target
    folder=tmp_path/'killed'
    script=('import json,time; from pathlib import Path; from slm_lab.bounded_waiter_pause_v1 import pause; '
            'row=json.loads('+repr(json.dumps(value))+'); folder=Path('+repr(str(folder))+'); '
            'manager=pause([row],folder,maximum_seconds=5); manager.__enter__(); '
            '(folder/"entered").write_text("yes"); time.sleep(30)')
    owner=subprocess.Popen([sys.executable,'-c',script],env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
    try:
        until(lambda:(folder/'entered').exists())
        assert state(value) in {'T','t'}
        owner.kill();owner.wait()
        until(lambda:state(value) in {'R','S'})
        until(lambda:(folder/'rescue-complete.json').exists())
        receipt=json.loads((folder/'rescue-complete.json').read_text())
        assert receipt['reason']=='coordinator_exit_or_release_pipe' and receipt['complete']
        assert not terminal(value)
    finally:
        if owner.poll() is None:owner.kill();owner.wait()
