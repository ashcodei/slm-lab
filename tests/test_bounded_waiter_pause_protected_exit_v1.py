"""An actual protected predecessor exit must promptly release idle pollers."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from slm_lab.bounded_waiter_pause_v1 import state


def identity(pid):
    fields=Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()
    return dict(pid=pid,ticks=fields[19])


def until(test):
    deadline=time.monotonic()+5
    while not test():
        if time.monotonic()>deadline:raise AssertionError('Actual protected-exit rescue not observed')
        time.sleep(.01)


def test_actual_protected_process_exit_resumes_original_waiter_before_coordinator_exits(tmp_path):
    env={**os.environ,'CUDA_VISIBLE_DEVICES':''}
    target=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],env=env)
    protected=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],env=env)
    row=identity(target.pid);predecessor=identity(protected.pid)
    script='''import json,time
from pathlib import Path
from slm_lab.exact_waiting_process_retirement_v1 import observe
from slm_lab.bounded_waiter_pause_v1 import pause,state
row=json.loads(ROW);protected=json.loads(PROTECTED);folder=Path(FOLDER)
with pause([observe(row)],folder,maximum_seconds=4,release_on_exit=[protected]):
    (folder/'entered').write_text('yes')
    assert state(row) in {'T','t'}
    deadline=time.monotonic()+3
    while state(row) in {'T','t'}:
        if time.monotonic()>deadline:raise AssertionError('Protected exit did not release waiter')
        time.sleep(.01)
'''.replace('ROW',repr(json.dumps(row))).replace('PROTECTED',repr(json.dumps(predecessor))).replace('FOLDER',repr(str(tmp_path/'protected')))
    owner=subprocess.Popen([sys.executable,'-c',script],env=env)
    try:
        until(lambda:(tmp_path/'protected/entered').exists())
        assert state(row) in {'T','t'}
        protected.terminate();protected.wait()
        until(lambda:state(row) in {'R','S'})
        owner.wait(timeout=5);assert owner.returncode==0
        receipt=json.loads((tmp_path/'protected/rescue-complete.json').read_text())
        assert receipt['reason']=='protected_predecessor_exit' and receipt['complete']
    finally:
        for process in [owner,protected,target]:
            if process.poll() is None:
                process.send_signal(__import__('signal').SIGCONT);process.terminate();process.wait()
