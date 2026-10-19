import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from handoff_public_billion_source_language_CPU_v2 import send_int
from slm_lab.exact_waiting_process_retirement_v1 import pidfd_open


def test_actual_child_pidfd_SIGINT_works_on_this_Python_and_kernel_without_numeric_PID_signalling():
    child = subprocess.Popen([sys.executable, '-u', '-c',
        "import signal,sys,time;signal.signal(signal.SIGINT,lambda *a:sys.exit(0));print('ready',flush=True);time.sleep(30)"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == 'ready'
        fd = pidfd_open(child.pid)
        try:
            send_int(fd)
        finally:
            os.close(fd)
        assert child.wait(timeout=5) == 0
    finally:
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=5)
