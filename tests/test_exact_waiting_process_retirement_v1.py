import os
from pathlib import Path
import subprocess
import sys

import pytest

from slm_lab.exact_waiting_process_retirement_v1 import observe,retire,terminal


def identity(process):
    return dict(pid=process.pid,ticks=(Path('/proc')/str(process.pid)/'stat').read_text().split(') ',1)[1].split()[19])


def test_real_hidden_process_birth_checks_and_pidfd_retirement_preserve_other_process():
    processes=[subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],env={**os.environ,'CUDA_VISIBLE_DEVICES':''}) for _ in range(2)]
    try:
        a,b=map(identity,processes)
        with pytest.raises(ValueError,match='birth'):observe(dict(a,ticks='0'))
        with pytest.raises(ValueError,match='exact hidden'):observe(a,'not-the-actual-command')
        row=observe(a);retire([row]);assert terminal(a)
        assert processes[0].wait(timeout=3)==-15
        assert observe(b)['pid']==processes[1].pid and processes[1].poll() is None
    finally:
        for p in processes:
            if p.poll() is None:p.terminate()
            p.wait(timeout=3)
