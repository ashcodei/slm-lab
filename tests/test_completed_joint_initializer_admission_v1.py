import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from slm_lab.completed_joint_initializer_admission_v1 import completed_source
from slm_lab.full_experiment_launch_queue_v1 import identity


def write(path, data): path.write_text(json.dumps(data))


def test_actual_live_exact_CPU_owner_stays_waiting_even_if_a_completion_file_exists(tmp_path):
    write(tmp_path / 'plan.json', dict(owner=identity(os.getpid())))
    write(tmp_path / 'complete.json', dict(complete=True))
    assert completed_source(tmp_path) is None


@pytest.mark.parametrize('kind', ['missing', 'failed', 'small_CPU_control'])
def test_dead_incomplete_failed_and_small_sources_cannot_authorize_full_initializer(tmp_path, kind):
    write(tmp_path / 'plan.json', dict(owner={'pid': 99999970, 'ticks': '1'}))
    if kind == 'failed': write(tmp_path / 'failure.json', {'error': 'actual retained failure'})
    if kind == 'small_CPU_control':
        write(tmp_path / 'complete.json', dict(format='own-completed-human-DPO-and-own-foundation-joint-initializer-v1',
            complete=True, actual_counts={'total_unique': 1090146}, actual_full_size_CPU_initializer=False))
        for name in ['source-admission.json', 'actual-completed-human-state-readback.json']:write(tmp_path / name, {})
    with pytest.raises(ValueError): completed_source(tmp_path)


def test_admission_reader_import_never_imports_torch_or_initializes_a_GPU():
    result = subprocess.run([sys.executable, '-c',
        "import sys;import slm_lab.completed_joint_initializer_admission_v1;assert 'torch' not in sys.modules;print('non-neural admission verified')"],
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 0 and result.stdout.strip() == 'non-neural admission verified'
