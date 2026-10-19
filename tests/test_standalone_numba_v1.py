import json

import pytest

from slm_lab.native_code import ROOT, NativeBackendError
from slm_lab.standalone_numba_v1 import numba_runtime, prepare_numba, run_numba

pytestmark = pytest.mark.skipif(not (ROOT / '.tools/numba-runtime-v1.json').exists(), reason='Private JIT stack absent')


@pytest.fixture(scope='module')
def runtime():
    return numba_runtime()


def test_real_cached_jit_function_and_expected_private_versions(tmp_path, runtime):
    code = b'import numpy as np,numba,json,sys\n@numba.njit(cache=True)\ndef total(a):\n return a.sum()\na=np.array(list(map(int,sys.stdin.read().split())),np.int64)\nprint(json.dumps([int(total(a)),np.__version__,numba.__version__]))\n'
    constraints = prepare_numba(code, tmp_path, runtime)
    result = run_numba(constraints, b'1 2 3\n')
    assert result['status'] == 'completed', result
    assert json.loads(result['stdout']) == [6, '1.26.4', '0.59.1']
    assert list((tmp_path / 'jit-cache').rglob('*.nbc'))
    again = run_numba(constraints, b'7 8\n')
    assert again['status'] == 'completed' and json.loads(again['stdout'])[0] == 15, again


def test_jit_profile_only_writes_cache_and_cannot_access_host_or_spawn(tmp_path, runtime):
    private = tmp_path / 'private.txt'
    private.write_text('fixture-private-canary')
    work = tmp_path / 'work'
    work.mkdir()
    code = ('import os,socket,json,numpy,numba\nblocked=[]\n'
        'for check in [lambda:open(' + repr(str(private)) + ').read(),lambda:open("outside-cache","w"),'
        'lambda:socket.socket(),lambda:os.fork(),lambda:os.kill(os.getpid(),0),'
        'lambda:os.chmod(' + repr(str(private)) + ',0)]:\n'
        ' try:check();blocked.append(False)\n'
        ' except OSError:blocked.append(True)\n'
        'print(json.dumps(blocked))\n').encode()
    result = run_numba(prepare_numba(code, work, runtime), b'')
    assert result['status'] == 'completed', result
    assert json.loads(result['stdout']) == [True] * 6
    assert private.read_text() == 'fixture-private-canary' and not (work / 'outside-cache').exists()


def test_jit_profile_deadline_and_nonreading_stdin(tmp_path, runtime):
    constraints = prepare_numba(b'while True:pass\n', tmp_path, runtime)
    result = run_numba(constraints, b'x' * 1024**2, seconds=.75)
    assert result['status'] == 'wall_deadline_exceeded' and result['input_bytes_sent'] < result['input_bytes']


def test_invalid_inner_guard_does_not_execute_target(tmp_path, runtime):
    constraints = prepare_numba(b'print("TARGET_EXECUTED")\n', tmp_path, runtime)
    inner = json.loads(constraints['command'][-1])
    inner['paths'].append((str(tmp_path / 'missing-policy-path'), 4))
    constraints['command'][-1] = json.dumps(inner)
    with pytest.raises(NativeBackendError, match='SANDBOX_ERROR'):
        run_numba(constraints, b'')


@pytest.mark.parametrize('deleted', [False, True])
def test_total_cache_quota_includes_open_unlinked_files(tmp_path, runtime, deleted):
    code = ('import os,time\np=os.path.join(os.environ["NUMBA_CACHE_DIR"],"large")\nf=open(p,"wb")\n'
            + ('os.unlink(p)\n' if deleted else '')
            + 'while True:f.write(b"x"*65536);f.flush();time.sleep(.003)\n').encode()
    result = run_numba(prepare_numba(code, tmp_path, runtime), b'', cache_bytes=1024**2)
    assert result['status'] == 'private_cache_budget_exceeded', result
    assert result['private_cache_peak_bytes'] > result['private_cache_budget_bytes']


def test_output_flood_after_new_jit_bootstrap_is_still_bounded(tmp_path, runtime):
    code = b'import sys\nwhile True:sys.stdout.write("x"*65536);sys.stdout.flush()\n'
    result = run_numba(prepare_numba(code, tmp_path, runtime), b'', output_bytes=32768)
    assert result['status'] == 'combined_output_budget_exceeded'
    assert len(result['stdout']) + len(result['stderr']) == 32768


def test_closed_streams_do_not_hide_process_exit_or_create_false_deadline(tmp_path, runtime):
    code = b'import os,time\nos.close(1);os.close(2);time.sleep(.1)\n'
    result = run_numba(prepare_numba(code, tmp_path, runtime), b'', seconds=3)
    assert result['status'] == 'completed' and result['seconds'] < 2, result
