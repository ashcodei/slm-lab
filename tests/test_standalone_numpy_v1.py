import json
from pathlib import Path

from slm_lab.standalone_numpy_v1 import prepare_numpy, numpy_runtime_paths
from slm_lab.standalone_program_v1 import run_bounded


def test_numpy_stdio_vector_and_linear_algebra_without_worker_threads(tmp_path):
    code = b'import numpy as np,sys,json\na=np.array(list(map(int,sys.stdin.read().split())))\nprint(json.dumps([int(a.sum()),float(np.linalg.det(np.eye(3)))]))\n'
    result = run_bounded(prepare_numpy(code, tmp_path), b'1 2 3\n', python=True)
    assert result['status'] == 'completed', result
    assert json.loads(result['stdout']) == [6, 1.0]


def test_numpy_grant_does_not_expose_other_packages_or_allow_network(tmp_path):
    site = numpy_runtime_paths()[0].parent
    torch_file = site / 'torch/__init__.py'
    assert torch_file.exists()
    code = ('import numpy as np,socket,json\nblocked=[]\n'
            'for f in [lambda:open(' + repr(str(torch_file)) + ').read(),lambda:socket.socket()]:\n'
            ' try:f();blocked.append(False)\n'
            ' except OSError:blocked.append(True)\n'
            'print(json.dumps(blocked))\n').encode()
    result = run_bounded(prepare_numpy(code, tmp_path), b'', python=True)
    assert result['status'] == 'completed', result
    assert json.loads(result['stdout']) == [True, True]
