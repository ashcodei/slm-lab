from pathlib import Path
import json

import pytest

from slm_lab.native_code import ROOT, NativeBackendError, toolchain
from slm_lab.standalone_program_v1 import compile_cpp_program, prepare_python, run_bounded


def test_complete_python_stdio_and_standard_library(tmp_path):
    code = b'import sys,math,collections,json\nprint(json.dumps([sum(map(int,sys.stdin.read().split())),math.isqrt(25),dict(collections.Counter("aab"))]))\n'
    result = run_bounded(prepare_python(code, tmp_path), b'1 2 3\n', python=True)
    assert result['status'] == 'completed', result
    assert json.loads(result['stdout']) == [6, 5, {'a': 2, 'b': 1}]


def test_python_cannot_read_private_files_write_network_fork_or_signal(tmp_path):
    private = tmp_path / 'private.txt'
    private.write_text('fixture-private-canary')
    work = tmp_path / 'work'
    work.mkdir()
    code = ('import os,socket,json,sys\n'
            'failures=[]\n'
            'checks=[lambda:open(' + repr(str(private)) + ').read(),'
            'lambda:open("new.txt","w"),lambda:socket.socket(),lambda:os.fork(),'
            'lambda:os.kill(os.getpid(),0),lambda:open("/proc/self/environ").read()]\n'
            'for i,check in enumerate(checks):\n'
            ' try:check();failures.append(i)\n'
            ' except OSError:pass\n'
            'print(json.dumps([failures,sys.flags.isolated,sys.flags.no_site]))\n').encode()
    result = run_bounded(prepare_python(code, work), b'', python=True)
    assert result['status'] == 'completed', result
    assert json.loads(result['stdout']) == [[], 1, 1]
    assert not (work / 'new.txt').exists()
    assert private.read_text() == 'fixture-private-canary'


@pytest.mark.parametrize('stream', ['stdout', 'stderr'])
def test_infinite_output_is_bounded_for_both_streams(tmp_path, stream):
    code = f'import sys\nwhile True:sys.{stream}.write("x"*65536);sys.{stream}.flush()\n'.encode()
    result = run_bounded(prepare_python(code, tmp_path), b'', python=True, output_bytes=32768)
    assert result['status'] == 'combined_output_budget_exceeded'
    assert len(result['stdout']) + len(result['stderr']) == 32768


def test_nonreading_program_cannot_block_parent_stdin_write(tmp_path):
    result = run_bounded(prepare_python(b'while True:pass\n', tmp_path), b'x' * 1024**2,
                         python=True, seconds=.5)
    assert result['status'] == 'wall_deadline_exceeded'
    assert result['input_bytes_sent'] < result['input_bytes']
    assert result['seconds'] < 3


def test_early_closed_stdin_does_not_claim_unwritten_bytes_were_sent(tmp_path):
    code = b'import os\nos.close(0)\nprint("done")\n'
    result = run_bounded(prepare_python(code, tmp_path), b'x' * 1024**2, python=True)
    assert result['status'] == 'completed' and result['stdout'] == b'done\n', result
    assert result['input_pipe_closed_early']
    assert result['input_bytes_sent'] < result['input_bytes']


def test_invalid_python_guard_never_executes_program(tmp_path):
    constraints = prepare_python(b'print("TARGET_EXECUTED")\n', tmp_path)
    constraints['paths'].append((str(tmp_path / 'missing-policy-path'), 4))
    with pytest.raises(NativeBackendError, match='SANDBOX_ERROR'):
        run_bounded(constraints, b'', python=True)


def test_closed_output_streams_wait_on_process_exit_without_false_timeout(tmp_path):
    code = b'import os,time\nos.close(1);os.close(2);time.sleep(.1)\n'
    result = run_bounded(prepare_python(code, tmp_path), b'', python=True, seconds=3)
    assert result['status'] == 'completed', result
    assert result['seconds'] < 2


def test_python_site_exit_helpers_without_loading_site_or_packages(tmp_path):
    code = b'import sys\nprint(sys.argv[0].endswith("program.py"),len(sys.argv),sys.flags.no_site)\nexit()\nraise RuntimeError("unreachable")\n'
    result = run_bounded(prepare_python(code, tmp_path), b'', python=True)
    assert result['status'] == 'completed', result
    assert result['stdout'] == b'True 1 1\n'


@pytest.mark.skipif(not (ROOT / '.tools/native-toolchain.json').exists(), reason='Verified compiler unavailable')
def test_complete_cpp_stdio_runs_without_function_harness(tmp_path):
    code = b'#include <bits/stdc++.h>\nint main(){long long x,s=0;while(std::cin>>x)s+=x;std::cout<<s<<"\\n";}\n'
    constraints, compiled = compile_cpp_program(code, tmp_path, toolchain())
    assert constraints is not None, compiled
    result = run_bounded(constraints, b'1 2 -3 9\n')
    assert result['status'] == 'completed', result
    assert result['stdout'] == b'9\n'


@pytest.mark.skipif(not (ROOT / '.tools/native-toolchain.json').exists(), reason='Verified compiler unavailable')
def test_standalone_cpp_constructor_is_confined_before_main(tmp_path):
    private = tmp_path / 'private.txt'
    private.write_text('fixture-private-canary')
    work = tmp_path / 'work'
    work.mkdir()
    code = ('#include <cstdio>\n#include <fcntl.h>\n#include <unistd.h>\n#include <sys/socket.h>\n#include <signal.h>\n'
            'static int failures=0;\n'
            '__attribute__((constructor)) static void probe(){\n'
            'if(open(' + json.dumps(str(private)) + ',O_RDONLY)>=0)failures|=1;\n'
            'if(socket(AF_INET,SOCK_STREAM,0)>=0)failures|=2;\n'
            'if(fork()>=0)failures|=4;\n'
            'if(kill(getpid(),0)==0)failures|=8;\n'
            '}\nint main(){printf("%d\\n",failures);}\n').encode()
    constraints, compiled = compile_cpp_program(code, work, toolchain())
    assert constraints is not None, compiled
    result = run_bounded(constraints, b'')
    assert result['status'] == 'completed' and result['stdout'] == b'0\n', result
