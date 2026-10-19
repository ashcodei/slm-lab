import json

import pytest

from slm_lab.native_code import ROOT
from slm_lab.standalone_gnu_v1 import compile_gnu_program, gnu_toolchain
from slm_lab.standalone_program_v1 import run_bounded

pytestmark = pytest.mark.skipif(not (ROOT / '.tools/gnu-toolchain-v1.json').exists(), reason='Isolated GNU toolchain not installed')


@pytest.fixture(scope='module')
def tools():
    return gnu_toolchain()


def test_real_gnu_bitset_and_gcd_extensions_with_complete_stdio(tmp_path, tools):
    code = b'#include <bits/stdc++.h>\nint main(){std::bitset<100> b; b.set(7);b.set(18); std::cout<<b._Find_first()<<" "<<b._Find_next(7)<<" "<<std::__gcd(18,24)<<"\\n";}\n'
    constraints, compilation = compile_gnu_program(code, tmp_path, tools)
    assert constraints is not None, compilation
    result = run_bounded(constraints, b'')
    assert result['status'] == 'completed' and result['stdout'] == b'7 18 6\n', result


def test_gnu_preprocessor_cannot_read_host_private_file(tmp_path, tools):
    secret = tmp_path / 'private.hpp'
    secret.write_text('#error fixture-private-canary\n')
    work = tmp_path / 'work'
    work.mkdir()
    source = ('#include ' + json.dumps(str(secret)) + '\nint main(){}\n').encode()
    constraints, result = compile_gnu_program(source, work, tools)
    assert constraints is None
    assert b'fixture-private-canary' not in result['stderr']


def test_gnu_static_constructor_is_confined_before_main(tmp_path, tools):
    secret = tmp_path / 'private.txt'
    secret.write_text('fixture-private-canary')
    work = tmp_path / 'work'
    work.mkdir()
    code = ('#include <cstdio>\n#include <fcntl.h>\n#include <unistd.h>\n#include <sys/socket.h>\n#include <signal.h>\n'
            'static int f=0;\n__attribute__((constructor)) static void probe(){\n'
            'if(open(' + json.dumps(str(secret)) + ',O_RDONLY)>=0)f|=1;\n'
            'if(socket(AF_INET,SOCK_STREAM,0)>=0)f|=2;\n'
            'if(fork()>=0)f|=4;\nif(kill(getpid(),0)==0)f|=8;\n'
            'if(open("/proc/self/environ",O_RDONLY)>=0)f|=16;\n'
            '}\nint main(){printf("%d\\n",f);}\n').encode()
    constraints, compilation = compile_gnu_program(code, work, tools)
    assert constraints is not None, compilation
    result = run_bounded(constraints, b'')
    assert result['status'] == 'completed' and result['stdout'] == b'0\n', result


def test_gnu_infinite_program_stops_at_parent_deadline(tmp_path, tools):
    constraints, compilation = compile_gnu_program(b'int main(){for(;;){} }\n', tmp_path, tools)
    assert constraints is not None, compilation
    result = run_bounded(constraints, b'', seconds=.5)
    assert result['status'] == 'wall_deadline_exceeded' and result['seconds'] < 3


def test_gnu_large_static_array_fits_the_fixed_standalone_address_space_cap(tmp_path, tools):
    code = b'#include <cstdio>\nvolatile unsigned char a[1024ULL*1024*1024];\nint main(){a[0]=7;a[sizeof(a)-1]=9;printf("%d\\n",int(a[0]+a[sizeof(a)-1]));}\n'
    constraints, compilation = compile_gnu_program(code, tmp_path, tools)
    assert constraints is not None, compilation
    assert constraints['memory_bytes'] == 1536 * 1024**2
    result = run_bounded(constraints, b'')
    assert result['status'] == 'completed' and result['stdout'] == b'16\n', result
