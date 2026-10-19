import json
from pathlib import Path

import pytest

from slm_lab.native_code import ROOT,execute,efficiency,launch,policy
from slm_lab.cpp_tasks import reference

pytestmark=pytest.mark.skipif(not (ROOT/'.tools/native-toolchain.json').exists(),reason='Native toolchain not installed')


def payload(code,cases=None):
    return {'code':code,'function':'sum_even','cases':cases or [{'args':[[]],'expected':0}], 'stress':[]}


def test_native_reference_executes_and_uses_external_measurements():
    result=execute(payload(reference('sum_even'),[{'args':[[1,2,-4,5,6]],'expected':4}]))
    assert result['syntax_valid'] and result['functional_passed']
    assert 'Landlock' in result['sandbox']


def test_constructor_cannot_access_host_files_network_processes_or_signals(tmp_path):
    private=tmp_path/'private.txt';private.write_text('fixture-private-canary')
    outside=tmp_path/'outside.txt'
    code='''#include <fcntl.h>
#include <sys/socket.h>
#include <unistd.h>
#include <signal.h>
static Int failures=0;
__attribute__((constructor)) static void probe(){
if(open(PRIVATE_PATH,O_RDONLY)>=0)failures|=1;
if(open(OUTSIDE_PATH,O_WRONLY|O_CREAT,0600)>=0)failures|=2;
if(socket(AF_INET,SOCK_STREAM,0)>=0)failures|=4;
if(fork()>=0)failures|=8;
if(kill(getpid(),0)==0)failures|=16;
if(open("/proc/self/environ",O_RDONLY)>=0)failures|=32;
}
Int sum_even(const Ints&){return failures;}
'''
    code='#define PRIVATE_PATH '+json.dumps(str(private))+'\n#define OUTSIDE_PATH '+json.dumps(str(outside))+'\n'+code
    result=execute(payload(code))
    assert result['functional_passed'],result
    assert private.read_text()=='fixture-private-canary' and not outside.exists()


def test_preprocessor_cannot_include_a_private_file(tmp_path):
    private=tmp_path/'private.hpp';private.write_text('#error fixture-private-content\n')
    result=execute(payload('#include '+json.dumps(str(private))+'\n'+reference('sum_even')))
    assert not result['syntax_valid'] and not result['functional_passed']
    assert 'fixture-private-content' not in result['error']


def test_infinite_native_code_is_terminated():
    result=execute(payload('Int sum_even(const Ints&){for(;;){} }'))
    assert result['syntax_valid'] and not result['functional_passed']
    assert 'deadline' in result['error'].lower()


def test_quadratic_native_implementation_fails_relative_stress():
    cases=[{'args':[[1,2,3,4]],'expected':6}]
    stress=[{'args':[list(range(n))],'expected':sum(range(0,n,2))} for n in [2048,8192,32768]]
    code='''Int sum_even(const Ints& xs){
volatile Int waste=0;
for(size_t i=0;i<xs.size();++i)for(size_t j=0;j<xs.size();++j)waste+=(i+j)&1;
Int answer=0;for(Int x:xs)if(x%2==0)answer+=x;return answer;
}'''
    bad=execute({'code':code,'function':'sum_even','cases':cases,'stress':stress})
    good=execute({'code':reference('sum_even'),'function':'sum_even','cases':cases,'stress':stress})
    assert bad['functional_passed'] and good['efficiency_completed']
    assert not efficiency(bad,good)[0]


def test_invalid_guard_policy_never_executes_target(tmp_path):
    import subprocess
    binary=ROOT/'.tools/native-warm/hello'
    constraints=policy('runtime',[str(binary)],tmp_path)
    constraints['paths'].append((str(tmp_path/'missing-path'),4))
    process=launch(constraints,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    out,error=process.communicate(timeout=10)
    assert process.returncode==126 and b'SANDBOX_ERROR:' in error
    assert not out
