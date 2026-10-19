"""Compile/run native C++ contracts in fail-closed Landlock/seccomp processes."""
import hashlib
import json
import os
from pathlib import Path
import select
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time

from .cpp_tasks import HEADERS,RETURNS,SIGNATURES,harness

ROOT=Path(__file__).resolve().parents[2]
LAUNCHER=ROOT/'scripts/native_sandbox.py'
READ=(1<<2)|(1<<3);EXEC=1;ALL_FS=(1<<15)-1


class NativeBackendError(RuntimeError):pass


def toolchain():
    path=ROOT/'.tools/native-toolchain.json'
    if not path.exists():raise NativeBackendError('Verified native toolchain is not installed')
    data=json.loads(path.read_text())
    compiler=Path(data['compiler_path'])
    if hashlib.sha256(compiler.read_bytes()).hexdigest()!=data['compiler_sha256']:
        raise NativeBackendError('Native compiler hash changed')
    if not Path(data.get('cache_path','missing')).exists():raise NativeBackendError('Trusted compiler cache is not warmed')
    return data


def policy(mode,command,work,compiler_root=None):
    work=Path(work).resolve();paths=[(str(work),READ if mode=='runtime' else ALL_FS&~EXEC)]
    if mode=='runtime':paths.append((str(Path(command[0]).resolve()),EXEC|(1<<2)))
    else:
        paths.append((str(Path(compiler_root).resolve()),READ|EXEC))
        for path in ['/lib/x86_64-linux-gnu','/usr/lib/x86_64-linux-gnu','/lib64','/etc/ld.so.cache']:
            if Path(path).exists():paths.append((str(Path(path).resolve()),READ|EXEC))
    paths.append(('/dev/null',(1<<1)|(1<<2)))
    environment={'PATH':'/usr/bin:/bin','LANG':'C.UTF-8','TMPDIR':str(work)}
    if mode=='compile':
        environment.update(ZIG_GLOBAL_CACHE_DIR=str(work/'global-cache'),ZIG_LOCAL_CACHE_DIR=str(work/'local-cache'))
    return {'mode':mode,'command':list(map(str,command)),'cwd':str(work),'paths':paths,
            'environment':environment,'memory_bytes':3*1024**3 if mode=='compile' else 384*1024**2,
            'cpu_seconds':60 if mode=='compile' else 8,'file_bytes':128*1024**2 if mode=='compile' else 1024**2}


def launch(policy,**kwargs):
    return subprocess.Popen([sys.executable,'-I',str(LAUNCHER),json.dumps(policy)],
                            env={'PATH':'/usr/bin:/bin','LANG':'C.UTF-8'},close_fds=True,start_new_session=True,**kwargs)


def stop(process):
    try:os.killpg(process.pid,signal.SIGKILL)
    except ProcessLookupError:pass
    process.wait()


def compile_cpp(code,name,work,tools):
    if len(code)>50000:raise ValueError('Native code size limit exceeded')
    work=Path(work);(work/'contract.hpp').write_text(HEADERS+SIGNATURES[name]+';\n')
    (work/'candidate.cpp').write_text('#include "contract.hpp"\n'+code+'\n')
    (work/'harness.cpp').write_text(harness(name))
    shutil.copytree(tools['cache_path'],work/'global-cache')
    common=[tools['compiler_path'],'c++','-target','x86_64-linux-musl','-static','-std=c++17','-O2','-fno-lto']
    commands=[common+['-c','candidate.cpp','-o','candidate.o'],
              common+['candidate.o','harness.cpp','-o','program']]
    syntax=False
    for index,command in enumerate(commands):
        log=work/'compile.log'
        with log.open('wb') as output:
            process=launch(policy('compile',command,work,tools['toolchain_root']),stdout=output,stderr=output)
            try:returncode=process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                stop(process);return None,syntax,'Compiler deadline exceeded'
        diagnostic=log.read_bytes()[-4000:].decode(errors='replace')
        if 'SANDBOX_ERROR:' in diagnostic:raise NativeBackendError(diagnostic)
        if returncode:return None,syntax,diagnostic[-1200:]
        if index==0:syntax=True
    return work/'program',True,None


def input_line(args,repetitions=1):
    values=[ord(character) for character in args[0]] if isinstance(args[0],str) else args[0]
    return (' '.join(map(str,[len(values),*values,args[1] if len(args)>1 else 0,repetitions]))+'\n').encode()


def parse_output(text,name):
    text=text.strip();category=RETURNS.get(name,'int')
    if category=='optional' and text=='null':return None
    values=[int(value) for value in text.split()]
    if category=='vector':
        if not values or values[0]!=len(values)-1:raise ValueError('Malformed native vector response')
        return values[1:]
    if len(values)!=1:raise ValueError('Malformed native scalar response')
    if category=='bool':
        if values[0] not in {0,1}:raise ValueError('Malformed native boolean response')
        return bool(values[0])
    return values[0]


class Runner:
    def __init__(self,binary,work):
        self.stderr_path=Path(work)/'runtime.stderr';self.stderr=self.stderr_path.open('wb')
        self.process=launch(policy('runtime',[binary],work),stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.stderr)
        self.buffer=b'';self.total_output=0

    def close(self):
        if self.process.poll() is None:stop(self.process)
        else:self.process.wait()
        self.stderr.close()

    def invoke(self,args,name,repetitions=1):
        started=time.perf_counter();deadline=started+2
        request=input_line(args,repetitions)
        # Bound writes too: a candidate that never reads stdin cannot hang the
        # parent on a large test input.
        os.set_blocking(self.process.stdin.fileno(),False)
        sent=0
        while sent<len(request):
            remaining=deadline-time.perf_counter()
            if remaining<=0:raise TimeoutError('Native call deadline exceeded')
            _,ready,_=select.select([], [self.process.stdin.fileno()], [], remaining)
            if ready:
                try:sent+=os.write(self.process.stdin.fileno(),request[sent:])
                except BlockingIOError:pass
        while b'\n' not in self.buffer:
            remaining=deadline-time.perf_counter()
            if remaining<=0:raise TimeoutError('Native call deadline exceeded')
            ready,_,_=select.select([self.process.stdout.fileno()],[],[],remaining)
            if ready:
                block=os.read(self.process.stdout.fileno(),65536)
                if not block:
                    error=self.stderr_path.read_text(errors='replace')[-1000:]
                    if 'SANDBOX_ERROR:' in error:raise NativeBackendError(error)
                    raise RuntimeError('Native process exited: '+error)
                self.buffer+=block;self.total_output+=len(block)
                if len(self.buffer)>2*1024**2 or self.total_output>16*1024**2:raise ValueError('Native output budget exceeded')
        line,self.buffer=self.buffer.split(b'\n',1)
        elapsed=time.perf_counter()-started
        try:
            status=Path(f'/proc/{self.process.pid}/status').read_text()
            peak=int(next(value.split()[1] for value in status.splitlines() if value.startswith('VmHWM:')))*1024
        except (OSError,StopIteration):peak=None
        return parse_output(line.decode('ascii'),name),elapsed,peak


def execute(payload):
    tools=toolchain();name=payload['function']
    if name not in SIGNATURES:raise ValueError('Unknown native function contract')
    result={'syntax_valid':False,'functional_passed':False,'passed_cases':0,'total_cases':len(payload['cases']),
            'efficiency_completed':False,'stress':[],'compiler_version':tools['version'],
            'sandbox':'Landlock filesystem restrictions + seccomp; installed before exec/constructors'}
    with tempfile.TemporaryDirectory(prefix='slm-native-') as work:
        binary,syntax,error=compile_cpp(payload['code'],name,work,tools);result['syntax_valid']=syntax
        if binary is None:return {**result,'error':error}
        runner=Runner(binary,work)
        try:
            for case in payload['cases']:
                answer,_,_=runner.invoke(case['args'],name)
                if answer!=case['expected']:return {**result,'error':'Hidden case mismatch'}
                result['passed_cases']+=1
            result['functional_passed']=True
            for case in payload['stress']:
                timings=[];memory=0
                repetitions=10000 if name=='lower_bound' else 1
                for _ in range(3):
                    answer,elapsed,peak=runner.invoke(case['args'],name,repetitions)
                    if answer!=case['expected']:raise ValueError('Large-input result mismatch')
                    if peak is None:raise ValueError('Native peak-memory measurement unavailable')
                    timings.append(elapsed);memory=max(memory,peak)
                result['stress'].append({'input_size':len(case['args'][0]),'median_seconds':statistics.median(timings),
                    'memory_bytes':memory,'memory_measure':'Native process VmHWM, includes runtime/harness',
                    'repetitions':repetitions,'timing':'Trusted parent wall time including transport/input conversion'})
            result['efficiency_completed']=True
        except NativeBackendError:raise
        except (RuntimeError,ValueError,TimeoutError,BrokenPipeError) as error:result['error']=str(error)[:1000]
        finally:runner.close()
    return result


def efficiency(candidate,reference):
    if not candidate.get('functional_passed') or not candidate.get('efficiency_completed'):return False,{}
    first=candidate['stress'];second=reference['stress']
    if len(first)!=len(second) or not first:return False,{}
    ratios=[a['median_seconds']/max(b['median_seconds'],1e-6) for a,b in zip(first,second)]
    growth=first[-1]['median_seconds']/max(first[0]['median_seconds'],1e-6)
    reference_growth=second[-1]['median_seconds']/max(second[0]['median_seconds'],1e-6)
    passed=(first[-1]['median_seconds']<=max(.025,second[-1]['median_seconds']*30)
            and growth/max(reference_growth,1)<=2.5 and first[-1]['memory_bytes']<=64*1024**2)
    return bool(passed),{'relative_runtime':ratios,'observed_growth':growth,'reference_growth':reference_growth,
                         'protocol':'End-to-end native call timing; relative stress only, not an asymptotic proof.'}
