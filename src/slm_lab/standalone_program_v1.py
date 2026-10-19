"""Bounded standalone reference-program checks, separate from live evaluators."""
import os
import json
import hashlib
from pathlib import Path
import selectors
import shutil
import subprocess
import sys
import sysconfig
import time

from .native_code import ROOT, NativeBackendError, launch, policy, stop


PYTHON_LAUNCHER = ROOT / 'scripts/python_program_sandbox_v1.py'
PYTHON_BOOTSTRAP = ('import builtins,runpy,sys;'
    'builtins.exit=builtins.quit=sys.exit;'
    'program=sys.argv.pop(1);runpy.run_path(program,run_name="__main__")')
CPP_COMPAT_HEADERS = ('algorithm array atomic bitset cassert cctype cerrno cfenv cfloat chrono cinttypes '
    'climits clocale cmath complex condition_variable csetjmp csignal cstdarg cstddef cstdint cstdio '
    'cstdlib cstring ctime cuchar cwchar cwctype deque exception filesystem forward_list fstream '
    'functional future initializer_list iomanip ios iosfwd iostream istream iterator limits list '
    'locale map memory mutex new numeric optional ostream queue random ratio regex scoped_allocator '
    'set sstream stack stdexcept streambuf string string_view system_error thread tuple type_traits '
    'typeindex typeinfo unordered_map unordered_set utility valarray variant vector').split()
CPP_COMPAT = '#pragma once\n' + ''.join('#include <' + name + '>\n' for name in CPP_COMPAT_HEADERS)


def python_policy(work):
    work = Path(work).resolve()
    interpreter = Path(sys.executable).resolve()
    constraints = policy('runtime', [str(interpreter), '-I', '-S', '-B', '-c', PYTHON_BOOTSTRAP,
                                     str(work / 'program.py')], work)
    stdlib = Path(sysconfig.get_path('stdlib')).resolve()
    # Directory enumeration is needed for imports. Reading third-party package
    # contents is not granted, and site/user/custom startup is disabled by -I -S.
    constraints['paths'].append((str(stdlib), 1 << 3))
    for path in stdlib.iterdir():
        if path.name not in {'site-packages', '__pycache__'} and (path.is_dir() or path.is_file()):
            constraints['paths'].append((str(path.resolve()), (1 << 2) | (1 << 3)))
    for path in ['/lib/x86_64-linux-gnu', '/usr/lib/x86_64-linux-gnu', '/lib64', '/etc/ld.so.cache']:
        if Path(path).exists():
            constraints['paths'].append((str(Path(path).resolve()), 1 | (1 << 2) | (1 << 3)))
    constraints['paths'].append(('/dev/urandom', 1 << 2))
    return constraints


def python_launch(constraints, **kwargs):
    return subprocess.Popen([sys.executable, '-I', str(PYTHON_LAUNCHER), json.dumps(constraints)],
        env={'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8'}, close_fds=True, start_new_session=True, **kwargs)


def run_bounded(constraints, data, seconds=3, output_bytes=1024**2, python=False):
    """Bound stdin writes, combined stdout/stderr reads, wall time and cleanup."""
    if not isinstance(data, bytes) or len(data) > 4 * 1024**2 or not 0 < seconds <= 120 or not 0 < output_bytes <= 4 * 1024**2:
        raise ValueError('Standalone input/time/output budget exceeded')
    started = time.perf_counter()
    deadline = started + seconds
    process = (python_launch if python else launch)(constraints, stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    buffers = {'stdout': bytearray(), 'stderr': bytearray()}
    reason, sent, total, input_closed_early = None, 0, 0, False
    try:
        with selectors.DefaultSelector() as selector:
            for stream, name in [(process.stdout, 'stdout'), (process.stderr, 'stderr')]:
                os.set_blocking(stream.fileno(), False)
                selector.register(stream, selectors.EVENT_READ, name)
            os.set_blocking(process.stdin.fileno(), False)
            if data:
                selector.register(process.stdin, selectors.EVENT_WRITE, 'stdin')
            else:
                process.stdin.close()
            while selector.get_map() or process.poll() is None:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    reason = 'wall_deadline_exceeded'
                    break
                if not selector.get_map():
                    # EOF can precede process exit. An empty selector cannot
                    # wake when the child exits, so wait on the child itself.
                    try:
                        process.wait(timeout=remaining)
                    except subprocess.TimeoutExpired:
                        reason = 'wall_deadline_exceeded'
                    break
                for key, _ in selector.select(remaining):
                    if key.data == 'stdin':
                        try:
                            sent += os.write(key.fd, data[sent:sent + 65536])
                        except BlockingIOError:
                            continue
                        except BrokenPipeError:
                            input_closed_early = True
                        if sent == len(data) or input_closed_early:
                            selector.unregister(key.fileobj)
                            key.fileobj.close()
                    else:
                        try:
                            block = os.read(key.fd, 65536)
                        except BlockingIOError:
                            continue
                        if not block:
                            selector.unregister(key.fileobj)
                            key.fileobj.close()
                            continue
                        capacity = max(0, output_bytes - total)
                        buffers[key.data].extend(block[:capacity])
                        total += len(block)
                        if total > output_bytes:
                            reason = 'combined_output_budget_exceeded'
                            break
                if reason:
                    break
        if reason is None:
            reason = 'completed' if process.returncode == 0 else 'runtime_exit'
    finally:
        if process.poll() is None:
            stop(process)
        else:
            process.wait()
        for stream in [process.stdin, process.stdout, process.stderr]:
            if not stream.closed:
                stream.close()
    if process.returncode == 126 and b'SANDBOX_ERROR:' in buffers['stderr']:
        raise NativeBackendError(bytes(buffers['stderr']).decode(errors='replace'))
    return {'status': reason, 'returncode': process.returncode, 'stdout': bytes(buffers['stdout']),
            'stderr': bytes(buffers['stderr']), 'seconds': time.perf_counter() - started,
            'input_bytes': len(data), 'input_bytes_sent': sent,
            'input_pipe_closed_early': input_closed_early, 'combined_output_bytes_observed': total}


def prepare_python(code, work):
    if not isinstance(code, bytes) or not code or len(code) > 32768 or b'\0' in code:
        raise ValueError('Original Python program size/encoding invalid')
    (Path(work) / 'program.py').write_bytes(code)
    return python_policy(work)


def compile_cpp_program(code, work, tools, seconds=90):
    """Compile an unmodified standalone source with the existing isolated toolchain.

    A recorded local bits/stdc++.h shim includes only standard C++ headers.
    This is ISO C++17/libc++, so other GNU-specific features may be unsupported.
    Compilation failure is a runtime/toolchain qualification, not a wrong-code label.
    """
    if not isinstance(code, bytes) or not code or len(code) > 32768 or b'\0' in code:
        raise ValueError('Original C++ program size/encoding invalid')
    work = Path(work)
    (work / 'candidate.cpp').write_bytes(code)
    compat = work / 'compat' / 'bits'
    compat.mkdir(parents=True)
    (compat / 'stdc++.h').write_text(CPP_COMPAT)
    shutil.copytree(tools['cache_path'], work / 'global-cache')
    command = [tools['compiler_path'], 'c++', '-target', 'x86_64-linux-musl', '-static',
               '-std=c++17', '-O2', '-fno-lto', '-I', str(work / 'compat'), 'candidate.cpp', '-o', 'program']
    constraints = policy('compile', command, work, tools['toolchain_root'])
    result = run_bounded(constraints, b'', seconds=seconds)
    result.update(compiler_mode='ISO C++17/libc++/musl/static, original source bytes',
                  bits_compat_header_sha256=hashlib.sha256(CPP_COMPAT.encode()).hexdigest())
    if result['status'] != 'completed':
        return None, result
    binary = work / 'program'
    if not binary.is_file():
        raise NativeBackendError('Successful compiler did not produce the expected binary')
    return policy('runtime', [binary], work), result
