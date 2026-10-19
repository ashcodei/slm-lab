"""Apply filesystem/process/network limits before Rust compilation or native tests."""
import ctypes
import ctypes.util
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys

spec = importlib.util.spec_from_file_location('rust_thread_policy', Path(__file__).with_name('javascript_sandbox.py'))
threads = importlib.util.module_from_spec(spec); spec.loader.exec_module(threads)


def main():
    policy = json.loads(sys.argv[1]); os.chdir(policy['cwd'])
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    for kind, value in [(resource.RLIMIT_AS, policy['memory_bytes']),
                        (resource.RLIMIT_CPU, policy['cpu_seconds']), (resource.RLIMIT_FSIZE, policy['file_bytes'])]:
        resource.setrlimit(kind, (value, value))
    library = ctypes.util.find_library('seccomp')
    if not library: raise RuntimeError('libseccomp unavailable for Rust')
    threads.native.SECCOMP = ctypes.CDLL(library, use_errno=True)
    threads.native.landlock(policy)
    if policy['mode'] == 'compile': threads.native.seccomp('compile')
    elif policy['mode'] == 'runtime':
        original = ctypes.util.find_library
        ctypes.util.find_library = lambda name: library if name == 'seccomp' else original(name)
        threads.seccomp()
    else: raise ValueError('Unknown Rust confinement mode')
    os.execve(policy['command'][0], policy['command'], policy['environment'])


if __name__ == '__main__':
    try: main()
    except BaseException as error:
        print('SANDBOX_ERROR: ' + str(error), file=sys.stderr, flush=True); sys.exit(126)
