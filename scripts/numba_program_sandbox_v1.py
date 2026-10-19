"""Constrain a pinned Numba interpreter before its startup or target code.

Only the private temporary cache has filesystem write rights. Atomic cache
operations are permitted; processes, networking and chmod remain denied.
"""
import ctypes
import ctypes.util
import errno
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys


def main():
    spec = importlib.util.spec_from_file_location('trusted_native_sandbox', Path(__file__).with_name('native_sandbox.py'))
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    policy = json.loads(sys.argv[1])
    if policy['mode'] != 'runtime':
        raise ValueError('Python launcher accepts only the restrictive runtime mode')
    # The trusted outer launcher already restricts filesystem/syscall access.
    # Use the dynamic loader directly, avoiding ambient ldconfig subprocesses.
    sec = ctypes.CDLL('libseccomp.so.2', use_errno=True)
    sec.seccomp_init.argtypes = [ctypes.c_uint32]
    sec.seccomp_init.restype = ctypes.c_void_p
    sec.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    sec.seccomp_syscall_resolve_name.restype = ctypes.c_int
    sec.seccomp_rule_add_array.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint, ctypes.POINTER(native.Comparison)]
    sec.seccomp_rule_add_array.restype = ctypes.c_int
    sec.seccomp_load.argtypes = [ctypes.c_void_p]
    sec.seccomp_load.restype = ctypes.c_int
    sec.seccomp_release.argtypes = [ctypes.c_void_p]
    context = sec.seccomp_init(0x00050000 | errno.EPERM)
    if not context:
        raise RuntimeError('seccomp_init failed')
    try:
        os.chdir(policy['cwd'])
        for limit, value in [(resource.RLIMIT_CORE, 0), (resource.RLIMIT_AS, policy['memory_bytes']),
                             (resource.RLIMIT_CPU, policy['cpu_seconds']), (resource.RLIMIT_FSIZE, policy['file_bytes']),
                             (resource.RLIMIT_NOFILE, 32)]:
            resource.setrlimit(limit, (value, value))
        native.landlock(policy)
        names = ('read write readv writev close fstat stat lstat newfstatat statx lseek mmap mprotect munmap '
                 'brk mremap madvise rt_sigaction rt_sigprocmask rt_sigreturn sigaltstack '
                 'arch_prctl set_tid_address set_robust_list futex getrandom clock_gettime '
                 'gettimeofday nanosleep clock_nanosleep getpid gettid getuid geteuid getgid '
                 'getegid sched_yield sched_getaffinity exit exit_group open openat access '
                 'faccessat faccessat2 readlink readlinkat getcwd fcntl dup dup2 dup3 getrlimit '
                 'getrusage uname execve execveat pread64 getdents64 '
                 'mkdir mkdirat unlink unlinkat rename renameat renameat2 '
                 'fsync fdatasync ftruncate umask').split()
        for name in names + ['prlimit64']:
            number = sec.seccomp_syscall_resolve_name(name.encode())
            if number < 0:
                continue
            comparison = native.Comparison(0, 4, 0, 0) if name == 'prlimit64' else None
            result = sec.seccomp_rule_add_array(context, 0x7fff0000, number,
                0 if comparison is None else 1, None if comparison is None else ctypes.byref(comparison))
            if result < 0:
                raise RuntimeError('Cannot install Python syscall rule: ' + name)
        # This Python build uses FIOCLEX when opening its main script. Permit
        # only close-on-exec descriptor flags and the read-only terminal query.
        for request in [0x5451, 0x5450, 0x5401]:
            comparison = native.Comparison(1, 4, request, 0)
            result = sec.seccomp_rule_add_array(context, 0x7fff0000,
                sec.seccomp_syscall_resolve_name(b'ioctl'), 1, ctypes.byref(comparison))
            if result < 0:
                raise RuntimeError('Cannot install restricted Python descriptor rule')
        if sec.seccomp_load(context) < 0:
            raise RuntimeError('Cannot load Python syscall policy')
    finally:
        sec.seccomp_release(context)
    os.execve(policy['command'][0], policy['command'], policy['environment'])


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        print('SANDBOX_ERROR: ' + str(error), file=sys.stderr, flush=True)
        sys.exit(126)
