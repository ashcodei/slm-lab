"""Confine Go before runtime startup, including init functions and threads."""
import ctypes
import ctypes.util
import errno
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys

spec = importlib.util.spec_from_file_location('owned_go_landlock', Path(__file__).with_name('native_sandbox.py'))
native = importlib.util.module_from_spec(spec); spec.loader.exec_module(native)


def runtime_seccomp(library):
    sec = library
    sec.seccomp_init.argtypes = [ctypes.c_uint32]; sec.seccomp_init.restype = ctypes.c_void_p
    sec.seccomp_release.argtypes = [ctypes.c_void_p]
    sec.seccomp_load.argtypes = [ctypes.c_void_p]; sec.seccomp_load.restype = ctypes.c_int
    sec.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]; sec.seccomp_syscall_resolve_name.restype = ctypes.c_int
    sec.seccomp_rule_add_array.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint, ctypes.POINTER(native.Comparison)]
    sec.seccomp_rule_add_array.restype = ctypes.c_int
    allow = 0x7fff0000; context = sec.seccomp_init(0x00050000 | errno.EPERM)
    if not context: raise RuntimeError('Cannot create Go runtime seccomp policy')

    def rule(name, action=allow, comparison=None):
        number = sec.seccomp_syscall_resolve_name(name.encode())
        if number < 0: return
        result = sec.seccomp_rule_add_array(context, action, number, 0 if comparison is None else 1,
            None if comparison is None else ctypes.byref(comparison))
        if result < 0: raise RuntimeError('Cannot constrain Go syscall: ' + name)

    try:
        for name in ('read write readv writev close fstat newfstatat statx lseek mmap mprotect munmap brk mremap madvise '
            'rt_sigaction rt_sigprocmask rt_sigreturn sigaltstack arch_prctl set_tid_address set_robust_list futex '
            'getrandom clock_gettime gettimeofday nanosleep clock_nanosleep getpid getppid gettid getuid geteuid '
            'getgid getegid sched_yield sched_getaffinity exit exit_group open openat openat2 access faccessat '
            'faccessat2 readlink readlinkat getcwd fcntl dup dup2 dup3 getrlimit getrusage uname execve execveat '
            'pread64 getdents64 statfs fstatfs epoll_create epoll_create1 epoll_ctl epoll_wait epoll_pwait '
            'epoll_pwait2 poll ppoll pipe pipe2 eventfd eventfd2 ioctl rseq membarrier ftruncate mkdir mkdirat '
            'unlink unlinkat rename renameat renameat2 fsync fdatasync umask').split(): rule(name)
        rule('prlimit64', comparison=native.Comparison(0, 4, 0, 0))
        rule('clone3', 0x00050000 | errno.ENOSYS)
        rule('clone', comparison=native.Comparison(0, 7, 0x10000, 0x10000))
        # Go asynchronous preemption signals its own runtime threads. Kernel
        # validation ensures the target TID belongs to this exact TGID.
        rule('tgkill', comparison=native.Comparison(0, 4, os.getpid(), 0))
        if sec.seccomp_load(context) < 0: raise RuntimeError('Cannot enforce Go runtime seccomp policy')
    finally: sec.seccomp_release(context)


def main():
    policy = json.loads(sys.argv[1]); os.chdir(policy['cwd'])
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_AS, (policy['address_bytes'], policy['address_bytes']))
    resource.setrlimit(resource.RLIMIT_CPU, (policy['cpu_seconds'], policy['cpu_seconds']))
    resource.setrlimit(resource.RLIMIT_FSIZE, (policy['file_bytes'], policy['file_bytes']))
    library = ctypes.util.find_library('seccomp')
    if not library: raise RuntimeError('libseccomp unavailable')
    native.SECCOMP = ctypes.CDLL(library, use_errno=True)
    native.landlock(policy)
    if policy['mode'] == 'compile': native.seccomp('compile')
    elif policy['mode'] == 'runtime': runtime_seccomp(native.SECCOMP)
    else: raise ValueError('Unknown Go confinement mode')
    os.execve(policy['command'][0], policy['command'], policy['environment'])


if __name__ == '__main__':
    try: main()
    except BaseException as error:
        print('SANDBOX_ERROR: ' + str(error), file=sys.stderr, flush=True); sys.exit(126)
