"""Constrain Node/Jest before startup; allow threads, deny processes and networking."""
import ctypes
import ctypes.util
import errno
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys

spec = importlib.util.spec_from_file_location('owned_landlock_launcher', Path(__file__).with_name('native_sandbox.py'))
native = importlib.util.module_from_spec(spec); spec.loader.exec_module(native)


def seccomp():
    sec = ctypes.CDLL(ctypes.util.find_library('seccomp'), use_errno=True)
    sec.seccomp_init.argtypes = [ctypes.c_uint32]; sec.seccomp_init.restype = ctypes.c_void_p
    sec.seccomp_release.argtypes = [ctypes.c_void_p]
    sec.seccomp_load.argtypes = [ctypes.c_void_p]; sec.seccomp_load.restype = ctypes.c_int
    sec.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]; sec.seccomp_syscall_resolve_name.restype = ctypes.c_int
    sec.seccomp_rule_add_array.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint, ctypes.POINTER(native.Comparison)]
    sec.seccomp_rule_add_array.restype = ctypes.c_int
    allow = 0x7fff0000; deny = 0x00050000 | errno.EPERM
    context = sec.seccomp_init(deny)
    if not context: raise RuntimeError('Cannot create JavaScript seccomp rules')

    def rule(name, action=allow, comparison=None):
        number = sec.seccomp_syscall_resolve_name(name.encode())
        if number < 0: return
        code = sec.seccomp_rule_add_array(context, action, number, 0 if comparison is None else 1,
                    None if comparison is None else ctypes.byref(comparison))
        if code < 0: raise RuntimeError('Cannot constrain JavaScript syscall: ' + name)

    try:
        for name in ('read write readv writev close fstat newfstatat statx lseek mmap mprotect munmap '
                     'brk mremap madvise rt_sigaction rt_sigprocmask rt_sigreturn sigaltstack arch_prctl '
                     'set_tid_address set_robust_list futex getrandom clock_gettime gettimeofday '
                     'nanosleep clock_nanosleep getpid getppid gettid getuid geteuid getgid getegid '
                     'sched_yield sched_getaffinity exit exit_group open openat access faccessat '
                     'faccessat2 readlink readlinkat getcwd fcntl dup dup2 dup3 getrlimit getrusage uname '
                     'execve execveat pread64 getdents64 statfs fstatfs epoll_create epoll_create1 '
                     'epoll_ctl epoll_wait epoll_pwait epoll_pwait2 poll ppoll pipe pipe2 eventfd '
                     'eventfd2 ioctl rseq membarrier ftruncate mkdir mkdirat unlink unlinkat rename '
                     'renameat renameat2 chmod fchmod fchmodat fsync fdatasync umask').split(): rule(name)
        rule('prlimit64', comparison=native.Comparison(0, 4, 0, 0))
        # glibc falls back to clone when clone3 returns ENOSYS. A seccomp
        # filter cannot safely inspect clone3's flags behind its pointer.
        rule('clone3', 0x00050000 | errno.ENOSYS)
        rule('clone', comparison=native.Comparison(0, 7, 0x10000, 0x10000))
        code = sec.seccomp_load(context)
        if code < 0: raise RuntimeError('Cannot enforce JavaScript seccomp policy')
    finally: sec.seccomp_release(context)


def main():
    policy = json.loads(sys.argv[1]); os.chdir(policy['cwd'])
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    # V8 reserves address space beyond its actual heap; the parent also checks
    # physical RSS, while --max-old-space-size bounds the managed heap.
    resource.setrlimit(resource.RLIMIT_AS, (32 * 1024**3, 32 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (55, 55))
    resource.setrlimit(resource.RLIMIT_FSIZE, (2 * 1024**2, 2 * 1024**2))
    # Resolve both libraries while the trusted Python launcher can still read
    # system metadata. Restrictions are installed before Node or user code.
    library = ctypes.util.find_library('seccomp')
    if not library: raise RuntimeError('libseccomp is unavailable')
    native.landlock(policy)
    # Use the resolved absolute library path in the constrained loader.
    original_find = ctypes.util.find_library
    ctypes.util.find_library = lambda name: library if name == 'seccomp' else original_find(name)
    seccomp()
    os.execve(policy['command'][0], policy['command'], policy['environment'])


if __name__ == '__main__':
    try: main()
    except BaseException as error:
        print('SANDBOX_ERROR: ' + str(error), file=sys.stderr, flush=True); sys.exit(126)
