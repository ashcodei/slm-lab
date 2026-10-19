"""Single-threaded trusted launcher: constrain native code before exec/constructors.

Requires Linux Landlock ABI >=3 and libseccomp. Never run the target if applying
either policy fails. Only explicit software/temporary paths are granted.
"""
import ctypes
import ctypes.util
import errno
import json
import os
from pathlib import Path
import platform
import resource
import sys

LIBC=ctypes.CDLL(None,use_errno=True)
SECCOMP=None
READ=(1<<2)|(1<<3)
EXEC=1
ALL_FS=(1<<15)-1


class Ruleset(ctypes.Structure):
    _fields_=[('handled_access_fs',ctypes.c_uint64),('handled_access_net',ctypes.c_uint64)]


class PathRule(ctypes.Structure):
    _pack_=1
    _fields_=[('allowed_access',ctypes.c_uint64),('parent_fd',ctypes.c_int32)]


class Comparison(ctypes.Structure):
    _fields_=[('arg',ctypes.c_uint),('op',ctypes.c_int),('a',ctypes.c_uint64),('b',ctypes.c_uint64)]


def checked(value,operation):
    if value<0:raise OSError(ctypes.get_errno(),operation)
    return value


def landlock(policy):
    if platform.machine()!='x86_64':raise RuntimeError('Native sandbox currently supports x86_64 Linux only')
    abi=checked(LIBC.syscall(444,0,0,1),'query Landlock ABI')
    if abi<3:raise RuntimeError('Landlock ABI >=3 required')
    attrs=Ruleset(ALL_FS,3 if abi>=4 else 0)
    descriptor=checked(LIBC.syscall(444,ctypes.byref(attrs),ctypes.sizeof(attrs),0),'create Landlock policy')
    try:
        for path,rights in policy['paths']:
            path=Path(path).resolve(strict=True)
            if not path.is_dir():rights&=EXEC|(1<<1)|(1<<2)|(1<<14)
            fd=os.open(path,getattr(os,'O_PATH',0x200000)|os.O_CLOEXEC)
            try:
                rule=PathRule(rights,fd)
                checked(LIBC.syscall(445,descriptor,1,ctypes.byref(rule),0),'add Landlock path')
            finally:os.close(fd)
        checked(LIBC.prctl(38,1,0,0,0),'set no_new_privs')
        checked(LIBC.syscall(446,descriptor,0),'enforce Landlock policy')
    finally:os.close(descriptor)


def seccomp(mode):
    sec=SECCOMP
    if sec is None:raise RuntimeError('libseccomp was not resolved before confinement')
    sec.seccomp_init.argtypes=[ctypes.c_uint32];sec.seccomp_init.restype=ctypes.c_void_p
    sec.seccomp_release.argtypes=[ctypes.c_void_p]
    sec.seccomp_load.argtypes=[ctypes.c_void_p];sec.seccomp_load.restype=ctypes.c_int
    sec.seccomp_syscall_resolve_name.argtypes=[ctypes.c_char_p];sec.seccomp_syscall_resolve_name.restype=ctypes.c_int
    sec.seccomp_rule_add_array.argtypes=[ctypes.c_void_p,ctypes.c_uint32,ctypes.c_int,ctypes.c_uint,ctypes.POINTER(Comparison)]
    sec.seccomp_rule_add_array.restype=ctypes.c_int
    allow=0x7fff0000;deny=0x00050000|errno.EPERM
    context=sec.seccomp_init(deny if mode=='runtime' else allow)
    if not context:raise RuntimeError('seccomp_init failed')
    def rule(name,action,comparison=None):
        number=sec.seccomp_syscall_resolve_name(name.encode())
        if number<0:return
        result=sec.seccomp_rule_add_array(context,action,number,0 if comparison is None else 1,
                                         None if comparison is None else ctypes.byref(comparison))
        if result<0:raise RuntimeError('Cannot add seccomp rule for '+name+': '+str(result))
    try:
        if mode=='runtime':
            names=('read write readv writev close fstat newfstatat statx lseek mmap mprotect munmap '
                   'brk mremap madvise rt_sigaction rt_sigprocmask rt_sigreturn sigaltstack '
                   'arch_prctl set_tid_address set_robust_list futex getrandom clock_gettime '
                   'gettimeofday nanosleep clock_nanosleep getpid gettid getuid geteuid getgid '
                   'getegid sched_yield sched_getaffinity exit exit_group open openat access '
                   'faccessat faccessat2 readlink readlinkat getcwd fcntl dup dup2 dup3 getrlimit '
                   'getrusage uname execve execveat pread64').split()
            for name in names:rule(name,allow)
            rule('prlimit64',allow,Comparison(0,4,0,0)) # pid==0: own limits only
        elif mode=='compile':
            names=('socket socketpair connect bind listen accept accept4 sendto sendmsg sendmmsg '
                   'recvfrom recvmsg recvmmsg shutdown ptrace process_vm_readv process_vm_writev '
                   'pidfd_open pidfd_getfd pidfd_send_signal kill tkill tgkill mount umount2 '
                   'pivot_root chroot unshare setns bpf perf_event_open io_uring_setup io_uring_enter '
                   'io_uring_register open_by_handle_at fanotify_init keyctl add_key request_key '
                   'reboot kexec_load kexec_file_load swapon swapoff setpriority ioprio_set '
                   'sched_setscheduler sched_setparam shmget shmat shmctl shmdt msgget msgsnd '
                   'msgrcv msgctl semget semop semctl semtimedop').split()
            for name in names:rule(name,deny)
            rule('prlimit64',deny,Comparison(0,1,0,0)) # forbid changing another process
        else:raise ValueError('Unknown native sandbox mode')
        result=sec.seccomp_load(context)
        if result<0:raise RuntimeError('Cannot load seccomp policy: '+str(result))
    finally:sec.seccomp_release(context)


def main():
    global SECCOMP
    policy=json.loads(sys.argv[1]);mode=policy['mode']
    os.chdir(policy['cwd'])
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    resource.setrlimit(resource.RLIMIT_AS,(policy['memory_bytes'],policy['memory_bytes']))
    resource.setrlimit(resource.RLIMIT_CPU,(policy['cpu_seconds'],policy['cpu_seconds']))
    resource.setrlimit(resource.RLIMIT_FSIZE,(policy['file_bytes'],policy['file_bytes']))
    # Resolve libraries before Landlock blocks the loader's ambient searches.
    # seccomp is installed after Landlock, and both remain active across exec.
    library=ctypes.util.find_library('seccomp')
    if not library:raise RuntimeError('libseccomp missing')
    SECCOMP=ctypes.CDLL(library,use_errno=True)
    landlock(policy);seccomp(mode)
    os.execve(policy['command'][0],policy['command'],policy['environment'])


if __name__=='__main__':
    try:main()
    except BaseException as error:
        print('SANDBOX_ERROR: '+str(error),file=sys.stderr,flush=True);sys.exit(126)
