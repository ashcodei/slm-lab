"""Retire only exact, CUDA-hidden, childless CPU controllers using pidfds."""
import os
import ctypes
from pathlib import Path
import platform
import signal
import sys
import time


def pidfd_open(pid):
    if hasattr(os,'pidfd_open'):return os.pidfd_open(pid)
    return syscall(434,ctypes.c_int(pid),ctypes.c_uint(0))


def send_term(fd):
    if hasattr(signal,'pidfd_send_signal'):return signal.pidfd_send_signal(fd,signal.SIGTERM)
    return syscall(424,ctypes.c_int(fd),ctypes.c_int(signal.SIGTERM),ctypes.c_void_p(None),ctypes.c_uint(0))


def syscall(number,*arguments):
    # This conda Python/libc predates the wrappers; the Linux kernel still
    # supplies pidfds. Never fall back to signalling a numeric reused PID.
    if sys.platform!='linux' or platform.machine()!='x86_64':raise ValueError('An available native pidfd API or verified Linux x86_64 syscall ABI required')
    libc=ctypes.CDLL(None,use_errno=True);call=libc.syscall;call.restype=ctypes.c_long
    result=call(ctypes.c_long(number),*arguments)
    if result<0:
        error=ctypes.get_errno();raise OSError(error,os.strerror(error))
    return result


def observe(identity,expected_command=None):
    proc=Path('/proc')/str(identity['pid'])
    try:fields=(proc/'stat').read_text().split(') ',1)[1].split()
    except FileNotFoundError:raise ValueError('Exact waiting controller is absent')
    if fields[19]!=identity['ticks'] or fields[0] in {'Z','X'}:raise ValueError('Exact live process birth required')
    children=(proc/'task'/str(identity['pid'])/'children').read_text().strip()
    visible=next((v.split('=',1)[1] for v in (proc/'environ').read_bytes().decode().split('\0') if v.startswith('CUDA_VISIBLE_DEVICES=')),None)
    command=(proc/'cmdline').read_bytes().decode().replace('\0',' ').strip()
    if children or visible!='' or (expected_command is not None and expected_command not in command):
        raise ValueError('Only the exact hidden childless waiting CPU controller may be retired')
    return dict(pid=identity['pid'],ticks=fields[19],state=fields[0],children=children,CUDA_VISIBLE_DEVICES=visible,command=command)


def terminal(identity):
    try:fields=(Path('/proc')/str(identity['pid'])/'stat').read_text().split(') ',1)[1].split()
    except FileNotFoundError:return True
    return fields[19]!=identity['ticks'] or fields[0] in {'Z','X'}


def retire(observations):
    held=[]
    try:
        # Open every process descriptor before signalling any process. A reused
        # numeric PID can never receive a signal through another birth's pidfd.
        for row in observations:
            observe(row,row['command']);fd=pidfd_open(row['pid']);held.append((row,fd));observe(row,row['command'])
        for row,fd in held:
            observe(row,row['command']);send_term(fd)
        deadline=time.monotonic()+10
        while not all(terminal(row) for row,_ in held):
            if time.monotonic()>deadline:raise ValueError('A declared waiting controller did not terminate; handoff remains closed')
            time.sleep(.05)
    finally:
        for _,fd in held:os.close(fd)
