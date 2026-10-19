"""Bounded pidfd suspension of exact childless CUDA-hidden idle controllers.

Only a single-threaded, non-neural coordinator may fork the independent rescue
guard. Pipe EOF, a protected predecessor's exit, or the hard deadline resumes
every held process descriptor, including when the coordinator is killed.
"""
from contextlib import contextmanager
import ctypes
import json
import os
from pathlib import Path
import select
import signal
import time

from .exact_waiting_process_retirement_v1 import observe, pidfd_open, syscall, terminal


def atomic(path, value):
    path=Path(path); temporary=path.with_suffix(path.suffix+'.pending')
    with temporary.open('x') as stream:
        stream.write(json.dumps(value,indent=2)+'\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def state(identity):
    if terminal(identity):return None
    return (Path('/proc')/str(identity['pid'])/'stat').read_text().split(') ',1)[1].split()[0]


def send(fd, value):
    if hasattr(signal,'pidfd_send_signal'):
        return signal.pidfd_send_signal(fd,value)
    return syscall(424,ctypes.c_int(fd),ctypes.c_int(value),ctypes.c_void_p(None),ctypes.c_uint(0))


def resume(held):
    result=[]
    for identity,fd in held:
        try:
            send(fd,signal.SIGCONT)
            result.append(dict(pid=identity['pid'],ticks=identity['ticks'],SIGCONT_sent_through_held_pidfd=True))
        except ProcessLookupError:
            result.append(dict(pid=identity['pid'],ticks=identity['ticks'],actual_original_birth_terminal=True))
    return result


@contextmanager
def pause(observations, folder, *, maximum_seconds=45, release_on_exit=()):
    folder=Path(folder)
    if (folder.exists() or not observations or not 0<maximum_seconds<=120
            or len(list(Path('/proc/self/task').iterdir()))!=1
            or len({v['pid'] for v in observations})!=len(observations)
            or {v['pid'] for v in observations}&{v['pid'] for v in release_on_exit}):
        raise ValueError('Fresh bounded single-threaded coordinator, distinct exact idle waiters and protected predecessors required')
    held=[]; protected=[]; child=None; reader=writer=None; stopped=False
    folder.mkdir(parents=True)
    deadline=time.monotonic()+maximum_seconds
    try:
        for identity in observations:
            current=observe(identity,identity['command'])
            if current['state'] not in {'R','S'}:
                raise ValueError('Never take ownership of an already paused or uninterruptible controller')
            fd=pidfd_open(identity['pid']);held.append((identity,fd));observe(identity,identity['command'])
        for identity in release_on_exit:
            if terminal(identity):raise ValueError('Every protected predecessor must still be live')
            fd=pidfd_open(identity['pid']);protected.append((identity,fd))
            if terminal(identity):raise ValueError('A protected predecessor ended before the bounded operation')
        reader,writer=os.pipe()
        child=os.fork()
        if child==0:
            os.close(writer)
            def interrupted(signum,frame):raise SystemExit('rescue interrupted')
            signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
            poller=select.poll();poller.register(reader,select.POLLIN|select.POLLHUP)
            for _,fd in protected:poller.register(fd,select.POLLIN)
            reason='hard_monotonic_deadline'
            try:
                while time.monotonic()<deadline:
                    events=poller.poll(max(1,min(100,int((deadline-time.monotonic())*1000))))
                    if events:
                        reason='coordinator_exit_or_release_pipe' if any(fd==reader for fd,_ in events) else 'protected_predecessor_exit'
                        break
            finally:
                try:
                    restored=resume(held)
                    atomic(folder/'rescue-complete.json',dict(complete=True,reason=reason,restored=restored,
                        no_numeric_PID_signals=True,neural_worker_signals=0))
                finally:os._exit(0)
        os.close(reader);reader=None
        child_info=(Path('/proc')/str(child)/'stat').read_text().split(') ',1)[1].split()
        atomic(folder/'rescue-plan.json',dict(owner=dict(pid=os.getpid(),ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]),
            rescue_owner=dict(pid=child,ticks=child_info[19]),maximum_seconds=maximum_seconds,
            targets=observations,protected_predecessors=list(release_on_exit),all_target_pidfds_opened_before_any_signal=True))
        for identity,fd in held:
            if time.monotonic()>=deadline or any(terminal(v) for v,_ in protected):
                raise ValueError('Bounded operation expired or protected predecessor ended before suspension')
            observe(identity,identity['command']);send(fd,signal.SIGSTOP);stopped=True
        end=min(deadline,time.monotonic()+3)
        while not all(state(v) in {'T','t'} for v,_ in held):
            if time.monotonic()>=end:raise ValueError('Every exact idle controller must be observed suspended')
            time.sleep(.01)
        for identity,_ in held:observe(identity,identity['command'])
        yield dict(rescue_owner=dict(pid=child,ticks=child_info[19]),actual_suspended=observations,
                   hard_monotonic_deadline=deadline,release_pipe_rescue_armed=True)
    finally:
        if stopped:resume(held)
        if writer is not None:os.close(writer)
        if reader is not None:os.close(reader)
        if child not in (None,0):
            end=time.monotonic()+3
            while True:
                actual,_=os.waitpid(child,os.WNOHANG)
                if actual:break
                if time.monotonic()>=end:break
                time.sleep(.01)
        for _,fd in held+protected:os.close(fd)
