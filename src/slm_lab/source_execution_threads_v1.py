"""Give newly created source-scoring threads idle CPU priority on Linux.

Use the actual native thread ID. Never change the parent neural thread, its
affinity or an existing process. Bounded workers spawned here inherit this
low-priority execution context; GPUwork remains in the parent thread.
"""
import os
import threading


def initialize():
    tid = threading.get_native_id()
    if tid == os.getpid(): raise ValueError('The main neural/process thread cannot be reprioritized by this initializer')
    os.setpriority(os.PRIO_PROCESS, tid, 19)
    os.sched_setscheduler(tid, os.SCHED_IDLE, os.sched_param(0))
    if os.getpriority(os.PRIO_PROCESS, tid) != 19 or os.sched_getscheduler(tid) != os.SCHED_IDLE:
        raise ValueError('Every new source-scoring thread must actually be nice19/SCHED_IDLE')


def observation():
    tid = threading.get_native_id()
    return dict(native_thread_id=tid, process_id=os.getpid(), actual_nice=os.getpriority(os.PRIO_PROCESS, tid),
        actual_scheduler=os.sched_getscheduler(tid), thread_is_main=tid == os.getpid())
