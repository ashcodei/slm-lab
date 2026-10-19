"""Exclusive owned GPU runs after the exact legacy photo waiter is terminal.

The legacy failure and process birth stay explicit and immutable. No synthetic
waiter, signal to a reused PID, or resumption of the failed old run is needed.
The actual kernel flock must belong to the exact live parent for worker work.
"""
import ctypes
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import assess_joint_java_comparison_v1 as resources
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.exact_waiting_process_retirement_v1 import terminal, pidfd_open, send_term, syscall

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports/unconfigured-terminal-waiter-run'
RECEIPT = ROOT / 'reports/legacy-photo-waiter-terminal-v1/complete.json'
LEASE = ROOT / 'reports/code-residual-production-v4.gpu.lock'
WAITER = dict(pid=2754743, ticks='27602681')


def terminal_receipt_sha256():
    return file_hash(RECEIPT)


def lease_sources():
    receipt = json.loads(RECEIPT.read_text())
    return {str(RECEIPT): file_hash(RECEIPT), receipt['legacy_status_file']: receipt['legacy_status_sha256']}


def idle_original():
    receipt = json.loads(RECEIPT.read_text())
    if (receipt.get('complete') is not True or receipt.get('original_waiter') != WAITER
            or receipt.get('original_waiter_terminal') is not True or not terminal(WAITER)
            or file_hash(receipt['legacy_status_file']) != receipt['legacy_status_sha256']):
        raise ValueError('The exact terminal legacy birth and unchanged preserved failure receipt are required')


def lock_owner():
    info = LEASE.stat(); device = (os.major(info.st_dev), os.minor(info.st_dev), info.st_ino)
    found = []
    for line in Path('/proc/locks').read_text().splitlines():
        values = line.split()
        if len(values) < 8 or values[1:4] != ['FLOCK', 'ADVISORY', 'WRITE']:
            continue
        major, minor, inode = values[5].split(':')
        if (int(major, 16), int(minor, 16), int(inode)) == device:
            found.append(int(values[4]))
    if len(found) != 1:
        raise ValueError('Exactly one actual exclusive kernel flock owner is required')
    return found[0]


def owned(plan):
    idle_original()
    expected = terminal_receipt_sha256()
    if (not resources.same(plan['owner']) or plan['waiter'] != WAITER
            or plan.get('terminal_waiter_receipt_sha256') != expected
            or plan['source_files_sha256'].get(str(RECEIPT)) != expected
            or (REPORTS / 'restored.json').exists() or time.time() > plan['expires_at_unix']
            or lock_owner() != plan['owner']['pid']):
        raise ValueError('Exact live parent, terminal legacy receipt, unexpired run and its exclusive kernel flock required')


def hold_original(plan):
    # The original process is terminal, so there is nothing to stop or resume.
    owned(plan)


def registered(subject):
    plan = json.loads((REPORTS / 'plan.json').read_text()); owned(plan)
    with (REPORTS / 'children.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        actors = json.loads((REPORTS / 'children.json').read_text()); actual = resources.identity(os.getpid())
        if not any(a['pid'] == actual['pid'] and a['ticks'] == actual['ticks'] for a in actors):
            actors.append(dict(**actual, subject=subject)); atomic_json(REPORTS / 'children.json', actors)
    return plan


def stop_registered(actor):
    if terminal(actor):
        return
    fd = pidfd_open(actor['pid'])
    try:
        if terminal(actor):
            return
        send_term(fd); end = time.monotonic() + 8
        while not terminal(actor) and time.monotonic() < end:
            time.sleep(.05)
        if not terminal(actor):
            if hasattr(signal, 'pidfd_send_signal'):
                signal.pidfd_send_signal(fd, signal.SIGKILL)
            else:
                syscall(424, ctypes.c_int(fd), ctypes.c_int(signal.SIGKILL), ctypes.c_void_p(None), ctypes.c_uint(0))
    finally:
        os.close(fd)


def restore_owned(plan, reason):
    with (REPORTS / 'restore.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if (REPORTS / 'restored.json').exists():
            return
        actors = json.loads((REPORTS / 'children.json').read_text())
        for actor in actors:
            stop_registered(actor)
        until = time.monotonic() + 10
        while not all(terminal(a) for a in actors):
            if time.monotonic() > until:
                raise ValueError('Registered owned workers remain; do not release successor work')
            time.sleep(.05)
        atomic_json(REPORTS / 'restored.json', dict(at=datetime.now(timezone.utc).isoformat(), reason=reason,
            waiter=None, original_waiter_terminal=True, original_waiter_not_signalled_or_restarted=True,
            terminal_waiter_receipt_sha256=plan['terminal_waiter_receipt_sha256'],
            all_registered_point_actors_terminal=True, goal_achieved=False))


def watchdog():
    plan = json.loads((REPORTS / 'plan.json').read_text())
    while resources.same(plan['owner']) and not (REPORTS / 'restored.json').exists():
        if time.time() > plan['expires_at_unix']:
            stop_registered(plan['owner'])
            restore_owned(plan, 'Owned run lease expired; exact registered workers ended')
            return
        time.sleep(.5)
    if not (REPORTS / 'restored.json').exists():
        restore_owned(plan, 'Exact owned parent ended; registered workers ended before successor admission')


def wait_children(plan, children):
    while any(p.poll() is None for p in children):
        owned(plan)
        if any(p.poll() not in [None, 0] for p in children):
            raise ValueError('An exact owned child failed; preserve partial evidence')
        time.sleep(2)
    if any(p.returncode != 0 for p in children):
        raise ValueError('Every exact owned child must complete successfully')


def launch(plan, args, subject, visible):
    owned(plan)
    with (REPORTS / (subject + '.log')).open('xb') as stream:
        child = subprocess.Popen(['.venv/bin/python', *args], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
            start_new_session=True, env={**os.environ, 'CUDA_VISIBLE_DEVICES': visible,
            'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'TOKENIZERS_PARALLELISM': 'false',
            'CUBLAS_WORKSPACE_CONFIG': ':4096:8'})
    actual = resources.identity(child.pid)
    if not actual:
        raise ValueError('An owned child ended before exact registration')
    with (REPORTS / 'children.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        actors = json.loads((REPORTS / 'children.json').read_text()); actors.append(dict(**actual, subject=subject))
        atomic_json(REPORTS / 'children.json', actors)
    return child
