"""Restore two unchanged optional CPU scopes after terminal dependency failure."""
import json
import os
from pathlib import Path
import subprocess
import time

from slm_lab.corpus import file_hash
from slm_lab.exact_waiting_process_retirement_v1 import terminal

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/public-CPU-queues-assessment-recovery-v1'


def read(path):
    return json.loads(Path(path).read_text())


def scope(queue):
    return {k: v for k, v in queue.items() if k not in ['at', 'status', 'source_files_sha256', 'expires_at_unix']
        and 'owner' not in k and 'predecessor' not in k}


def main():
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19 or OUT.exists()
            or '3 passed' not in (ROOT / 'reports/public-reasoning-warmup-gate-v2-tests.log').read_text()):
        raise ValueError('Fresh tested CUDA-hidden nice19 queue recovery required')
    OUT.mkdir()
    rows = []
    for stem, script in [('public-edge-preference-comparison-CPU', 'train_public_edge_preferences_CPU'),
            ('public-reasoning-warmup-CPU', 'train_public_reasoning_warmup_CPU')]:
        old = read(ROOT / ('reports/' + stem + '-v1-queue.json'))
        if not terminal(old['owner']):
            raise ValueError('Never replace an actually live CPU predecessor')
        queue = ROOT / ('reports/' + stem + '-v2-queue.json')
        target = ROOT / ('scripts/' + script + '_v2.py')
        if queue.exists():
            raise ValueError('Preserve every previous new queue')
        with (OUT / (stem + '.log')).open('xb') as log:
            child = subprocess.Popen([str(ROOT / '.venv/bin/python'), '-u', str(target)], cwd=ROOT,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=dict(os.environ))
        parts = Path('/proc', str(child.pid), 'stat').read_text().split(') ', 1)[1].split()
        birth = dict(pid=child.pid, ticks=parts[19])
        while not queue.exists():
            if child.poll() is not None:
                raise ValueError('New CPU controller ended before its real queue: ' + script)
            time.sleep(2)
        new = read(queue)
        if new['owner'] != birth or scope(old) != scope(new):
            raise ValueError('Every actual CPU birth and full original scientific queue scope must match')
        rows.append(dict(queue=str(queue), owner=birth, source_sha256=file_hash(target),
            original_scope=scope(old), actual_new_scope=scope(new)))
        (OUT / 'progress.json').write_text(json.dumps(dict(queued=rows, neural_updates=0), indent=2) + '\n')
    (OUT / 'complete.json').write_text(json.dumps(dict(complete=True, queues=rows,
        exact_live_interim_assessment_unchanged=True, no_signals_sent=True, no_GPU_context_or_lease=True,
        original_complete_training_and_native_populations_preserved=True, goal_achieved=False), indent=2) + '\n')


if __name__ == '__main__':
    main()
