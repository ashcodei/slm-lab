"""Retire only the verified GO-spelling-bug CPU collector; preserve all data."""
from collections import Counter
import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.exact_waiting_process_retirement_v1 import pidfd_open, syscall

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/public-billion-source-language-handoff-v2'
OLD = ROOT/'reports/public-billion-causal-candidate-CPU-v1'
EXPECTED = dict(pid=3815051, ticks='43542441')


def observe(identity):
    proc = Path('/proc')/str(identity['pid'])
    if not proc.exists():
        return dict(identity, terminal=True, state='absent')
    values = (proc/'stat').read_text().split(') ', 1)[1].split()
    if values[19] != identity['ticks']:
        return dict(identity, terminal=True, state='different_birth')
    return dict(identity, terminal=values[0] in {'Z', 'X'}, state=values[0],
                cmdline=(proc/'cmdline').read_bytes().replace(b'\0', b' ').decode())


def gpu_workers():
    raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                  '--format=csv,noheader,nounits'], text=True)
    pids = sorted({int(p.strip()) for p in raw.splitlines() if p.strip()})
    if pids != [3811367, 3811368, 3811369, 3811370]:
        raise ValueError('The four exact original training workers must remain the sole GPU owners')
    result = [observe(dict(pid=p, ticks='42912649')) for p in pids]
    if any(p['terminal'] for p in result):
        raise ValueError('Every protected GPU birth must still be live')
    return result


def send_int(fd):
    if hasattr(signal, 'pidfd_send_signal'):
        return signal.pidfd_send_signal(fd, signal.SIGINT)
    return syscall(424, ctypes.c_int(fd), ctypes.c_int(signal.SIGINT), ctypes.c_void_p(None), ctypes.c_uint(0))


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or OUT.exists():
        raise ValueError('Fresh GPU-hidden source-only handoff required')
    compatibility_test = ROOT/'reports/public-source-sigint-v2-tests.log'
    if '1 passed' not in compatibility_test.read_text():
        raise ValueError('Actual Linux pidfd SIGINT compatibility control must pass')
    import pyarrow.parquet as pq
    plan_path = OLD/'plan.json'
    plan = json.loads(plan_path.read_text())
    if plan['owner'] != EXPECTED:
        raise ValueError('Only the exact verified original collector may be signalled')
    guard = FileBindings(plan['source_files_sha256'])
    guard.check(full=True)
    original = observe(EXPECTED)
    if original['terminal'] or 'scripts/collect_billion_public_text_candidate_CPU_v1.py' not in original['cmdline']:
        raise ValueError('Exact live affected CPU collector required')
    tests = ROOT/'reports/public-pretraining-intake-v2-tests.log'
    if '3 passed' not in tests.read_text():
        raise ValueError('Real GO-source, native-byte and unchanged-license tests must pass first')
    replacement = ROOT/'scripts/collect_billion_public_text_candidate_CPU_v2.py'
    body = replacement.read_text()
    if ('language=canonical_language(row.get(\'language\'))' not in body
            or 'public_pretraining_intake_v2 import candidate,language as canonical_language' not in body):
        raise ValueError('Replacement must normalize both the quota route and the intake language')
    source = ROOT/'data/public-pretraining-source-preflight-v1/code/original.parquet'
    language_counts = Counter()
    for batch in pq.ParquetFile(source).iter_batches(columns=['language'], batch_size=4096, use_threads=False):
        language_counts.update(batch.column(0).to_pylist())
    if language_counts['GO'] < 1 or language_counts['Go'] != 0:
        raise ValueError('Original complete publisher census must demonstrate the exact spelling mismatch')
    protected = gpu_workers()
    files = [Path(__file__), plan_path, tests, replacement, compatibility_test,
             ROOT/'src/slm_lab/exact_waiting_process_retirement_v1.py',
             ROOT/'reports/public-billion-source-language-handoff-v1.log',
             ROOT/'src/slm_lab/public_pretraining_intake_v2.py',
             ROOT/'tests/test_public_pretraining_intake_v2.py']
    OUT.mkdir()
    receipt = dict(at=datetime.now(timezone.utc).isoformat(), original_owner=EXPECTED,
        actual_original_observation=original, actual_publisher_language_census=dict(language_counts),
        source_file_sha256=file_hash(source), actual_GPU_workers_before=protected,
        verified_bug='Original publisher GO was excluded by Go-only quota routing and intake; the declared 25M Go quota was unreachable.',
        replacement_scope='GO to Go spelling only; original publisher fields, program bytes, native BPE, masks, EOS, license rules and roles retained.',
        replacement_ready=True, original_progress_snapshot=json.loads((OLD/'progress.json').read_text()),
        source_files_sha256={str(p):file_hash(p) for p in files},
        all_partial_original_data_will_be_preserved=True, neural_jobs_will_not_be_signalled=True,
        optimization_admitted=False, model_updates=0, goal_achieved=False)
    atomic_json(OUT/'before-retirement.json', receipt)
    fd = pidfd_open(EXPECTED['pid'])
    try:
        current = observe(EXPECTED)
        if current['terminal'] or current['cmdline'] != original['cmdline']:
            raise ValueError('Target birth or command changed before exact pidfd SIGINT')
        send_int(fd)
    finally:
        os.close(fd)
    deadline = time.monotonic()+60
    while not observe(EXPECTED)['terminal']:
        if time.monotonic() > deadline:
            raise TimeoutError('Source handoff stays closed until the exact collector is terminal')
        time.sleep(.2)
    failure_path = OLD/'failure.json'
    failure = json.loads(failure_path.read_text())
    if failure['error'] != 'KeyboardInterrupt()' or not failure['partial_original_data_retained']:
        raise ValueError('The collector must close its whole-source files and preserve the partial candidate')
    guard.check(full=True)
    gpu_workers()
    data = ROOT/'data/public-billion-causal-candidate-v1'
    receipt.update(complete=True, original_owner_terminal=observe(EXPECTED),
        partial_source_candidate=failure['partial_candidate'], actual_GPU_workers_after=gpu_workers(),
        partial_original_data_files_sha256={str(p):file_hash(p) for p in data.rglob('*') if p.is_file()},
        original_failure_sha256=file_hash(failure_path), original_source_bindings_still_exact=True,
        only_original_CPU_collector_received_exact_pidfd_SIGINT=True,
        all_original_training_and_evaluation_controllers_preserved=True,
        replacement_not_launched_by_this_receipt=True)
    atomic_json(OUT/'complete.json', receipt)
    print(json.dumps(dict(complete=str(OUT/'complete.json'), preserved_partial_train_targets=
        failure['partial_candidate']['actual_train_targets'], original_GPU_workers_preserved=True)), flush=True)


if __name__ == '__main__':
    main()
