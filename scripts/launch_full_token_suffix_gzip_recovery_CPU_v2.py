"""Launch two source-only replacements after exact failed predecessors stop."""
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

from record_visual_stage_and_first_repair_arm_v1 import atomic, read, sha

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/full-token-suffix-gzip-recovery-v2-launch'
OLD = ROOT / 'reports/full-public-token-suffix-qualification-CPU-v1'
PRODUCER = ROOT / 'reports/full-public-token-suffix-qualification-gzip-recovery-CPU-v2'
READER = ROOT / 'reports/full-public-token-suffix-readback-gzip-recovery-CPU-v2'


def main():
    if OUT.exists() or PRODUCER.exists() or READER.exists():
        raise ValueError('Preserve all earlier launches, complete or failed source work')
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19
            or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'])):
        raise ValueError('Only CUDA-hidden nice19 single-thread source handoff is allowed')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    from slm_lab.exact_waiting_process_retirement_v1 import terminal
    original = dict(pid=146084, ticks='55111340')
    original_reader = dict(pid=146633, ticks='55128888')
    failure, reader_failure = read(OLD / 'failure.json'), read(ROOT / 'reports/full-public-token-suffix-readback-CPU-v1/failure.json')
    if (not terminal(original) or not terminal(original_reader) or failure['owner'] != original
            or reader_failure['owner'] != original_reader
            or 'Original disk reserve remains unavailable after the bounded wait' not in failure['error']):
        raise ValueError('Only the actual terminal disk-failed source and its terminal original reader may be recovered')
    for name, passed in [('bounded-gzip-observation-stream-v1-tests.log', '5 passed'),
            ('full-token-suffix-gzip-recovery-v2-tests.log', '3 passed'), ('independent-token-suffix-readback-v2-tests.log', '8 passed')]:
        if passed not in (ROOT / 'reports' / name).read_text(): raise ValueError('All original and new complete source/stream controls must pass')
    import torch
    from slm_lab.bounded_gzip_observation_stream_v1 import lines, NEXT_RECORD
    from slm_lab.bounded_disk_reserve_wait_v1 import wait_for_reserve
    from slm_lab.expanded_execution_corpus_v1 import ExpandedExecutionCorpus
    from qualify_full_public_token_suffixes_gzip_recovery_CPU_v2 import validate_record
    torch.set_num_threads(1)
    corpus = ExpandedExecutionCorpus(ROOT)
    metadata_path = OLD / 'all14884-native-view-descriptors.jsonl'
    control_path = OLD / 'all3721-reference-and14884-prefix-controls.jsonl'
    before_sha = sha(control_path)
    metadata_rows = [r for _, r in lines(metadata_path)]
    metadata = {(r['source_position'], r['nominal_missing_non_EOS_tokens']): r for r in metadata_rows}
    if len(metadata_rows) != 14884 or len(metadata) != 14884: raise ValueError('All original four-level metadata must be intact')
    counts = Counter()
    retained = 0
    for _, row in lines(control_path):
        counts.update(validate_record(corpus, retained, row, metadata))
        retained += 1
    corpus.guard.check(full=True)
    if not 0 < retained < 3721 or sha(control_path) != before_sha or torch.cuda.is_initialized():
        raise ValueError('All exact closed original records must remain unchanged and source-only')
    wait_for_reserve(ROOT, minimum_bytes=40 * 1024**3 + NEXT_RECORD, guard=lambda: corpus.guard.check())
    OUT.mkdir()
    names = ['scripts/launch_full_token_suffix_gzip_recovery_CPU_v2.py',
        'scripts/qualify_full_public_token_suffixes_gzip_recovery_CPU_v2.py',
        'scripts/read_full_public_token_suffixes_gzip_recovery_CPU_v2.py',
        'src/slm_lab/bounded_gzip_observation_stream_v1.py',
        'reports/bounded-gzip-observation-stream-v1-tests.log', 'reports/full-token-suffix-gzip-recovery-v2-tests.log',
        'reports/independent-token-suffix-readback-v2-tests.log']
    files = {str(p): sha(p) for p in [OLD / 'failure.json', ROOT / 'reports/full-public-token-suffix-readback-CPU-v1/failure.json',
        metadata_path, control_path, *[ROOT / n for n in names]]}
    atomic(OUT / 'closed-original-prefix-readback.json', dict(at=datetime.now(timezone.utc).isoformat(),
        actual_original_source_records=retained, actual_original_native_views=retained * 4,
        actual_original_reference_and_prefix_controls=retained * 5, actual_reconstructed_counts=dict(counts),
        all_original_source_masks_prefixes_raw_observations_and_dispositions_reconstructed=True,
        all_closed_original_bytes_retained_for_recovery=True, original_source_owner_terminal=True,
        source_files_sha256=files, source_training_admitted=False, neural_forwards=0, goal_achieved=False))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONPATH='src:scripts')
    prefix = ['nice', '-n', '19', 'chrt', '--idle', '0', 'taskset', '-c', '22', str(ROOT / '.venv/bin/python'), '-u']
    launched = []
    def start(script, arguments, report, label):
        log = OUT / (label + '.log')
        with log.open('xb') as stream:
            process = subprocess.Popen([*prefix, str(ROOT / 'scripts' / script), *arguments], cwd=ROOT, env=env,
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        ticks = Path('/proc', str(process.pid), 'stat').read_text().split(') ', 1)[1].split()[19]
        owner = dict(pid=process.pid, ticks=ticks)
        launched.append(dict(label=label, owner=owner, log=str(log)))
        expires = time.monotonic() + 30
        while True:
            if process.poll() is not None: raise ValueError('New source owner exited before publishing its plan: ' + label)
            path = report / 'plan.json'
            if path.exists():
                actual = read(path)
                if actual['owner'] != owner: raise ValueError('The exact newly launched source birth differs')
                break
            if time.monotonic() > expires: raise ValueError('New source plan observation timed out; inspect this same actual owner')
            time.sleep(.1)
        if os.sched_getscheduler(process.pid) != os.SCHED_IDLE:
            raise ValueError('Every new source-only owner must have idle scheduling')
        return owner
    try:
        producer = start('qualify_full_public_token_suffixes_gzip_recovery_CPU_v2.py', [], PRODUCER, 'producer')
        reader = start('read_full_public_token_suffixes_gzip_recovery_CPU_v2.py',
            ['--parent-pid', str(producer['pid']), '--parent-ticks', producer['ticks']], READER, 'independent-reader')
        if any(terminal(r['owner']) for r in launched): raise ValueError('Every newly launched source owner must actually remain live')
        atomic(OUT / 'complete.json', dict(complete=True, at=datetime.now(timezone.utc).isoformat(),
            actual_source_producer=producer, actual_independent_fresh_reader=reader,
            actual_original_closed_source_rows_reconstructed=retained, actual_remaining_sources=3721 - retained,
            original_source_and_reader_failures_preserved=True, old_owners_terminal_before_recovery=True,
            original_full_sources=3721, original_full_native_views=14884, original_reference_prefix_controls=18605,
            new_independent_reader_still_requires_all18605fresh_controls=True,
            original_uncompressed_observation_quota=1024**3, original_other_writer_floor=40 * 1024**3,
            next_complete_record_reserve_bytes=NEXT_RECORD, output_encoding='lossless_gzip_byte_LF_JSON',
            source_files_sha256=files, no_neural_worker_changed_or_new_GPU_context_job_lease=True,
            source_qualification_complete=False, source_training_admitted=False, production_RL_admitted=False, goal_achieved=False))
        print(json.dumps(dict(launched_source_only_owners=launched, retained_original_sources=retained, remaining_original_sources=3721-retained)))
    except BaseException as exc:
        atomic(OUT / 'failure.json', dict(error=repr(exc), actual_launched_source_owners=launched,
            automatic_retry=False, existing_neural_workers_unchanged=True, goal_achieved=False))
        raise


if __name__ == '__main__': main()
