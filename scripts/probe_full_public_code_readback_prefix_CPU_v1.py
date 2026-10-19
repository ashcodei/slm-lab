"""Exercise full-shard readback on a sealed original 1,024-row prefix.

The producer may append; its live output is never bound as a frozen file.
Only the copied complete-LF prefix is sealed. This is an integration probe,
not full source admission, fresh execution or a model-quality assessment.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import pyarrow.parquet as pq
from tokenizers import Tokenizer

from slm_lab.bounded_disk_reserve_wait_v1 import wait_for_reserve
from slm_lab.local_source_closure_v2 import python_sources
from slm_lab.public_code_disposition_readback_v1 import rebuild, row_sha
from slm_lab.public_code_instruction_screen_v1 import parse_whole
from slm_lab.public_synthetic_code_screen_policy_v1 import body_fingerprint

ROOT = Path(__file__).resolve().parents[1]
QUAL = ROOT/'reports/full-public-code-shard-qualification-CPU-v1'
OUT = ROOT/'reports/full-public-code-readback-prefix-probe-CPU-v1'
DATA = ROOT/'data/public-code-instruction-candidate-v1'
VETO = ROOT/'reports/public-code-instruction-precision-counterexample-CPU-v1'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text())


def atomic(path, value):
    pending = path.with_suffix(path.suffix+'.pending')
    with pending.open('x') as stream:
        stream.write(json.dumps(value, indent=2)+'\n')
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(path)


def main():
    if (OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19
            or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'])
            or '19 passed' not in (ROOT/'reports/full-public-code-readback-v1-tests.log').read_text()):
        raise ValueError('Fresh tested CUDA-hidden single-thread nice19 source probe required')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    wait_for_reserve(ROOT, minimum_bytes=40*1024**3)
    producer = read(QUAL/'plan.json')
    files = dict(producer['source_files_sha256'])
    files.update({str(p):sha(p) for p in [QUAL/'plan.json', ROOT/'reports/full-public-code-readback-v1-tests.log',
        *python_sources(ROOT, [Path(__file__), ROOT/'scripts/read_full_public_code_shard_CPU_v1.py',
            ROOT/'tests/test_public_code_candidate_roles_v1.py', ROOT/'tests/test_public_code_disposition_readback_v1.py'])]})
    if any(sha(p) != digest for p,digest in files.items()):
        raise ValueError('Every producer, original source and reader byte must match')
    OUT.mkdir()
    me = dict(pid=os.getpid(), ticks=Path('/proc/self/stat').read_text().split(') ', 1)[1].split()[19])
    try:
        sealed = OUT/'sealed-first1024-full-shard-dispositions.jsonl'
        live_file = QUAL/'all100000-original-dispositions.jsonl'
        before = live_file.stat()
        with live_file.open('rb') as stream, sealed.open('xb') as target:
            for position in range(1024):
                line = stream.readline(8*1024**2)
                if (not line.endswith(b'\n') or json.loads(line)['original_shard_position'] != position
                        or target.tell()+len(line) > 64*1024**2):
                    raise ValueError('Exactly 1024 complete bounded original records required')
                target.write(line)
            target.flush()
            os.fsync(target.fileno())
        after = live_file.stat()
        if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino) or after.st_size < before.st_size:
            raise ValueError('The original append-only producer file was replaced or truncated')
        files[str(sealed)] = sha(sealed)
        atomic(OUT/'plan.json', dict(owner=me, source_files_sha256=files,
            producer_owner=producer['owner'], complete_LF_prefix_sealed_only=True,
            live_producer_output_not_frozen_or_rewritten=True, exact_prefix_positions=1024,
            no_new_program_executions=True, no_neural_forward_or_GPU_lease=True, training_admitted=False))
        with sealed.open('rb') as stream:
            prior = list(map(json.loads, stream))
        with (ROOT/'reports/public-code-instruction-qualification-CPU-v2/all1024-original-dispositions.jsonl').open('rb') as stream:
            old = list(map(json.loads, stream))
        with (ROOT/'reports/public-code-instruction-readback-CPU-v1/all1024-readback-observations.jsonl').open('rb') as stream:
            pilot = list(map(json.loads, stream))
        rows = []
        for batch in pq.ParquetFile(DATA/'train-00000-of-00050.parquet').iter_batches(batch_size=256, use_threads=False):
            rows.extend(batch.to_pylist())
            if len(rows) >= 1024:
                break
        if len(rows) != 1024 or len(old) != 1024 or len(pilot) != 1024:
            raise ValueError('Whole original source and frozen prefix readbacks required')
        veto = read(VETO/'semantic-veto.json')
        if row_sha(rows[veto['original_shard_position']]) != veto['original_entry_sha256']:
            raise ValueError('Known wrong whole teacher response must remain exact and excluded')
        bad_ast = body_fingerprint(parse_whole(rows[veto['original_shard_position']])['program'])
        tokenizer = Tokenizer.from_file(str(ROOT/'data/public-reasoning-curriculum-v1/tokenizer.json'))
        counts, reused = Counter(), Counter()
        natives = targets = contexts = 0
        for position, (row, disposition) in enumerate(zip(rows, prior)):
            result, native, _ = rebuild(row, position, old[position]['protected_overlap'], tokenizer, veto, bad_ast,
                disposition, pilot[position])
            counts[result['status']] += 1
            natives += native is not None
            if result['screen_pass_candidate']:
                targets += native['response_EOS_targets']
                contexts += native['complete_context_tokens']
            if 'execution_reused_from_exact_fresh_pilot_readback' in result:
                reused['reused' if result['execution_reused_from_exact_fresh_pilot_readback'] else 'new'] += 1
        if any(sha(p) != digest for p,digest in files.items()):
            raise ValueError('All original, sealed prefix and reader bytes must remain unchanged')
        atomic(OUT/'complete.json', dict(complete=True, passed=True, at=datetime.now(timezone.utc).isoformat(), owner=me,
            source_files_sha256=files, exact_original_positions_rebuilt=1024,
            actual_disposition_counts=dict(counts), actual_execution_reuse_counts=dict(reused),
            complete_native_rows_rebuilt=natives, original_candidate_response_EOS_targets=targets,
            original_candidate_complete_context_tokens=contexts,
            uses_previously_freshly_checked_protected_prefix_not_new_full_scan=True,
            full100000_readback_and_fresh_candidate_execution_still_pending=True,
            new_program_executions=0, new_neural_updates=0, training_admitted=False, goal_achieved=False))
        print(json.dumps(dict(passed=True, exact_original_positions=1024, counts=dict(counts))), flush=True)
    except BaseException as error:
        atomic(OUT/'failure.json', dict(owner=me, error=repr(error), automatic_retry=False, training_admitted=False))
        raise


if __name__ == '__main__':
    main()
