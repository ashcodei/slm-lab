"""Read the producer's already closed full metadata while controls continue."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'reports/full-public-token-suffix-qualification-CPU-v1'
OUT = ROOT / 'reports/full-public-token-suffix-metadata-readback-CPU-v1'
PARENT = dict(pid=146084, ticks='55111340')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    if (OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19
            or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'])):
        raise ValueError('Fresh CUDA-hidden nice19 single-thread source-only metadata readback required')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    from record_visual_stage_and_first_repair_arm_v1 import read, atomic
    from slm_lab.exact_waiting_process_retirement_v1 import terminal
    plan = read(SOURCE / 'plan.json')
    if plan['owner'] != PARENT or terminal(PARENT):
        raise ValueError('This reader expects the exact original producer still executing its separate control journal')
    metadata = SOURCE / 'all14884-native-view-descriptors.jsonl'
    target = metadata.resolve()
    for fd in Path('/proc', str(PARENT['pid']), 'fd').iterdir():
        try:
            if fd.resolve() == target:
                raise ValueError('The original metadata writer must close before independent full reconstruction')
        except FileNotFoundError:
            pass
    tests = ROOT / 'reports/independent-token-suffix-readback-v2-tests.log'
    if '8 passed' not in tests.read_text():
        raise ValueError('The independent view reconstruction controls must pass')
    import torch
    from slm_lab.expanded_execution_corpus_v1 import ExpandedExecutionCorpus
    from slm_lab.independent_token_suffix_readback_v1 import rebuild
    from slm_lab.local_source_closure_v2 import python_sources
    torch.set_num_threads(1)
    files = dict(plan['source_files_sha256'])
    files.update({str(p): sha(p) for p in [SOURCE / 'plan.json', metadata, tests, *python_sources(ROOT, [Path(__file__)])]})
    corpus = ExpandedExecutionCorpus(ROOT)
    totals = {str(n): Counter() for n in [4, 16, 64, 0]}
    rows = 0
    OUT.mkdir()
    try:
        with metadata.open('rb') as stream:
            for position in range(len(corpus)):
                source = corpus.row(position)
                ids, labels = corpus.sequence(position)
                for level in [4, 16, 64, 0]:
                    raw = next(stream)
                    if not raw.endswith(b'\n'):
                        raise ValueError('Closed native descriptor rows must be complete LF records')
                    row = json.loads(raw)
                    identity = {k: row.pop(k) for k in ['source_position', 'corpus_id', 'kind']}
                    if identity != dict(source_position=position, corpus_id=source['id'], kind=source['kind']):
                        raise ValueError('Every original source and exact per-level position must reconstruct')
                    actual = rebuild(corpus.tokenizer, source['original'], ids, labels, level, row)
                    info = actual['metadata']
                    totals[str(level)].update(views=1, original_context_tokens=len(ids),
                        original_response_EOS_targets=len(ids) - source['prefix_positions'],
                        actual_hint_views=int(info['retained_reference_tokens'] > 0),
                        actual_missing_response_EOS_targets=info['actual_missing_response_EOS_targets'])
                    rows += 1
            if next(stream, None) is not None:
                raise ValueError('Extra hidden metadata rows are forbidden')
        if rows != 14884 or any(t['views'] != 3721 or t['original_context_tokens'] != 2287522
                or t['original_response_EOS_targets'] != 584218 for t in totals.values()):
            raise ValueError('All3721human source contexts and every unconditional CE/EOS target must be preserved at all4levels')
        if torch.cuda.is_initialized() or any(sha(p) != h for p, h in files.items()):
            raise ValueError('Every immutable source byte and source-only CUDA-hidden execution must remain unchanged')
        atomic(OUT / 'complete.json', dict(complete=True, passed=True, at=datetime.now(timezone.utc).isoformat(),
            source_files_sha256=files, actual_original_human_sources=3721, actual_all_four_native_views=14884,
            actual_per_level_totals={k: dict(v) for k, v in totals.items()},
            all3721original_source_roles_inputs_masks_reference_bytes_unicode_prefixes_suffixes_and_EOS_budgets_rebuilt=True,
            original_unconditional_CE_context_tokens_per_level=2287522, original_unconditional_CE_EOS_targets_per_level=584218,
            original_metadata_closed_before_reader=True, actual_original_producer_owner=PARENT,
            no_live_journal_or_progress_file_source_binding=True,
            full_reference_prefix_runtime_qualification_and_full_fresh_execution_readback_still_running=True,
            source_training_admitted=False, production_RL_admitted=False, student_generations=0, neural_forwards=0,
            model_updates=0, GPUcontexts=0, every_existing_neural_worker_unchanged=True, goal_achieved=False))
        print(json.dumps(dict(complete=True, passed=True, original_human_sources=3721, native_views=14884, source_training_admitted=False)), flush=True)
    except BaseException as exc:
        atomic(OUT / 'failure.json', dict(error=repr(exc), automatic_retry=False, production_RL_admitted=False))
        raise


if __name__ == '__main__':
    main()
