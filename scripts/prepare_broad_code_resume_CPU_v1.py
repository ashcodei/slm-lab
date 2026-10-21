"""Revalidate all original broad-code sources and the committed four-rank prefix.

Reuses the existing independent interim reader on the original own policy and
all 3,721 sources. Neither optimizer/RNG restoration nor GPU recovery is claimed.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT/'reports/broad-anchored-code-v7'
OUT = ROOT/'reports/broad-code-resume-preparation-v1'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(name, value):
    p = OUT/name; q = p.with_suffix(p.suffix+'.pending')
    with q.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(q, p)


def main():
    if OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('A fresh explicit CPU-only full-source preparation is required')
    os.nice(10); allowed = sorted(os.sched_getaffinity(0)); os.sched_setaffinity(0, {allowed[min(2, len(allowed)-1)]})
    import torch
    from slm_lab.broad_anchored_policy_v3 import fresh
    from slm_lab.broad_interim_snapshot_reader_v1 import materialize
    from slm_lab.broad_resume_journal_v1 import copy_prefix
    from slm_lab.codenet_checkpoint_v1 import FileBindings
    from slm_lab.expanded_execution_corpus_v1 import ExpandedExecutionCorpus
    from slm_lab.local_source_closure_v2 import python_sources
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    plan_path = OLD/'plan.json'; receipt_path = OLD/'train-anchor_CE-resume-distributed-state.json'
    plan = json.loads(plan_path.read_text()); receipt = json.loads(receipt_path.read_text()); digest = sha(plan_path)
    if receipt['completed'] != 1536 or plan['planned_steps'] != 3721 or plan['original_coding_sources'] != 3721:
        raise ValueError('The exact original full-source epoch and committed generation are required')
    private = sorted(receipt['all_four_private_states'], key=lambda x:x['rank'])
    if [p['rank'] for p in private] != list(range(4)):
        raise ValueError('Every original private worker checkpoint is required')
    prefixes = {(p['external_journal_prefix_bytes'], p['external_journal_prefix_sha256']) for p in private}
    if len(prefixes) != 1:
        raise ValueError('Every original private state must bind the same complete journal prefix')
    files = dict(plan['source_files_sha256'])
    files.update({str(p):sha(p) for p in [plan_path, receipt_path, *python_sources(ROOT, [Path(__file__).resolve()])]})
    for record in private:
        if sha(record['path']) != record['sha256']:
            raise ValueError('An actual original committed private file changed')
        files[record['path']] = record['sha256']
    OUT.mkdir(); start = time.perf_counter()
    write('plan.json', dict(at=datetime.now(timezone.utc).isoformat(), owner=dict(pid=os.getpid(),
        birth=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]), original_neural_plan_sha256=digest,
        source_files_sha256=files, original_total_sources=3721, committed_updates=1536, GPU_hidden=True,
        optimizer_and_private_RNG_not_restored=True, no_new_neural_training_or_HF_download=True))
    guard = FileBindings(files); guard.check(full=True)
    write('progress.json', dict(phase='all_original_source_bindings_rehashed', bound_files=len(files)))
    length, prefix_sha = prefixes.pop(); prefix = OUT/'anchor_CE-committed-prefix.jsonl'
    copied = copy_prefix(OLD/'anchor_CE-metrics.jsonl', prefix, length=length, expected_sha256=prefix_sha,
                         completed=1536, minimum_free_bytes=8*1024**3)
    write('prefix.json', copied)
    corpus = ExpandedExecutionCorpus(ROOT)
    if len(corpus) != 3721 or sum(row['response_EOS_targets'] for row in corpus.rows) != 584218:
        raise ValueError('All173documented/3548APPS whole sources and complete targets must reconstruct')
    write('progress.json', dict(phase='all3721original_sources_reconstructed', original_source_provenance=corpus.provenance))
    model, metadata = fresh(plan, 'cpu')
    proof = materialize(model, corpus, plan, digest, [r['path'] for r in private], prefix, expected_completed=1536)
    if torch.cuda.is_initialized():
        raise ValueError('Preparation must never initialize a CUDA context')
    guard.check(full=True)
    write('complete.json', dict(complete=True, at=datetime.now(timezone.utc).isoformat(), source_files_sha256=files,
        original_neural_plan=str(plan_path), original_neural_plan_sha256=digest, all_original_sources=3721,
        complete_reference_EOS_targets=584218, original_source_provenance=corpus.provenance,
        original_committed_updates=1536, original_planned_updates=3721, private_states=private,
        committed_prefix=copied, actual_full_policy_and_all_source_prefix_readback=proof,
        actual_registered_parameters=sum(p.numel() for p in model.parameters()),
        whole_original_dataset_and_committed_prompts_actions_masks_reconstructed=True,
        old_uncommitted_tail_never_counted_as_training=True, no_model_optimization_or_neural_replay=True,
        actual_CUDA_RNG_recovery_and_GPU_capacity_still_required=True,
        original_full_execution_arm_and_all_assessments_still_required=True,
        seconds=time.perf_counter()-start, candidate_promoted=False, goal_achieved=False))


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        if OUT.exists():
            write('failure.json', dict(error=repr(exc), traceback=traceback.format_exc(), goal_achieved=False))
        raise
