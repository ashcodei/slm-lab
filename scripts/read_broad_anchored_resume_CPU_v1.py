"""Inspect all four retained private broad-code states and their exact journal.

Loads the original own foundation/human policy and imports the saved actor on
CPU. No optimization, CUDA RNG restoration or GPU resume admission is claimed.
Uncommitted journal tails and all original artifacts remain untouched.
"""
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'reports/broad-anchored-code-v7'
OUT = ROOT/'reports/broad-anchored-retained-state-readback-20261021.json'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    if OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Fresh explicit CPU-only retained-state readback required')
    os.nice(10); allowed = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {allowed[min(2, len(allowed)-1)]})
    import torch
    from slm_lab.broad_anchored_policy_v3 import fresh, sources
    from slm_lab.broad_anchored_execution_stage_v1 import semantic_digest
    from slm_lab.broad_code_capacity_v1 import equal
    from slm_lab.local_source_closure_v2 import python_sources
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    start = time.perf_counter(); plan_path = RUN/'plan.json'
    receipt_path = RUN/'train-anchor_CE-resume-distributed-state.json'
    plan = json.loads(plan_path.read_text()); receipt = json.loads(receipt_path.read_text())
    digest = sha(plan_path); files = {str(p): sha(p) for p in [plan_path, receipt_path, *python_sources(ROOT, [Path(__file__).resolve()])]}
    if (receipt['completed'] != 1536 or plan['planned_steps'] != 3721
            or {p['rank'] for p in receipt['all_four_private_states']} != set(range(4))):
        raise ValueError('The original complete four-worker committed generation is required')
    model, metadata = fresh(plan, 'cpu')
    base_before = tensor_digest(model.base.named_parameters())
    if base_before != plan['frozen_dense_sha256']:
        raise ValueError('The exact original human-policy foundation must load; no replacement base')
    selected = model.named_selected(('actor',)); names = [name for name, _ in selected]
    baseline = None; records = []
    for item in sorted(receipt['all_four_private_states'], key=lambda x: x['rank']):
        path = Path(item['path'])
        if sha(path) != item['sha256'] or item['actual_completed'] != 1536:
            raise ValueError('Every complete original private checkpoint must match its durable receipt')
        files[str(path)] = item['sha256']
        payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        core = payload['core']; cursor = core['cursor']; optimizer = core['optimizer']
        if (payload['settings']['rank'] != item['rank'] or payload['settings']['world'] != 4
                or core['completed'] != 1536 or core['planned'] != 3721
                or core['optimized_names'] != names or core['groups'] != ['actor']
                or cursor['completed_updates'] != 1536 or cursor['planned_updates'] != 3721
                or sorted(cursor['source_order']) != list(range(3721))
                or len(core['metrics']) != 1536 or len(optimizer['state']) != len(selected)
                or core['model']['sources'] != sources(plan, digest)
                or core['model']['frozen_dense_sha256'] != base_before
                or payload['candidate_promoted'] or payload['goal_achieved']):
            raise ValueError('Every original actor partition, plan, source cursor and moment population must reconstruct')
        if len(optimizer['param_groups']) != 1 or optimizer['param_groups'][0]['params'] != list(range(len(selected))):
            raise ValueError('The complete original Adam parameter order must match the actual policy')
        for position, (_, parameter) in enumerate(selected):
            state = optimizer['state'][position]
            if (float(state['step']) != 1536
                    or any(state[k].shape != parameter.shape or not bool(torch.isfinite(state[k]).all())
                           for k in ['exp_avg', 'exp_avg_sq'])
                    or bool((state['exp_avg_sq'] < 0).any())):
                raise ValueError('Every original finite full-shape Adam moment and step must survive')
        for i, metric in enumerate(core['metrics']):
            if (metric['step'] != i+1 or metric['position'] != cursor['source_order'][i]
                    or not metric['original_reference_used_regardless_of_rewards']):
                raise ValueError('Every committed original source-order update must remain complete')
        if baseline is None:
            baseline = payload
            model.import_export(core['model'], expected_sources=sources(plan, digest))
        elif (not equal(baseline['core']['optimizer'], optimizer)
              or not equal(baseline['core']['model'], core['model'])
              or not equal(baseline['core']['metrics'], core['metrics'])):
            raise ValueError('Every common replica model, complete Adam and committed metrics must agree exactly')
        rng = core['rng']; cuda = rng['cuda']
        if cuda is None or cuda.dtype != torch.uint8 or cuda.ndim != 1 or not len(cuda):
            raise ValueError('Every private original CUDA RNG byte record must remain present')
        records.append(dict(rank=item['rank'], checkpoint_sha256=item['sha256'],
            complete_Adam_states=len(optimizer['state']), private_CPU_rng_sha256=hashlib.sha256(rng['torch'].numpy().tobytes()).hexdigest(),
            private_CUDA_rng_sha256=hashlib.sha256(cuda.numpy().tobytes()).hexdigest()))
    prefixes = {(x['external_journal_prefix_bytes'], x['external_journal_prefix_sha256']) for x in receipt['all_four_private_states']}
    if len(prefixes) != 1:
        raise ValueError('Every original private file must bind the same full committed journal prefix')
    length, expected = prefixes.pop(); journal = RUN/'anchor_CE-metrics.jsonl'; hasher = hashlib.sha256()
    with journal.open('rb') as stream:
        for i, metric in enumerate(baseline['core']['metrics']):
            line = stream.readline(); hasher.update(line); row = json.loads(line)
            if (row['step'] != i+1 or row['position'] != metric['position'] or row['source_id'] != metric['source_id']
                    or semantic_digest(row['trajectories']) != metric['global_trajectory_semantic_sha256']):
                raise ValueError('Every committed source and complete trajectory semantic binding must reconstruct')
        if stream.tell() != length or hasher.hexdigest() != expected:
            raise ValueError('The exact complete committed journal prefix must survive unchanged')
    if (torch.cuda.is_initialized() or tensor_digest(model.base.named_parameters()) != base_before
            or len({x['private_CUDA_rng_sha256'] for x in records}) != 4
            or sum(p.numel() for p in model.parameters()) != plan['actual_counts']['total_unique']):
        raise ValueError('Original frozen base, counted policy and all four private RNG records required without CUDA')
    for name, expected_sha in files.items():
        if sha(name) != expected_sha:
            raise ValueError('An original closed source or checkpoint changed during readback')
    result = dict(complete=True, source_files_sha256=files, original_committed_updates=1536, original_planned_updates=3721,
        actual_registered_parameters=sum(p.numel() for p in model.parameters()), original_frozen_dense_sha256=base_before,
        saved_actor_imported_on_original_own_CPU_policy=True, all_four_common_model_Adam_and_metrics_bitwise_identical=True,
        private_states=records, journal=dict(path=str(journal), committed_prefix_bytes=length, committed_prefix_sha256=hasher.hexdigest(),
            uncommitted_tail_bytes_preserved=journal.stat().st_size-length),
        no_source_corpus_or_capacity_requalification=True, no_optimizer_steps_or_neural_forward=True,
        private_CUDA_RNG_bytes_retained_not_executed=True, GPU_resume_admission_still_required=True,
        original_anchor_execution_and_all_final_assessments_still_required=True,
        seconds=time.perf_counter()-start, candidate_promoted=False, goal_achieved=False)
    with OUT.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps({k:v for k,v in result.items() if k not in ['source_files_sha256', 'private_states']}, indent=2))


if __name__ == '__main__':
    main()
