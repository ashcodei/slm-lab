"""Read the closed first GPU arm, all actual sources, Adam and rank cursors.

No neural model, forward, GPU context, new evaluation, retry or promotion.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import traceback

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/first-GPU-repair-arm-complete-readback-CPU-v1'
GPU = ROOT / 'reports/full-function-repair-comparison-GPU-v2'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def main():
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19
            or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'])):
        raise ValueError('CUDA-hidden nice19 one-thread source/state reader required')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    if OUT.exists():
        raise ValueError('Preserve all original readbacks and failed attempts')
    if shutil.disk_usage(ROOT).free < 40 * 1024**3 + 8 * 1024**2:
        raise ValueError('The optional bounded readback must preserve the other writers40GiB reserve')
    OUT.mkdir()
    try:
        import numpy as np
        import torch
        from slm_lab.exact_waiting_process_retirement_v1 import terminal
        from slm_lab.function_repair_curriculum_v1 import FunctionRepairCurriculum
        from slm_lab.parallel_repair_durable_state_v1 import read_latest
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        parents = [r for r in read(GPU / 'children.json') if r['subject'].startswith('train-plain_repeat-')]
        if (len(parents) != 5 or not all(terminal(r) for r in parents)
                or {r['subject'] for r in parents} != {'train-plain_repeat-launcher',
                    *['train-plain_repeat-rank-' + str(r) for r in range(4)]}):
            raise ValueError('The exact first-arm launcher and four actual ranks must be terminal')
        write(OUT / 'terminal-first-arm-owner-observation.json', dict(at=datetime.now(timezone.utc).isoformat(),
            exact_original_owners=parents, every_owner_terminal=True, no_worker_signalled=True))
        completion_path = GPU / 'train-plain_repeat/complete.json'
        original_journal = GPU / 'train-plain_repeat/training-metrics.jsonl'
        complete, plan = read(completion_path), read(GPU / 'plan.json')
        if (not complete['complete'] or complete['actual_complete_updates'] != 306
                or complete['actual_complete_presentations'] != 4851
                or complete['original_response_EOS_targets'] != 2287110):
            raise ValueError('Only the full completed original production arm is eligible')
        raw = original_journal.read_bytes()
        if not raw.endswith(b'\n') or len(raw) > 8 * 1024**2:
            raise ValueError('Every complete original LF-delimited bounded record is required')
        snapshot = OUT / 'all306-original-production-records.jsonl'
        with snapshot.open('xb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        rows = [json.loads(line) for line in raw.split(b'\n')[:-1]]
        data = FunctionRepairCurriculum(ROOT, 'source_preparation')
        entries = plan['all_complete_slot_batches']
        if len(rows) != len(entries) or len(rows) != 306:
            raise ValueError('All actual optimizer records and complete schedule entries are required')
        state, receipt, pointer_sha = read_latest(ROOT / 'checkpoints/full-function-repair-comparison-GPU-v2/train-plain_repeat')
        if (receipt != complete['final_shared_receipt'] or state['metrics'] != rows
                or state['completed'] != 306 or state['planned'] != 306 or state['world_size'] != 4
                or state['profile'] != 'plain_repeat' or state['mode'] != 'train'
                or state['schedule_identity_sha256'] != data.identity(entries)
                or state['schedule_identity_sha256'] != plan['source_schedule_identity_sha256']
                or state['model']['frozen_dense_sha256'] != plan['frozen_dense_sha256']):
            raise ValueError('The independently read actual shared state must match the closed full journal and sources')
        counts = Counter()
        contexts = targets = source_fingerprints = actual_forwards = 0
        for step, (slots, row) in enumerate(zip(entries, rows), 1):
            counts.update(slots)
            expected_fingerprints = []
            for slot in slots:
                position, second = data.slots[slot]
                ids, labels = data.native('plain_repeat', slot)
                positive = [v for v in labels if v != -100]
                expected_fingerprints.append(dict(slot=slot, original_train_position=position, second_view=second,
                    original_source_id=data.original_rows[position]['id'],
                    target_ids_sha256=hashlib.sha256(np.asarray(positive, dtype='<u2').tobytes()).hexdigest()))
                source_fingerprints += 1
            expected_targets = sum(data.targets[p] for p in slots)
            expected_context = sum(len(data.native('plain_repeat', p)[0]) for p in slots)
            if (row['step'] != step or row['profile'] != 'plain_repeat' or row['data_arm'] != 'plain_repeat'
                    or row['code_weighted'] or row['actual_world_size'] != 4 or row['actual_device_type'] != 'cuda'
                    or row['original_presentation_slots'] != slots or row['original_source_fingerprints'] != expected_fingerprints
                    or row['response_EOS_targets'] != expected_targets or row['complete_context_tokens'] != expected_context
                    or row['actual_global_weighted_targets'] != expected_targets
                    or row['actual_rank_source_shards'] != [slots[r::4] for r in range(4)]
                    or not row['every_original_target_and_true_EOS_retained'] or row['neural_teacher_or_reference_queried']
                    or row['learning_rate'] != plan['training_options']['learning_rate']):
                raise ValueError('A complete source identity, original target/mask or actual rank observation differs')
            width = max(1, min(4, 4096 // max(data.lengths[p] for p in slots)))
            observations = row['actual_rank_observations']
            if len(observations) != 4:
                raise ValueError('Every actual rank observation is required')
            for rank, observation in enumerate(observations):
                local = slots[rank::4]
                forwards = math.ceil(len(local) / width)
                if (observation['rank'] != rank or observation['actual_original_slots'] != local
                        or observation['actual_response_EOS_targets'] != sum(data.targets[p] for p in local)
                        or observation['complete_context_tokens'] != sum(len(data.native('plain_repeat', p)[0]) for p in local)
                        or observation['actual_microbatches'] != forwards
                        or observation['actual_precision_per_forward'] != [dict(actual_logits_dtype='torch.bfloat16',
                            autocast_enabled=True, actual_autocast_dtype='torch.bfloat16')] * forwards
                        or observation['no_forward_for_empty_tail'] != bool(not local)):
                    raise ValueError('Every real source shard, precision and empty-rank forward count must reconstruct')
                actual_forwards += forwards
            loss = math.fsum(o['original_CE_sum'] for o in observations)
            if row['original_CE_sum'] != loss or row['original_token_mean_NLL'] != loss / expected_targets:
                raise ValueError('The actual global target-normalized training loss differs')
            contexts += expected_context
            targets += expected_targets
        if counts != Counter({p: 3 for p in range(1617)}) or contexts != 4678500 or targets != 2287110:
            raise ValueError('All826original/791second-view functions and every3epoch budget must be retained')
        adam = state['optimizer']['state']
        if len(adam) != 561 or len(state['optimized_names']) != 561:
            raise ValueError('Every actual selected actor leaf must retain its ordinary Adam state')
        for value in adam.values():
            if (set(value) != {'step', 'exp_avg', 'exp_avg_sq'} or float(value['step']) != 306
                    or not all(bool(torch.isfinite(t).all()) for t in value.values())
                    or bool((value['exp_avg_sq'] < 0).any())):
                raise ValueError('All561actual Adam leaves must be finite with exact production clock306')
        parts = state['model']['parts']
        if not all(bool(torch.isfinite(t).all()) for t in parts.values()):
            raise ValueError('Every actual saved learned actor/unused critic value must be finite')
        if [w['rank'] for w in state['private_workers']] != [0, 1, 2, 3]:
            raise ValueError('All four actual private source/RNG boundaries are required')
        from slm_lab.parallel_function_repair_stage_v1 import value_digest
        if len({value_digest(w['rng']) for w in state['private_workers']}) != 4:
            raise ValueError('Every real rank must retain its distinct private RNG')
        cold = ROOT / 'checkpoints/full-function-repair-comparison-GPU-v2/plain_repeat-actor.pt'
        if (sha(cold) != complete['cold_export_import']['checkpoint_sha256']
                or not complete['cold_export_import']['actual4096text448photo_real_word_outputs_bit_exact']
                or not complete['cold_export_import']['shared_unchanged_frozen_base']):
            raise ValueError('The exact closed actor export and original actual cold-forward proof are required')
        paths = [Path(__file__), GPU / 'plan.json', completion_path, snapshot, Path(receipt['checkpoint']), cold,
            OUT / 'terminal-first-arm-owner-observation.json', *[Path(p) for p in data.source_files_sha256]]
        bindings = {str(p): sha(p) for p in paths}
        if sha(original_journal) != sha(snapshot) or torch.cuda.is_initialized():
            raise ValueError('The same closed whole journal and zero CUDA contexts must be preserved')
        write(OUT / 'complete.json', dict(complete=True, passed=True, at=datetime.now(timezone.utc).isoformat(),
            source_files_sha256=bindings, actual_optimizer_updates=306, actual_complete_presentations=4851,
            actual_complete_context_tokens=contexts, actual_original_response_EOS_targets=targets,
            actual_original_source_fingerprints_rebuilt=source_fingerprints,
            actual_four_rank_source_delivery_records=4 * len(rows), actual_BF16_neural_forwards_read=actual_forwards,
            actual_ordinary_Adam_leaves_at_exact_clock306=len(adam), all_saved_actor_and_critic_values_finite=True,
            actual_saved_actor_and_critic_values=sum(t.numel() for t in parts.values()),
            all_four_distinct_private_RNGs_and_shared_semantic_manifest_verified=True,
            actual_final_checkpoint_sha256=receipt['checkpoint_sha256'], external_pointer_sha256=pointer_sha,
            original4096text448photo_real_word_cold_import_proof_and_exact_export_retained=True,
            newly_executed_neural_forwards=0, new_GPU_contexts=0, new_accuracy_result=False,
            all_original_sources_and_other_arms_and_full_evaluations_required=True, goal_achieved=False))
        print(json.dumps(dict(complete=True, actual_updates=306, actual_original_targets=targets, new_accuracy_result=False)))
    except BaseException:
        write(OUT / 'failure.json', dict(complete=False, traceback=traceback.format_exc(), goal_achieved=False))
        raise


if __name__ == '__main__':
    main()
