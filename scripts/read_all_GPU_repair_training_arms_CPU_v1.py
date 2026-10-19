"""Independently rebuild all three closed GPU source/target/Adam histories.

No model construction, neural forward, GPUcontext, re-execution or score
selection. Copy complete LF-delimited closed journals, retain every outcome.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import traceback

ROOT = Path(__file__).resolve().parents[1]
GPU = ROOT / 'reports/full-function-repair-comparison-GPU-v2'
DEST = ROOT / 'checkpoints/full-function-repair-comparison-GPU-v2'
OUT = ROOT / 'reports/all-three-GPU-repair-training-readback-CPU-v1'
PROFILES = ['plain_repeat', 'semantic_repair', 'weighted_semantic_repair']


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
    if (OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19
            or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'])):
        raise ValueError('Fresh CUDA-hidden nice19 one-thread source/state reader required')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    from slm_lab.bounded_disk_reserve_wait_v1 import wait_for_reserve
    from slm_lab.exact_waiting_process_retirement_v1 import terminal
    wait_for_reserve(ROOT, minimum_bytes=40 * 1024**3)
    all_done = read(GPU / 'all-arms-training-complete.json')
    if not all_done['complete'] or list(all_done['profiles']) != PROFILES:
        raise ValueError('Every original full production arm must finish before this readback')
    observed = []
    for profile in PROFILES:
        workers = [r for r in read(GPU / 'children.json') if r['subject'].startswith('train-' + profile + '-')]
        if len(workers) != 5 or not all(terminal(r) for r in workers):
            raise ValueError('The exact closed arm launcher and all four ranks must be terminal')
        observed.extend(workers)
    OUT.mkdir()
    write(OUT / 'all15-terminal-production-owner-observations.json', dict(at=datetime.now(timezone.utc).isoformat(),
        exact_owners=observed, all_terminal=True, no_worker_signalled=True))
    try:
        import numpy as np
        import torch
        from slm_lab.function_repair_curriculum_v1 import FunctionRepairCurriculum
        from slm_lab.full_function_repair_stage_v1 import CodeWeights
        from slm_lab.parallel_repair_durable_state_v1 import read_latest
        from slm_lab.parallel_function_repair_stage_v1 import value_digest
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        plan = read(GPU / 'plan.json')
        entries = plan['all_complete_slot_batches']
        data = FunctionRepairCurriculum(ROOT, 'source_preparation')
        weights = CodeWeights(data)
        baseline = torch.load(DEST / 'baseline-actor.pt', map_location='cpu', weights_only=True, mmap=True)
        if len(entries) != 306 or Counter(p for e in entries for p in e) != Counter({p: 3 for p in range(1617)}):
            raise ValueError('All826original/791second views must occur in all three complete source epochs')
        bindings = {**data.source_files_sha256, **weights.source_files_sha256}
        paths = [Path(__file__), GPU / 'plan.json', GPU / 'all-arms-training-complete.json', DEST / 'baseline-actor.pt',
            OUT / 'all15-terminal-production-owner-observations.json']
        results = {}
        for profile in PROFILES:
            arm = 'plain_repeat' if profile == 'plain_repeat' else 'semantic_repair'
            weighted = profile == 'weighted_semantic_repair'
            complete_path = GPU / ('train-' + profile) / 'complete.json'
            complete = read(complete_path)
            original = GPU / ('train-' + profile) / 'training-metrics.jsonl'
            raw = original.read_bytes()
            if not raw.endswith(b'\n') or len(raw) > 64 * 1024**2:
                raise ValueError('Every complete original bounded LF-delimited production record is required')
            snapshot = OUT / (profile + '-all306-original-records.jsonl')
            with snapshot.open('xb') as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            rows = [json.loads(line) for line in raw.split(b'\n')[:-1]]
            state, receipt, pointer = read_latest(DEST / ('train-' + profile))
            actor_path = DEST / (profile + '-actor.pt')
            actor = torch.load(actor_path, map_location='cpu', weights_only=True, mmap=True)
            if (len(rows) != 306 or state['metrics'] != rows or receipt != complete['final_shared_receipt']
                    or state['completed'] != state['planned'] or state['completed'] != 306
                    or state['profile'] != profile or state['world_size'] != 4 or state['mode'] != 'train'
                    or state['schedule_identity_sha256'] != data.identity(entries)
                    or state['model']['frozen_dense_sha256'] != plan['frozen_dense_sha256']
                    or actor['frozen_dense_sha256'] != plan['frozen_dense_sha256']
                    or actor['actual_counts'] != plan['actual_student_counts']
                    or any(not torch.equal(t, actor['parts'][n]) for n, t in state['model']['parts'].items())):
                raise ValueError('The closed source history, actual shared state and cold actor export must agree for every arm')
            epoch = [dict(targets=0, contexts=0, weighted_targets=0., original_CE=0., weighted_CE=0.) for _ in range(3)]
            forwards = 0
            for step, (slots, row) in enumerate(zip(entries, rows), 1):
                expected_targets = expected_contexts = 0
                denominator = 0.
                fingerprints = []
                for slot in slots:
                    position, second = data.slots[slot]
                    ids, labels = data.native(arm, slot)
                    positive = [v for v in labels if v != -100]
                    expected_targets += len(positive)
                    expected_contexts += len(ids)
                    denominator += math.fsum(map(float, weights.for_slot(data, slot))) if weighted else len(positive)
                    fingerprints.append(dict(slot=slot, original_train_position=position, second_view=second,
                        original_source_id=data.original_rows[position]['id'],
                        target_ids_sha256=hashlib.sha256(np.asarray(positive, dtype='<u2').tobytes()).hexdigest()))
                if (row['step'] != step or row['profile'] != profile or row['data_arm'] != arm or row['code_weighted'] is not weighted
                        or row['original_presentation_slots'] != slots or row['original_source_fingerprints'] != fingerprints
                        or row['response_EOS_targets'] != expected_targets or row['complete_context_tokens'] != expected_contexts
                        or row['actual_global_weighted_targets'] != denominator
                        or row['actual_device_type'] != 'cuda' or row['actual_world_size'] != 4
                        or row['actual_rank_source_shards'] != [slots[r::4] for r in range(4)]
                        or row['every_original_target_and_true_EOS_retained'] is not True
                        or row['neural_teacher_or_reference_queried'] is not False):
                    raise ValueError('Every actual original source, target, positive weight and real rank shard must reconstruct')
                width = max(1, min(4, 4096 // max(data.lengths[p] for p in slots)))
                observations = row['actual_rank_observations']
                for rank, observation in enumerate(observations):
                    local = slots[rank::4]
                    calls = math.ceil(len(local) / width)
                    if (observation['rank'] != rank or observation['actual_original_slots'] != local
                            or observation['actual_response_EOS_targets'] != sum(data.targets[p] for p in local)
                            or observation['complete_context_tokens'] != sum(len(data.native(arm, p)[0]) for p in local)
                            or observation['actual_microbatches'] != calls
                            or observation['actual_precision_per_forward'] != [dict(actual_logits_dtype='torch.bfloat16',
                                autocast_enabled=True, actual_autocast_dtype='torch.bfloat16')] * calls):
                        raise ValueError('Every actual BF16forward and whole rank source budget must match')
                    forwards += calls
                total = math.fsum(v['original_CE_sum'] for v in observations)
                weighted_total = math.fsum(v['weighted_CE_sum'] for v in observations)
                if (len(observations) != 4 or row['original_CE_sum'] != total or row['weighted_CE_sum'] != weighted_total
                        or row['original_token_mean_NLL'] != total / expected_targets
                        or row['weighted_token_mean_NLL'] != weighted_total / denominator):
                    raise ValueError('Every complete actual global CE numerator and declared denominator must agree')
                item = epoch[(step - 1) // 102]
                for key, value in [('targets', expected_targets), ('contexts', expected_contexts), ('weighted_targets', denominator),
                        ('original_CE', total), ('weighted_CE', weighted_total)]:
                    item[key] += value
            if (sum(e['targets'] for e in epoch) != 2287110
                    or sum(e['contexts'] for e in epoch) != (4678500 if arm == 'plain_repeat' else 5416461)
                    or any(e['targets'] != 762370 for e in epoch)):
                raise ValueError('Every source epoch and its original complete response/EOS budget must be preserved')
            adam = state['optimizer']['state']
            if len(adam) != 561 or len(state['optimized_names']) != 561:
                raise ValueError('All561actual ordinary selected actor Adam leaves are required')
            for value in adam.values():
                if (set(value) != {'step', 'exp_avg', 'exp_avg_sq'} or float(value['step']) != 306
                        or any(not bool(torch.isfinite(t).all()) for t in value.values()) or bool((value['exp_avg_sq'] < 0).any())):
                    raise ValueError('Every actual actor Adam leaf must be finite with exact production clock306')
            if (any(not bool(torch.isfinite(t).all()) for t in actor['parts'].values())
                    or not torch.equal(actor['parts']['critic.weight'], baseline['parts']['critic.weight'])
                    or len({value_digest(w['rng']) for w in state['private_workers']}) != 4):
                raise ValueError('Every actor value, unchanged critic and four distinct private RNGs must be preserved')
            cold = complete['cold_export_import']
            if (sha(actor_path) != cold['checkpoint_sha256'] or cold['actual4096text448photo_real_word_outputs_bit_exact'] is not True
                    or cold['shared_unchanged_frozen_base'] is not True):
                raise ValueError('Every original actual complete cold proof and exact actor export must match')
            results[profile] = dict(actual_complete_updates=306, actual_source_presentations=4851,
                actual_original_response_EOS_targets=2287110, actual_source_validation_gradients=0,
                actual_global_weighted_targets=sum(e['weighted_targets'] for e in epoch),
                actual_complete_context_tokens=sum(e['contexts'] for e in epoch), actual_Adam_leaves_at_clock306=561,
                actual_BF16forwards_read=forwards, actual_actor_tensors_changed_from_common_initializer=sum(
                    not torch.equal(t, baseline['parts'][n]) for n, t in actor['parts'].items()),
                original_same_critic_preserved=True, four_distinct_private_RNGs_and_manifest_verified=True,
                actual_export_sha256=sha(actor_path), actual_shared_state_sha256=receipt['checkpoint_sha256'],
                actual_external_pointer_sha256=pointer, actual_full_native_cold_proof_retained=True,
                online_source_epoch_metrics=[dict(source_epoch=i + 1, response_EOS_targets=e['targets'],
                    original_token_mean_NLL=e['original_CE'] / e['targets'],
                    weighted_token_mean_NLL=e['weighted_CE'] / e['weighted_targets']) for i, e in enumerate(epoch)],
                online_training_loss_is_not_held_accuracy=True)
            paths.extend([complete_path, snapshot, Path(receipt['checkpoint']), actor_path])
            if sha(original) != sha(snapshot):
                raise ValueError('The same complete closed original journal must remain unchanged')
            del state, actor, rows
        bindings.update({str(p): sha(p) for p in paths})
        if torch.cuda.is_initialized():
            raise ValueError('This full source/state reader must never create a CUDAcontext')
        write(OUT / 'complete.json', dict(complete=True, passed=True, at=datetime.now(timezone.utc).isoformat(),
            source_files_sha256=bindings, actual_profiles=results, actual_total_separate_arm_updates=918,
            actual_same_initializer_each_arm=True, every_original826function791second_view_three_epoch_budget_rebuilt=True,
            all_original_positive_EOS_targets_and_third_arm_code_weights_retained=True,
            all15actual_production_owner_births_terminal=True, new_models_constructed=0,
            new_neural_forwards=0, new_GPUcontexts=0, new_accuracy_result=False,
            original_full_matched_native_assessments_still_required=True, candidate_promoted=False, goal_achieved=False))
        print(json.dumps(dict(complete=True, profiles=PROFILES, actual_separate_arm_updates=918, new_accuracy_result=False)))
    except BaseException:
        write(OUT / 'failure.json', dict(complete=False, traceback=traceback.format_exc(), goal_achieved=False))
        raise


if __name__ == '__main__':
    main()
