"""Fresh complete pack readback and weak-SFT-only source admission.

Preserve all original source decisions and roles. Rebuild native arrays from
the pinned original question/response, not the producer token arrays. No new
program execution, model weights, neural teacher API or GPU lease is used.
"""
from collections import Counter
from datetime import datetime, timezone
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import struct
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/public-weak-code-pack-readback-admission-CPU-v1'
PACK = ROOT / 'data/public-weak-code-SFT-v1'
PRODUCER = ROOT / 'reports/public-weak-code-pack-CPU-v1'
SOURCE = ROOT / 'data/public-code-instruction-candidate-v1'
PRIOR = ROOT / 'reports/full-public-code-shard-qualification-CPU-v1'
FRESH = ROOT / 'reports/full-public-code-shard-readback-CPU-v1'
VETO = ROOT / 'reports/public-code-instruction-precision-counterexample-CPU-v1/semantic-veto.json'
RESERVE = 40 * 1024**3


def jsonl(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rb') as stream:
        for raw in stream:
            if not raw.endswith(b'\n'): raise ValueError('Every actual closed source record must retain its complete byte-LF boundary')
            yield json.loads(raw)


def main(parent):
    if (OUT.exists() or (PACK / 'admission.json').exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19
            or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'])):
        raise ValueError('A fresh CUDA-hidden nice19 one-thread source-only weak-SFT reader is required')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    from record_visual_stage_and_first_repair_arm_v1 import atomic, read, sha
    from slm_lab.exact_waiting_process_retirement_v1 import terminal
    from slm_lab.local_source_closure_v2 import python_sources
    if read(PRODUCER / 'plan.json')['owner'] != parent: raise ValueError('Exact original weak source pack producer birth required')
    tests = ROOT / 'reports/public-weak-code-pack-rows-v2-tests.log'
    if '8 passed' not in tests.read_text(): raise ValueError('Every actual whole weak-source/native/role control must pass')
    paths = [Path(__file__), PRODUCER / 'plan.json', tests, ROOT / 'src/slm_lab/public_weak_code_train_data_v1.py',
        *python_sources(ROOT, [Path(__file__), ROOT / 'tests/test_public_weak_code_pack_rows_v2.py'])]
    files = {str(p): sha(p) for p in paths}
    states = {p: (Path(p).stat().st_ino, Path(p).stat().st_size, Path(p).stat().st_mtime_ns) for p in files}
    def waiting_guard():
        if os.sched_getscheduler(0) != os.SCHED_IDLE or any(
                (Path(p).stat().st_ino, Path(p).stat().st_size, Path(p).stat().st_mtime_ns) != s for p, s in states.items()):
            raise ValueError('Every immutable original source/readback input and idle context must remain')
    owner = dict(pid=os.getpid(), ticks=Path('/proc/self/stat').read_text().split(') ', 1)[1].split()[19])
    OUT.mkdir()
    atomic(OUT / 'plan.json', dict(owner=owner, exact_original_source_pack_owner=parent, source_files_sha256=files,
        at=datetime.now(timezone.utc).isoformat(), all_original_source_decisions=100000,
        complete_expected_native_rows=24248, original_positive_train_rows=21739, separately_reserved_rows=2509,
        full_original_question_response_EOS_masks_independently_rebuilt=True,
        every_failed_source_and_original_role_retained=True,
        source_teacher='Public precomputed Qwen2.5-Coder-32B-Instruct responses and tests',
        admitted_use_if_passed='weak_supervised_coding_only', human_preferences=False, own_model_on_policy=False,
        independent_semantic_correctness_certified=False, use_as_RL_reward_oracle_forbidden=True,
        maximum_new_report_bytes=128 * 1024**2, minimum_other_writer_free_bytes=RESERVE,
        new_program_executions=0, new_neural_models=0, GPUleases=0, production_updates=0,
        training_admitted=False, goal_achieved=False))
    try:
        deadline = time.monotonic() + 7 * 86400
        while not terminal(parent):
            waiting_guard()
            if time.monotonic() > deadline: raise ValueError('The source-only full weak pack readback lifetime expired')
            atomic(OUT / 'progress.json', dict(owner=owner, phase='waiting for exact complete weak source pack producer',
                exact_original_producer_live=True, training_admitted=False, new_neural_models=0))
            time.sleep(15)
        if (PRODUCER / 'failure.json').exists() or not (PRODUCER / 'complete.json').exists():
            raise ValueError('The exact original weak source pack producer did not complete; preserve partial artifacts')
        source = read(PRODUCER / 'complete.json')
        if (not source['complete'] or source['owner'] != parent or source['training_admitted'] is not False
                or source['complete_original_source_positions'] != 100000 or source['complete_native_positive_rows'] != 24248
                or sha(PACK / 'manifest.json') != source['dataset_manifest_sha256']
                or sha(PRODUCER / 'all100000-pack-dispositions.jsonl') != source['all_original_decisions_sha256']):
            raise ValueError('Every complete source/pack/role decision byte must match the exact producer')
        import pyarrow.parquet as pq
        from tokenizers import Tokenizer
        from slm_lab.active_execution_input_bindings_v1 import merge_bindings
        from slm_lab.bounded_disk_reserve_wait_v1 import wait_for_reserve
        from slm_lab.codenet_checkpoint_v1 import FileBindings
        from slm_lab.codenet_token_pack_v1 import CompleteCodePack, FILES
        from slm_lab.public_code_candidate_roles_v1 import verify_role_separation
        from slm_lab.public_code_instruction_screen_v1 import parse_whole
        from slm_lab.public_synthetic_code_screen_policy_v1 import body_fingerprint
        from slm_lab.public_weak_code_pack_rows_v2 import build, ATTRIBUTION, REVISION
        merge_bindings(files, source['source_files_sha256'])
        files.update({str(p): sha(p) for p in [PRODUCER / 'complete.json', PRODUCER / 'all100000-pack-dispositions.jsonl',
            OUT / 'plan.json', PACK / 'manifest.json', *[PACK / n for n in FILES]]})
        guard = FileBindings(files)
        guard.check(full=True)
        def reserve():
            wait_for_reserve(ROOT, minimum_bytes=RESERVE + 16 * 1024**2, guard=guard.check)
            if os.sched_getscheduler(0) != os.SCHED_IDLE: raise ValueError('All source readback must remain idle')
        reserve()
        packs = {role: CompleteCodePack(PACK, role, purpose='audit') for role in ['train', 'validation']}
        if len(packs['train']) != 21739 or len(packs['validation']) != 2509:
            raise ValueError('All exact original positive train and separately reserved populations required')
        actual_rows = iter(jsonl(PACK / 'original-rows.jsonl.gz'))
        decisions = iter(jsonl(PRODUCER / 'all100000-pack-dispositions.jsonl'))
        prior_rows, fresh_rows = iter(jsonl(PRIOR / 'all100000-original-dispositions.jsonl')), iter(jsonl(FRESH / 'all100000-fresh-readback-observations.jsonl'))
        roles = {r['original_shard_position']: r for r in jsonl(FRESH / 'all-original-candidate-roles.jsonl')}
        tokenizer = Tokenizer.from_file(str(PACK / 'tokenizer.json'))
        veto = read(VETO)
        first_batch = next(pq.ParquetFile(SOURCE / 'train-00000-of-00050.parquet').iter_batches(
            batch_size=veto['original_shard_position'] + 1, use_threads=False)).to_pylist()
        bad_AST = body_fingerprint(parse_whole(first_batch[veto['original_shard_position']])['program'])
        del first_batch
        counts, cursors = Counter(), Counter()
        candidates = []
        position = packed = 0
        output = OUT / 'all100000-pack-readback-observations.jsonl'
        with output.open('x') as journal:
            for batch in pq.ParquetFile(SOURCE / 'train-00000-of-00050.parquet').iter_batches(batch_size=256, use_threads=False):
                for original in batch.to_pylist():
                    if position % 128 == 0: reserve()
                    prior, fresh, decision = next(prior_rows), next(fresh_rows), next(decisions)
                    if prior['screen_pass_candidate']: candidates.append(prior)
                    value, status = build(original, prior, fresh, roles.get(position), tokenizer,
                        veto_sha256=veto['original_entry_sha256'], veto_AST_sha256=bad_AST)
                    if (decision['original_shard_position'] != position or decision['source_id'] != original.get('id')
                            or decision['pack_disposition'] != status or decision['original_fixed_role'] != fresh['source_role']
                            or decision['weak_SFT_only'] is not (value is not None)
                            or decision['packed_position'] != (packed if value is not None else None)):
                        raise ValueError('Every original source decision and unchanged preassigned role must independently reconstruct')
                    if value is not None:
                        row, _, _ = value
                        # Independently reconstruct all native arrays, rather than
                        # consuming or trusting the producer formatter's arrays.
                        prompt = tokenizer.encode('User: ' + original['input'] + '\nAssistant:', add_special_tokens=False).ids
                        answer = tokenizer.encode(original['output'], add_special_tokens=False).ids
                        expected = [1] + prompt + answer + [2]
                        boundary = len(prompt) + 1
                        labels = [-100] * boundary + expected[boundary:]
                        role = row['split']
                        actual_ids, actual_labels = packs[role].sequence(cursors[role])
                        actual = next(actual_rows)
                        row.update(token_pack_index=packed, complete_input_ids_sha256=hashlib.sha256(
                            struct.pack('<' + 'H' * len(expected), *expected)).hexdigest(), target_tokens=len(answer) + 1)
                        if (actual_ids != expected or actual_labels != labels or actual != row
                                or len(expected) > 4096 or expected[-1] != 2 or labels[-1] != 2):
                            raise ValueError('Every actual original question/whole response/EOS/native index/label must independently match')
                        cursors[role] += 1
                        packed += 1
                        counts[role + '_examples'] += 1
                        counts[role + '_context_tokens'] += len(expected)
                        counts[role + '_target_tokens'] += len(answer) + 1
                    counts[status] += 1
                    counts['all_original_source_decisions'] += 1
                    raw = json.dumps(dict(original_shard_position=position, original_source_id=original.get('id'),
                        original_role=fresh['source_role'], exact_original_decision_rebuilt=True, source_disposition=status,
                        complete_native_row_rebuilt=value is not None, weak_SFT_only=True, generated_tests_not_independent_oracle=True),
                        ensure_ascii=True, separators=(',', ':')) + '\n'
                    if journal.tell() + len(raw.encode()) > 128 * 1024**2: raise ValueError('The entire source readback must fit its declared report bound')
                    journal.write(raw)
                    position += 1
                    if position % 1000 == 0:
                        journal.flush()
                        atomic(OUT / 'progress.json', dict(owner=owner, phase='rebuild every original decision and complete actual weak-SFT pack row',
                            actual_source_decisions=position, actual_native_rows=packed, actual_counts=dict(counts), training_admitted=False))
            journal.flush()
            os.fsync(journal.fileno())
        if (position != 100000 or packed != 24248 or cursors != dict(train=21739, validation=2509)
                or any(next(iterator, None) is not None for iterator in [actual_rows, decisions, prior_rows, fresh_rows])):
            raise ValueError('Every original source position, positive row and whole source role must finish exactly')
        summary = verify_role_separation(candidates, {p: {k: v for k, v in r.items() if k != 'original_shard_position'} for p, r in roles.items()})
        if summary != source['actual_fixed_role_summary']: raise ValueError('All original duplicate components and fixed roles must remain')
        manifest = read(PACK / 'manifest.json')
        for role in ['train', 'validation']:
            if any(counts[role + '_' + key] != manifest['counts'][role][key] for key in ['examples', 'context_tokens', 'target_tokens']):
                raise ValueError('Every original complete native role/target total must independently reconstruct')
        guard.check(full=True)
        evidence = OUT / 'independent-complete-source-readback.json'
        atomic(evidence, dict(complete=True, owner=owner, at=datetime.now(timezone.utc).isoformat(), source_files_sha256=files,
            actual_counts=dict(counts), actual_native_rows=24248, actual_original_decisions=100000,
            actual_fixed_role_summary=summary, all_original_native_arrays_masks_EOS_and_source_records_match=True,
            actual_all_source_readback_observations_sha256=sha(output), new_program_executions=0,
            source_semantic_correctness_not_certified=True, use_as_RL_reward_oracle_forbidden=True, new_neural_models=0, goal_achieved=False))
        files.update({str(evidence): sha(evidence), str(output): sha(output)})
        reserve()
        admission = dict(admitted_to_training=True, admitted_use='weak_supervised_coding_only', owner=owner,
            at=datetime.now(timezone.utc).isoformat(), source_files_sha256=files, dataset_manifest_sha256=sha(PACK / 'manifest.json'),
            independent_readback_sha256=sha(evidence), complete_original_source_decisions=100000,
            complete_packed_rows_read_back=24248, positive_train_rows=21739, separately_reserved_rows=2509,
            actual_complete_pack_counts=manifest['counts'], synthetic_teacher_data=True,
            response_and_test_origin='Public precomputed Qwen2.5-Coder-32B-Instruct',
            attribution=ATTRIBUTION, source_revision=REVISION, license='CC-BY-4.0', human_preferences=False,
            own_model_on_policy=False, independent_semantic_correctness_certified=False,
            use_as_RL_reward_oracle_forbidden=True, original_reserved_roles_never_supplied_to_optimization=True,
            failed_or_abstaining_source_rows_and_decisions_preserved=True,
            runtime_or_GPU_capacity_or_model_quality_not_admitted=True, new_program_executions=0,
            new_neural_models=0, new_production_updates=0, goal_achieved=False)
        atomic(PACK / 'admission.json', admission)
        atomic(OUT / 'complete.json', dict(complete=True, owner=owner, data_only_weak_SFT_admitted=True,
            admission_sha256=sha(PACK / 'admission.json'), independent_readback_sha256=sha(evidence),
            actual_source_decisions=100000, actual_native_rows=24248, positive_train_rows=21739, separately_reserved_rows=2509,
            generated_tests_not_RL_oracle=True, new_neural_models=0, new_program_executions=0,
            full_size_GPUcapacity_training_and_quality_still_pending=True, goal_achieved=False))
        print(str(PACK / 'admission.json'), flush=True)
    except BaseException as exc:
        atomic(OUT / 'failure.json', dict(owner=owner, error=repr(exc), automatic_retry=False,
            all_original_and_partial_source_artifacts_retained=True, source_admission_published=(PACK / 'admission.json').exists(),
            no_model_or_GPUjob_started=True, goal_achieved=False))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent-pid', type=int, required=True)
    parser.add_argument('--parent-ticks', required=True)
    args = parser.parse_args()
    main(dict(pid=args.parent_pid, ticks=args.parent_ticks))
