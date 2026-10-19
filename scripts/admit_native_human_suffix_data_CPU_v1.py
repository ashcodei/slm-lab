"""Seal all original native source views after the exact complete fresh reader.

Build an indexed lossless transport, freshly read every packed record and
publish data-only admission. Source/role/control failures remain in the pack
as CE-only presentations. No model or GPU runtime is admitted or launched.
"""
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/native-human-code-suffix-data-admission-CPU-v1'
DATA = ROOT / 'data/native-human-code-suffix-v1'
SOURCE = ROOT / 'reports/full-native-token-suffix-qualification-CPU-v1'
READER = ROOT / 'reports/full-native-token-suffix-readback-CPU-v1'
PARENT = dict(pid=166298, ticks='56079802')
RESERVE = 40 * 1024**3
MAX_RAW = 2 * 1024**3


def main():
    if (OUT.exists() or DATA.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19
            or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'])):
        raise ValueError('Fresh CUDA-hidden nice19 one-thread source-only admission required')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    from record_visual_stage_and_first_repair_arm_v1 import atomic, read, sha
    from slm_lab.exact_waiting_process_retirement_v1 import terminal
    from slm_lab.local_source_closure_v2 import python_sources
    if read(READER / 'plan.json')['owner'] != PARENT: raise ValueError('Exact full native reader birth required')
    tests = ROOT / 'reports/native-suffix-indexed-data-v1-tests.log'
    if '8 passed' not in tests.read_text(): raise ValueError('All actual indexed transport/data/complete-source schedule tests must pass')
    seeds = [Path(__file__), ROOT / 'tests/test_native_suffix_indexed_data_v1.py']
    files = {str(p): sha(p) for p in [READER / 'plan.json', tests, *python_sources(ROOT, seeds)]}
    states = {p: (Path(p).stat().st_ino, Path(p).stat().st_size, Path(p).stat().st_mtime_ns) for p in files}
    owner = dict(pid=os.getpid(), ticks=Path('/proc/self/stat').read_text().split(') ', 1)[1].split()[19])
    OUT.mkdir()
    atomic(OUT / 'plan.json', dict(owner=owner, exact_original_full_native_reader=PARENT, source_files_sha256=files,
        at=datetime.now(timezone.utc).isoformat(), all_original_sources=3721, all_original_native_views=14884,
        original_context_tokens_per_level=2287522, original_response_EOS_targets_per_level=584218,
        original_levels=[4,16,64,0], all_original_failed_trivial_abstaining_disagreeing_CE_views_retained=True,
        full_single_epoch_source_presentations=14884, full_single_epoch_global_steps=3724,
        actual_single_source_three_empty_rank_training_steps_per_epoch=4,
        maximum_additional_raw_bytes=MAX_RAW, minimum_other_writer_reserve_bytes=RESERVE,
        data_only_admission_cannot_admit_full_GPUruntime=True, models_constructed=0, neural_forwards=0,
        GPUleases=0, optimizer_updates=0, goal_achieved=False))
    def waiting_guard():
        if os.sched_getscheduler(0) != os.SCHED_IDLE or any(
                (Path(p).stat().st_ino, Path(p).stat().st_size, Path(p).stat().st_mtime_ns) != s for p, s in states.items()):
            raise ValueError('Every immutable original source and idle source-only context must remain')
    try:
        deadline = time.monotonic() + 7 * 86400
        while not terminal(PARENT):
            waiting_guard()
            if time.monotonic() > deadline: raise ValueError('The source-only complete-admission waiting deadline expired')
            atomic(OUT / 'progress.json', dict(owner=owner, phase='waiting for exact complete native-profile independent reader',
                exact_original_reader_live=True, source_training_admitted=False, models_constructed=0))
            time.sleep(15)
        if (READER / 'failure.json').exists() or not (READER / 'complete.json').exists():
            raise ValueError('The exact full native reader did not finish; preserve every failed or partial observation')
        source, reader = read(SOURCE / 'complete.json'), read(READER / 'complete.json')
        if (not source['complete'] or not reader['complete'] or not reader['integrity_readback_passed']
                or reader['owner'] != PARENT or reader['source_training_admitted'] is not False
                or source['actual_counts']['all_original_sources'] != 3721 or source['actual_counts']['all_original_views'] != 14884
                or reader['actual_counts']['all_original_sources'] != 3721 or reader['actual_counts']['all_original_views_rebuilt'] != 14884
                or not reader['all_original_sources_views_full_CE_masks_and_failed_dispositions_retained']):
            raise ValueError('The entire original native producer and actual fresh source reader must remain intact')
        import torch
        from slm_lab.active_execution_input_bindings_v1 import merge_bindings
        from slm_lab.bounded_disk_reserve_wait_v1 import wait_for_reserve
        from slm_lab.bounded_gzip_observation_stream_v1 import lines, NEXT_RECORD
        from slm_lab.codenet_checkpoint_v1 import FileBindings
        from slm_lab.indexed_gzip_records_v1 import Writer, Records
        from slm_lab.native_suffix_train_data_v1 import FORMAT, SCOPE, schedule_source_views
        from slm_lab.parallel_function_repair_stage_v1 import value_digest
        torch.set_num_threads(1)
        for report in [source, reader]: merge_bindings(files, report['source_files_sha256'])
        source_path, reader_path = Path(source['complete_control_file']), Path(reader['complete_control_file'])
        if sha(source_path) != source['complete_control_file_sha256'] or sha(reader_path) != reader['complete_control_file_sha256']:
            raise ValueError('Every closed original native control byte must match before data packing')
        files.update({str(p): sha(p) for p in [SOURCE / 'complete.json', READER / 'complete.json',
            source_path, reader_path, OUT / 'plan.json']})
        guard = FileBindings(files)
        guard.check(full=True)
        def resources():
            guard.check()
            if os.sched_getscheduler(0) != os.SCHED_IDLE or torch.cuda.is_initialized():
                raise ValueError('Only idle source packing without any CUDAcontext is allowed')
        wait_for_reserve(ROOT, minimum_bytes=RESERVE + NEXT_RECORD, guard=resources)
        DATA.mkdir()
        pack, index = DATA / 'native-views.records.gz', DATA / 'native-views.index'
        counts = Counter()
        totals = {str(level): Counter() for level in [4,16,64,0]}
        def aligned(position, original, fresh):
            source_position, level = position // 4, [4,16,64,0][position % 4]
            proof = original['native_profile_control_proof']
            if (original['source_position'] != source_position or fresh['source_position'] != source_position
                    or original['level'] != level or fresh['level'] != level or original['corpus_id'] != fresh['corpus_id']
                    or fresh['actual_closed_producer_proof_sha256'] != value_digest(proof)
                    or fresh['independent_native_descriptor'] != proof['native_view']
                    or original['all_original_unconditional_CE_targets_retained'] is not True
                    or fresh['all_original_unconditional_CE_targets_retained'] is not True):
                raise ValueError('Every full original native source/view/proof/mask and independent result must align')
            return dict(position=position, source_position=source_position, level=level, producer=original, reader=fresh)
        original_rows, fresh_rows = iter(lines(source_path)), iter(lines(reader_path))
        with Writer(pack, index, root=ROOT, maximum_raw_bytes=MAX_RAW, minimum_free_bytes=RESERVE, guard=resources) as writer:
            for position in range(14884):
                original, fresh = next(original_rows)[1], next(fresh_rows)[1]
                record = aligned(position, original, fresh)
                if writer.append(record) != position: raise ValueError('Every original packed record offset must retain its native order')
                desc = fresh['independent_native_descriptor']
                counts['all_original_views'] += 1
                counts['native_execution_eligible_views'] += int(fresh['native_profile_eligible_in_both_readers'])
                counts['CE_only_views'] += int(not fresh['native_profile_eligible_in_both_readers'])
                totals[str(record['level'])].update(views=1, original_context_tokens=desc['original_context_tokens'],
                    original_response_EOS_targets=desc['original_response_EOS_targets'], full_native_action_budget=desc['full_native_action_budget'])
                if (position + 1) % 64 == 0 or position + 1 == 14884:
                    writer.fsync()
                    atomic(OUT / 'progress.json', dict(owner=owner, phase='pack every complete original native control and fresh result',
                        actual_counts=dict(counts), source_training_admitted=False, models_constructed=0))
            if next(original_rows, None) is not None or next(fresh_rows, None) is not None:
                raise ValueError('Complete original source journals must end at exactly14884views')
        # Read every actual compressed member through the downstream random-access
        # implementation, comparing against both complete original source streams.
        original_rows, fresh_rows = iter(lines(source_path)), iter(lines(reader_path))
        with Records(pack, index, count=14884) as records:
            for position in range(14884):
                expected = aligned(position, next(original_rows)[1], next(fresh_rows)[1])
                if records.read(position)[1] != expected:
                    raise ValueError('Every freshly decompressed indexed record must equal the complete original proof/result')
                counts['actual_fresh_indexed_records_read_back'] += 1
        if (counts['all_original_views'] != 14884 or counts['actual_fresh_indexed_records_read_back'] != 14884
                or any(v['views'] != 3721 or v['original_context_tokens'] != 2287522
                    or v['original_response_EOS_targets'] != 584218 for v in totals.values())):
            raise ValueError('Every source/view/full CEtarget and actual packed readback is required')
        entries = schedule_source_views(epochs=1, width=4, seed=989711)
        if len(entries) != 3724 or sorted(p for e in entries for p in e) != list(range(14884)):
            raise ValueError('All original source views and their four real rank tails must remain in training')
        manifest = dict(format=FORMAT, records=14884, original_sources=3721, original_levels=[4,16,64,0],
            actual_counts=dict(counts), actual_level_totals={k: dict(v) for k, v in totals.items()},
            pack_sha256=sha(pack), index_sha256=sha(index), indexed_record_transport=writer.receipt(),
            all_original_failed_and_CE_only_views_retained=True,
            source_hints_are_training_only_not_unassisted_capability=True, source_training_admitted=False)
        atomic(DATA / 'manifest.json', manifest)
        files.update({str(p): sha(p) for p in [pack, index, DATA / 'manifest.json']})
        resources()
        if any(sha(p) != digest for p, digest in files.items()): raise ValueError('Every original/admitted source and indexed byte must remain exact')
        admission = dict(training_admitted=True, admitted_role='train', scope=SCOPE, owner=owner,
            at=datetime.now(timezone.utc).isoformat(), source_files_sha256=files, manifest_sha256=sha(DATA / 'manifest.json'),
            original_sources=3721, all_native_views=14884, each_level_original_context_tokens=2287522,
            each_level_original_response_EOS_targets=584218, native_execution_eligible_views=counts['native_execution_eligible_views'],
            CE_only_views=counts['CE_only_views'], actual_complete_indexed_records_freshly_read_back=14884,
            full_source_views_per_epoch=14884, actual_global_updates_per_epoch=3724,
            actual_single_source_three_empty_rank_training_steps_per_epoch=4,
            runtime_or_GPU_admission_not_granted=True, no_neural_model_GPUcontext_or_lease_started=True,
            model_updates=0, goal_achieved=False)
        atomic(DATA / 'admission.json', admission)
        atomic(OUT / 'complete.json', dict(complete=True, admission=str(DATA / 'admission.json'),
            admission_sha256=sha(DATA / 'admission.json'), data_only_training_admitted=True,
            all_original_sources_views_masks_and_failed_dispositions_retained=True,
            full_size_GPUcapacity_recovery_actual_training_and_quality_still_required=True, goal_achieved=False))
        print(str(DATA / 'admission.json'), flush=True)
    except BaseException as exc:
        atomic(OUT / 'failure.json', dict(owner=owner, error=repr(exc), automatic_retry=False,
            source_training_admission_published=(DATA / 'admission.json').exists(), all_actual_partial_outputs_preserved=True,
            no_new_neural_model_or_GPUjob=True, goal_achieved=False))
        raise


if __name__ == '__main__': main()
