"""Admit the full visual data only after all exact full source readers pass.

No GPU lease, model construction, optimizer or runtime/kernel admission.
Never alter the original prepared index, schedule, pixel reports, corrections
or failed numerical controls. The new receipt is the existing batch adapter's
separate data-only admission boundary.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

from slm_lab.bounded_disk_reserve_wait_v1 import wait_for_reserve
from slm_lab.exact_waiting_process_retirement_v1 import terminal
from slm_lab.full_visual_source_admission_v1 import validate
from slm_lab.local_source_closure_v2 import python_sources

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/full-corrected-multimodal-source-admission-CPU-v1'
INDEX = ROOT / 'reports/full-multimodal-corrected-source-index-CPU-v1'
INDEX_READER = ROOT / 'reports/full-multimodal-corrected-source-index-v1-readback/complete.json'
PIXELS = ROOT / 'reports/full-multimodal-pixel-validation-CPU-v1'
PIXEL_READER = ROOT / 'reports/full-multimodal-pixel-readback-CPU-v1'
BATCH = ROOT / 'reports/full-corrected-multimodal-batch-readback-CPU-v1'
CORRECTIONS = ROOT / 'data/public-document-visible-corrections-admitted-v1'
WORDS = ROOT / 'reports/full-word-microbatch-source-audit-CPU-v2/complete.json'
TESTS = ROOT / 'reports/full-visual-source-admission-v1-tests.log'
PREDECESSORS = [(PIXELS, dict(pid=51248, ticks='53113593')),
    (PIXEL_READER, dict(pid=106808, ticks='54228332')), (BATCH, dict(pid=108345, ticks='54257301'))]
ADMISSION = INDEX / 'source-training-admission.json'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def state(path):
    stat = Path(path).stat()
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]


def atomic(path, value):
    pending = path.with_suffix(path.suffix + '.pending')
    with pending.open('x') as stream:
        stream.write(json.dumps(value, indent=2) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(path)


def main():
    if (OUT.exists() or ADMISSION.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19
            or '9 passed' not in TESTS.read_text()
            or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'])):
        raise ValueError('Fresh tested one-thread CUDA-hidden nice19 data-only admission required')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    for folder, expected in PREDECESSORS:
        actual = read(folder / 'plan.json')['owner']
        if any(actual[k] != expected[k] for k in ['pid', 'ticks']):
            raise ValueError('Every full source-stage plan must name its exact actual producer birth')
    seeds = [Path(__file__), ROOT / 'tests/test_full_visual_source_admission_v1.py']
    initial = [INDEX / 'complete.json', INDEX_READER, CORRECTIONS / 'admission.json', WORDS, TESTS,
        *[folder / 'plan.json' for folder, _ in PREDECESSORS], *python_sources(ROOT, seeds)]
    files = {str(p): sha(p) for p in initial}
    states = {p: state(p) for p in files}
    owner = dict(pid=os.getpid(), ticks=Path('/proc/self/stat').read_text().split(') ', 1)[1].split()[19])
    OUT.mkdir()
    atomic(OUT / 'plan.json', dict(at=datetime.now(timezone.utc).isoformat(), owner=owner, source_files_sha256=files,
        exact_full_predecessors=[dict(folder=str(p), owner=b) for p, b in PREDECESSORS],
        output_new_data_admission=str(ADMISSION), complete_original_views=302627, complete_physical_paths=195121,
        complete_original_global_steps=4733, complete_original_rank_calls=18932, complete_original_whole_words=91585,
        all_complete_positive_response_EOS_targets=4669243, all_original_negative_targets_retained=80,
        representative_cases_cannot_substitute_for_complete_full_sources=True,
        admission_scope='data_only; full_size_GPU_capacity_recovery_and_model_quality_still_required',
        old_failed_numerical_controls_remain_unwaived=True, no_pixels_labels_or_prepared_indices_written=True,
        all_original_neural_workers_human_RL_and_nine_final_endpoint_scopes_unchanged=True,
        scheduler='SCHED_IDLE', nice=os.nice(0), CUDA_hidden=True, CPU_threads=1,
        minimum_free_bytes=40 * 1024**3, models_constructed=0, neural_forwards=0,
        training_admitted=False, goal_achieved=False))

    def guard():
        if (os.sched_getscheduler(0) != os.SCHED_IDLE or ADMISSION.exists()
                or any(state(p) != expected for p, expected in states.items())):
            raise ValueError('Fresh data-only receipt, idle scheduling and every immutable original source required')

    try:
        while True:
            guard()
            for folder, birth in PREDECESSORS:
                if terminal(birth) and ((folder / 'failure.json').exists() or not (folder / 'complete.json').exists()):
                    raise ValueError('An exact full predecessor ended without its complete original source readback: ' + str(folder))
            if all(terminal(birth) for _, birth in PREDECESSORS):
                break
            atomic(OUT / 'progress.json', dict(owner=owner,
                phase='waiting for all exact complete full physical and rank-batch source readbacks',
                live_predecessors=[birth for _, birth in PREDECESSORS if not terminal(birth)],
                source_training_admitted=False, neural_forwards=0))
            time.sleep(30)
        index = read(INDEX / 'complete.json')
        reader = read(INDEX_READER)
        pixels, pixel_reader, batches = [read(folder / 'complete.json') for folder, _ in PREDECESSORS]
        corrections, words = read(CORRECTIONS / 'admission.json'), read(WORDS)
        validate(index, reader, pixels, pixel_reader, batches, corrections, words)
        for evidence in [index, reader, pixels, pixel_reader, batches, corrections, words]:
            for path, digest in evidence.get('source_files_sha256', {}).items():
                if path in files and files[path] != digest:
                    raise ValueError('Every original complete source and independent reader must agree on immutable bytes')
                files[path] = digest
        final = [*[folder / 'complete.json' for folder, _ in PREDECESSORS],
            INDEX / 'complete-native-source-index.npy', INDEX / 'complete-corrected-single-epoch-schedule.json',
            CORRECTIONS / 'all4-complete-corrections-and-preferences.jsonl']
        files.update({str(p): sha(p) for p in final})
        states.update({p: state(p) for p in files})
        wait_for_reserve(ROOT, minimum_bytes=40 * 1024**3, guard=guard)
        if (sha(INDEX / 'complete-native-source-index.npy') != index['index_sha256']
                or sha(CORRECTIONS / 'all4-complete-corrections-and-preferences.jsonl') != corrections['data_file_sha256']
                or any(sha(p) != digest for p, digest in files.items())):
            raise ValueError('Every original/current source, complete index, correction and source reader byte must match')
        guard()
        receipt = dict(training_admitted=True, admitted_role='train', at=datetime.now(timezone.utc).isoformat(), owner=owner,
            qualification_scope='complete_corrected_visual_source_data_only', source_files_sha256=files,
            index_sha256=index['index_sha256'], index_readback_sha256=sha(INDEX_READER),
            full_pixel_report=str(PIXELS / 'complete.json'), full_pixel_readback=str(PIXEL_READER / 'complete.json'),
            batch_source_readback=str(BATCH / 'complete.json'), complete_source_views=302627,
            complete_physical_paths=195121, complete_global_steps=4733, complete_rank_calls=18932,
            complete_context_tokens=17681556, complete_response_EOS_targets=4669243,
            whole_word_views=91585, whole_CTC_UTF8_bytes=644063, four_complete_original_negative_targets=80,
            only_four_independently_qualified_visible_source_corrections_applied=True,
            arbitrary_caption_or_transcription_semantics_not_certified=True,
            runtime_or_kernel_or_model_admission_not_granted=True,
            historical_failed_word_and_prefix_numeric_controls_not_waived=True,
            no_new_neural_job_or_GPU_lease_or_context=True, new_learned_parameters=0,
            new_production_optimizer_updates=0, goal_achieved=False)
        atomic(ADMISSION, receipt)
        atomic(OUT / 'complete.json', dict(complete=True, passed=True, admission=str(ADMISSION),
            admission_sha256=sha(ADMISSION), data_only_training_admitted=True,
            full_size_GPU_capacity_recovery_training_and_quality_still_pending=True, goal_achieved=False))
        print(json.dumps(dict(admission=str(ADMISSION), data_only=True, views=302627, neural_updates=0)))
    except BaseException as error:
        atomic(OUT / 'failure.json', dict(owner=owner, error=repr(error), automatic_retry=False,
            source_training_admission_published=ADMISSION.exists(), no_new_neural_job=True, goal_achieved=False))
        raise


if __name__ == '__main__':
    main()
