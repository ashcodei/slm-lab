"""Bind every original visual occurrence and select actual worst-shape cases.

No model, pixels, GPU context, training admission or completed capacity claim.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import traceback

import numpy as np

from slm_lab.bounded_disk_reserve_wait_v1 import wait_for_reserve
from slm_lab.full_visual_source_admission_v1 import FAMILIES
from slm_lab.joint_visual_capacity_cases_v1 import choose

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/joint-visual-actual-capacity-cases-CPU-v1'
SOURCE = ROOT / 'reports/full-multimodal-corrected-source-index-CPU-v1'
WORD = ROOT / 'reports/full-word-microbatch-source-audit-CPU-v2'


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
        raise ValueError('Fresh CUDA-hidden nice19 one-thread source-only capacity preparation required')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    wait_for_reserve(ROOT, minimum_bytes=40 * 1024**3)
    test_path = ROOT / 'reports/joint-visual-capacity-cases-v1-tests.log'
    if '9 passed' not in test_path.read_text():
        raise ValueError('All9whole-source capacity-selection controls must pass')
    source = read(SOURCE / 'complete.json')
    reader_path = ROOT / 'reports/full-multimodal-corrected-source-index-v1-readback/complete.json'
    reader = read(reader_path)
    words = read(WORD / 'complete.json')
    index_path = SOURCE / 'complete-native-source-index.npy'
    schedule_path = SOURCE / 'complete-corrected-single-epoch-schedule.json'
    word_path = WORD / 'all91585-complete-word-native-identities.jsonl'
    if (not source['complete'] or not reader['complete'] or not words['complete'] or not words['passed']
            or source['training_admitted'] or reader['training_admitted']
            or sha(index_path) != source['index_sha256'] or sha(schedule_path) != source['schedule_sha256']
            or sha(word_path) != words['identities_sha256']):
        raise ValueError('Every complete original source/index/readback and whole-word identity proof must agree')
    paths = [Path(__file__), ROOT / 'src/slm_lab/joint_visual_capacity_cases_v1.py',
        ROOT / 'tests/test_joint_visual_capacity_cases_v1.py', test_path, SOURCE / 'complete.json',
        reader_path, WORD / 'complete.json', index_path, schedule_path, word_path]
    files = {str(p): sha(p) for p in paths}
    OUT.mkdir()
    try:
        index = np.load(index_path, mmap_mode='r', allow_pickle=False)
        schedule = read(schedule_path)
        with word_path.open('rb') as stream:
            identities = []
            for line in stream:
                if not line.endswith(b'\n'):
                    raise ValueError('Every complete original LF-delimited word identity is required')
                identities.append(json.loads(line))
        plan = choose(source['family_names'], index, schedule['entries'], identities)
        if (plan['actual_original_family_counts'] != FAMILIES or plan['actual_original_epoch_entries'] != 4733
                or plan['actual_original_epoch_source_views'] != 302627
                or plan['maximum_original_native_context'] != 2952
                or plan['maximum_original_word_UTF8_bytes'] != 50
                or plan['maximum_original_photo_negative_set'] != 64
                or plan['actual_original_tail_steps'] != [3205]
                or sum(e['complete_context_tokens'] for e in schedule['entries']) != 17681556
                or sum(e['response_EOS_targets'] for e in schedule['entries']) != 4669243
                or plan['capacity_entries'][-1]['original_local_positions'] != [21055]
                or plan['capacity_entries'][-1]['proposed_rank_source_counts'] != [1, 0, 0, 0]):
            raise ValueError('The actual complete original302627view/4733step budgets and worst cases must be retained')
        if any(sha(path) != expected for path, expected in files.items()):
            raise ValueError('Every original source and capacity-selection implementation must retain its bytes')
        write(OUT / 'actual-complete-source-capacity-plan.json', plan)
        write(OUT / 'complete.json', dict(complete=True, passed=True, at=datetime.now(timezone.utc).isoformat(),
            source_files_sha256=files, actual_original_family_counts=plan['actual_original_family_counts'],
            actual_original_epoch_entries=4733, actual_original_epoch_source_views=302627,
            actual_selected_capacity_entries=len(plan['capacity_entries']),
            actual_selected_capacity_source_presentations=sum(len(e['original_local_positions']) for e in plan['capacity_entries']),
            original_context_tokens=17681556, original_response_EOS_targets=4669243,
            maximum_original_native_context=2952, full_original_image_prefix_tokens=196,
            maximum_original_padded_single_decoder_context=3148,
            largest_original_photo_global_negative_set=64, longest_original_word_UTF8_bytes=50,
            actual_singleton_word_with_three_empty_ranks=True,
            actual_plan_sha256=sha(OUT / 'actual-complete-source-capacity-plan.json'),
            selected_original_occurrences_and_full_native_targets_unchanged=True,
            only_capacity_cursor_steps_renumbered=True, training_admitted=False,
            actual_full_size_GPU_capacity_passed=False, neural_forwards=0, goal_achieved=False))
        print(json.dumps(dict(complete=True, actual_source_views=302627,
            actual_capacity_entries=len(plan['capacity_entries']), actual_GPU_capacity_passed=False)))
    except BaseException:
        write(OUT / 'failure.json', dict(complete=False, traceback=traceback.format_exc(), goal_achieved=False))
        raise


if __name__ == '__main__':
    main()
