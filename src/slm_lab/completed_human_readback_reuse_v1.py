"""Reuse only the completed four-state inspection of a failed CPU export.

The original export remains failed. Its exact successful intermediate proof,
all consumed source bytes and the narrow filename failure remain immutable.
This does not reuse a neural optimizer, generated answer or GPU admission.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .exact_waiting_process_retirement_v1 import terminal


def read(path): return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def capture(folder, current_admission, log):
    folder, log = Path(folder).resolve(), Path(log).resolve()
    plan = read(folder / 'plan.json')
    if not terminal(plan['owner']):
        raise ValueError('The exact original CPU source owner is still live; preserve it without a replacement')
    expected = "FileNotFoundError: [Errno 2] No such file or directory: '/home/jovyan/slm-lab/reports/complete-joint-actual-capacity-cases-CPU-v3/actual-complete-source-capacity-plan.json'"
    failure = read(folder / 'failure.json')
    if ((folder / 'complete.json').exists() or (folder / 'native').exists()
            or not log.read_text().strip().endswith(expected)
            or failure.get('error') != "FileNotFoundError(2, 'No such file or directory')"
            or read(folder / 'source-admission.json') != current_admission
            or plan.get('GPU_contexts') != 0 or plan.get('new_optimizer_updates') != 0):
        raise ValueError('Only the exact preserved filename failure after successful identical-source inspection may be recovered')
    verified = read(folder / 'actual-completed-human-state-readback.json')
    if (verified.get('actual_full_size_completed_human_state_readback') is not True
            or verified.get('all_four_final_private_states_match_exported_actor') is not True
            or verified.get('actual_completed_human_updates') != 5362 or verified.get('original_pair_population') != 85786
            or verified.get('new_Adam_calls') != 0 or verified.get('GPU_training_admitted') is not False
            or [r['rank'] for r in verified.get('actual_rank_observations', [])] != [0, 1, 2, 3]
            or len({r['actual_private_RNG_sha256'] for r in verified['actual_rank_observations']}) != 4
            or len({r['actual_common_actor_Adam_metrics_sha256'] for r in verified['actual_rank_observations']}) != 1):
        raise ValueError('The actual successful full four-state inspection must remain complete and independent')
    files = dict(plan['source_files_sha256']); files.update(current_admission['source_files_sha256'])
    paths = [folder / name for name in ['plan.json', 'source-admission.json', 'actual-completed-human-state-readback.json', 'failure.json']]
    files.update({str(p): sha(p) for p in [*paths, log]})
    if any(sha(p) != h for p, h in files.items()):
        raise ValueError('Every original failed-export source byte and successful intermediate proof must remain unchanged')
    return dict(verified=deepcopy(verified), source_files_sha256=files, exact_failed_original_owner=plan['owner'],
        original_export_failed_and_not_relabelled=True,
        completed_independent_four_state_readback_reused=True, original_successful_readback_sha256=sha(paths[2]),
        current_loaded_actor_values_still_require_comparison_to_original_private_state=True,
        GPU_training_admitted=False, goal_achieved=False)
