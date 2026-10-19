"""Admit evaluation-only recovery of an actually completed public epoch."""
import json
from pathlib import Path

from .corpus import file_hash
from .exact_waiting_process_retirement_v1 import terminal

PUBLIC = 'reports/public-pretraining-v1'
FAILED = 'reports/public-pretraining-recovery-v1'
REPORT = 'reports/public-pretraining-assessment-recovery-v1'
QUEUE = REPORT + '-queue.json'
EPOCH = 'reports/public-pretraining-complete-epoch-value-readback-CPU-v2/complete.json'
ORIGINAL = dict(pid=1241, ticks='49351203')


def read(path):
    return json.loads(Path(path).read_text())


def identity(value):
    return {k: value[k] for k in ('pid', 'ticks')}


def completed_training_after_observer_failure(root):
    root = Path(root)
    runtime = root / FAILED
    public = root / PUBLIC
    if not terminal(ORIGINAL):
        return False
    if any(not terminal(child) for child in read(runtime / 'children.json')):
        return False
    if (identity(read(runtime / 'plan.json')['owner']) != ORIGINAL
            or not read(runtime / 'restored.json')['all_registered_point_actors_terminal']
            or 'exact owned child failed' not in read(runtime / 'failure.json')['error']
            or 'TypeError: observe.<locals>.generate() takes 1 positional argument but 2 were given'
                not in (runtime / 'assess-initial-broad-2.log').read_text()):
        raise ValueError('Exact preserved terminal observer failure and resource restoration required')
    plan = read(public / 'plan.json')
    trained = read(public / 'training-complete.json')
    epoch = read(root / EPOCH)
    inputs = read(public / 'native-assessment-inputs.json')
    if (plan['planned_steps'] != 4447 or trained['actual_updates'] != 4447
            or not trained['all_new_admitted_raw_train_windows_once']
            or not trained['all826verified_train_functions_once']
            or trained['actual_selected_source_budget'] != plan['selected_source_budget']
            or not epoch['passed'] or epoch['actual_complete_updates'] != 4447
            or epoch['actual_unique_neural_values'] != 984399690
            or epoch['source_files_sha256'][str(public / 'plan.json')] != file_hash(public / 'plan.json')
            or any(inputs[arm]['checkpoint_sha256'] != epoch['source_files_sha256'][
                str(root / 'checkpoints/public-pretraining-v1' / arm / 'model.pt')]
                for arm in ('initial', 'trained'))):
        raise ValueError('The independently read full original epoch and both complete checkpoints required')
    return True


def assessment_ready(root):
    root = Path(root)
    if not completed_training_after_observer_failure(root):
        return False
    folder = root / REPORT
    if not (folder / 'plan.json').exists():
        return False
    plan = read(folder / 'plan.json')
    if not terminal(plan['owner']):
        return False
    if (folder / 'failure.json').exists():
        raise ValueError('The separately preserved assessment recovery failed')
    if any(not terminal(child) for child in read(folder / 'children.json')):
        return False
    done = read(folder / 'complete.json')
    if (not done.get('complete') or not done.get('all_original_native_populations_complete')
            or not read(folder / 'restored.json')['all_registered_point_actors_terminal']
            or done['original_complete_epoch_readback_sha256'] != file_hash(root / EPOCH)):
        raise ValueError('Every unchanged matched native population and actual GPU restoration required')
    return True
