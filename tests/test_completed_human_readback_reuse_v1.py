import json
from pathlib import Path
import shutil

import pytest

from slm_lab import completed_human_readback_reuse_v1 as recovery

ORIGINAL = Path('/tmp/slm-completed-joint-initializer-CPU-v1')
LOG = Path('/tmp/slm-completed-joint-initializer-CPU-v1-affinity-fixed.log')


def copied(tmp_path):
    if not (ORIGINAL / 'failure.json').is_file():
        pytest.skip('The exact closed full-size intermediate proof is local and excluded from Git')
    folder = tmp_path / 'original'; folder.mkdir()
    for name in ['plan.json', 'source-admission.json', 'actual-completed-human-state-readback.json', 'failure.json']:
        shutil.copyfile(ORIGINAL / name, folder / name)
    log = tmp_path / 'original.log'; shutil.copyfile(LOG, log)
    return folder, log, recovery.read(folder / 'source-admission.json')


def test_actual_successful_full_four_state_inspection_can_be_reused_without_relabelling_failed_export(tmp_path):
    folder, log, admission = copied(tmp_path)
    before = {str(p): recovery.sha(p) for p in [*folder.iterdir(), log]}
    result = recovery.capture(folder, admission, log)
    assert result['original_export_failed_and_not_relabelled']
    assert result['completed_independent_four_state_readback_reused'] and not result['GPU_training_admitted']
    assert result['verified']['actual_completed_human_updates'] == 5362
    assert result['verified']['original_pair_population'] == 85786
    assert before == {str(p): recovery.sha(p) for p in [*folder.iterdir(), log]}


@pytest.mark.parametrize('kind', ['live', 'unrelated_failure', 'different_donor', 'missing_rank', 'partial_native'])
def test_live_changed_incomplete_and_neural_results_cannot_be_reused_as_this_source_inspection(tmp_path, monkeypatch, kind):
    folder, log, admission = copied(tmp_path)
    if kind == 'live': monkeypatch.setattr(recovery, 'terminal', lambda owner: False)
    elif kind == 'unrelated_failure': log.write_text('ValueError: actual neural state mismatch\n')
    elif kind == 'different_donor': admission = dict(admission, checkpoint_sha256='0' * 64)
    elif kind == 'partial_native': (folder / 'native').mkdir()
    else:
        path = folder / 'actual-completed-human-state-readback.json'
        record = recovery.read(path); record['actual_rank_observations'].pop(); path.write_text(json.dumps(record))
    with pytest.raises(ValueError): recovery.capture(folder, admission, log)
