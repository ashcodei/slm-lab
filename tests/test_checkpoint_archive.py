import json
import subprocess
import sys

import pytest
import torch

from slm_lab.checkpoint_archive import archive_completed, reconstruct
from slm_lab.corpus import file_hash


def checkpoint(folder, vision, notes):
    folder.mkdir()
    torch.save({'model': {'language': torch.arange(8192), 'vision': torch.tensor(vision)},
                'notes': notes}, folder / 'model.pt')
    (folder / 'training.json').write_text(json.dumps({'status': 'training completed', 'steps': 3, 'planned_steps': 3}))
    return folder / 'model.pt'


def test_exact_archive_restore_handles_changed_metadata_offsets_and_vision(tmp_path):
    source = checkpoint(tmp_path / 'source', [1., 2.], 'short')
    candidate = checkpoint(tmp_path / 'candidate', [3., 4.], 'different metadata' * 30)
    expected = candidate.read_bytes()
    directory = tmp_path / 'archive'
    manifest = archive_completed(candidate, source, directory, remove_original=True)
    assert not candidate.exists() and manifest['original_removed']
    assert manifest['payload_bytes'] < manifest['original_bytes'] / 4
    reconstruct(directory, candidate)
    assert candidate.read_bytes() == expected
    assert torch.load(candidate, weights_only=True)['model']['vision'].tolist() == [3., 4.]
    with pytest.raises(ValueError, match='fresh destination'):
        reconstruct(directory, candidate)


@pytest.mark.parametrize('changed', ['anchor', 'payload'])
def test_changed_anchor_or_payload_prevents_restore_without_output(tmp_path, changed):
    source = checkpoint(tmp_path / 'source', [1., 2.], 'anchor')
    candidate = checkpoint(tmp_path / 'candidate', [3., 4.], 'candidate')
    directory = tmp_path / 'archive'
    archive_completed(candidate, source, directory)
    modified = source if changed == 'anchor' else directory / 'payload.bin'
    modified.write_bytes(modified.read_bytes() + b'changed')
    anchor_hash = file_hash(source)
    output = tmp_path / 'restored.pt'
    with pytest.raises(ValueError, match='anchor or payload changed'): reconstruct(directory, output)
    assert not output.exists() and file_hash(source) == anchor_hash and candidate.exists()


def test_incomplete_checkpoint_cannot_be_archived(tmp_path):
    source = checkpoint(tmp_path / 'source', [1., 2.], 'anchor')
    candidate = checkpoint(tmp_path / 'candidate', [3., 4.], 'candidate')
    (candidate.parent / 'training.json').write_text(json.dumps({'status': 'running', 'steps': 2, 'planned_steps': 3}))
    with pytest.raises(ValueError, match='Only completed'):
        archive_completed(candidate, source, tmp_path / 'archive', True)
    assert candidate.exists()


def test_exact_preparation_receipt_archives_weights_snapshot_without_claiming_training_completion(tmp_path):
    source = checkpoint(tmp_path / 'source', [1., 2.], 'anchor')
    candidate = checkpoint(tmp_path / 'prepared', [3., 4.], 'prepared weights')
    (candidate.parent / 'training.json').unlink()
    receipt = tmp_path / 'preparation.json'
    receipt.write_text(json.dumps({'branches': {'prepared': {'checkpoint': str(candidate), 'checkpoint_sha256': file_hash(candidate)}}}))
    directory = tmp_path / 'archive'
    manifest = archive_completed(candidate, source, directory, True, receipt)
    assert manifest['prepared_snapshot_receipt']['sha256'] == file_hash(receipt)
    assert not candidate.exists()
    reconstruct(directory, candidate)
    assert file_hash(candidate) == json.loads(receipt.read_text())['branches']['prepared']['checkpoint_sha256']


def test_real_archive_cli_ignores_its_own_input_argument_and_restores_exact_bytes(tmp_path):
    source = checkpoint(tmp_path / 'source', [1., 2.], 'anchor')
    candidate = checkpoint(tmp_path / 'candidate', [3., 4.], 'candidate')
    expected = candidate.read_bytes(); directory = tmp_path / 'archive'
    completed = subprocess.run([sys.executable, '-m', 'slm_lab.checkpoint_archive', 'archive',
                               '--checkpoint', str(candidate), '--source', str(source),
                               '--archive', str(directory), '--remove-original'], capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads(completed.stdout)['original_removed'] and not candidate.exists()
    restored = subprocess.run([sys.executable, '-m', 'slm_lab.checkpoint_archive', 'restore',
                               '--archive', str(directory), '--output', str(candidate)],
                              capture_output=True, text=True, timeout=30)
    assert restored.returncode == 0, restored.stdout + restored.stderr
    assert candidate.read_bytes() == expected


def test_another_live_process_still_blocks_archival_after_self_exclusion(tmp_path):
    source = checkpoint(tmp_path / 'source', [1., 2.], 'anchor')
    candidate = checkpoint(tmp_path / 'candidate', [3., 4.], 'candidate'); expected = candidate.read_bytes()
    holder = subprocess.Popen([sys.executable, '-c', 'import sys; sys.stdin.read()', str(candidate)],
                              stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        with pytest.raises(ValueError, match='live process'): archive_completed(candidate, source, tmp_path / 'archive', True)
        assert candidate.read_bytes() == expected and not (tmp_path / 'archive').exists()
    finally:
        holder.communicate(input=b'', timeout=5)
