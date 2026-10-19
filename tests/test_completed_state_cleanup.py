import json

import pytest
import torch

from slm_lab.completed_state_cleanup import retire


def fixture(root, mismatched=False):
    state = {'model': {'weight': torch.tensor([1., 2.])}, 'config': {'fixture': True},
             'stage': 'sft', 'steps': 2, 'planned_steps': 2, 'data_sha256': 'data', 'tokenizer_sha256': 'own'}
    torch.save({**state, 'optimizer': {'completed': True}}, root / 'training-state.pt')
    if mismatched: state['model'] = {'weight': torch.tensor([1., 3.])}
    torch.save(state, root / 'model.pt')
    (root / 'training.json').write_text(json.dumps({key: state[key] for key in ['stage', 'steps', 'data_sha256', 'tokenizer_sha256']} |
                                                     {'status': 'training run completed; capability unverified'}))


def test_retirement_keeps_exact_completed_weights_and_persists_audit(tmp_path):
    fixture(tmp_path)
    before = (tmp_path / 'model.pt').read_bytes()
    record = retire(tmp_path, tmp_path / 'audit.json')
    assert record['removed'] and not (tmp_path / 'training-state.pt').exists()
    assert (tmp_path / 'model.pt').read_bytes() == before
    assert json.loads((tmp_path / 'audit.json').read_text())['all_learned_tensors_verified_equal']


def test_mismatched_retained_model_prevents_retirement(tmp_path):
    fixture(tmp_path, True)
    with pytest.raises(ValueError, match='tensors differ'):
        retire(tmp_path, tmp_path / 'audit.json')
    assert (tmp_path / 'training-state.pt').exists() and not (tmp_path / 'audit.json').exists()
