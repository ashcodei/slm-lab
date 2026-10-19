from dataclasses import asdict

import torch
from slm_lab.corpus import file_hash
from slm_lab.model import ModelConfig, TinyMultimodalLM, VisionEncoder
from slm_lab.masked_visual_pretraining import MaskedVisualModel
from slm_lab.vision_transfer import prepare, language_digest


def test_transfer_changes_only_own_vision_and_excludes_temporary_decoder(tmp_path):
    config = ModelConfig(vocab_size=260, hidden_size=32, intermediate_size=64, num_layers=1,
                         num_heads=4, num_kv_heads=2, vision_hidden_size=16, vision_intermediate_size=32,
                         vision_layers=1, vision_heads=2, image_size=16, patch_size=4, pool_size=2)
    source_dir = tmp_path / 'source'; source_dir.mkdir(); (source_dir / 'tokenizer.json').write_text('own tokenizer fixture')
    source = source_dir / 'step.pt'; base = TinyMultimodalLM(config)
    token_hash = file_hash(source_dir / 'tokenizer.json')
    torch.save({'model': base.state_dict(), 'config': asdict(config), 'steps': 2000,
                'tokenizer_sha256': token_hash, 'optimizer': {'unused': 1}}, source)
    donor = MaskedVisualModel(VisionEncoder(config))
    with torch.no_grad(): donor.vision.patch.weight.add_(1)
    masked = tmp_path / 'masked.pt'
    torch.save({'model': donor.state_dict(), 'steps': 10,
                'plan': {'config': asdict(config), 'steps': 10, 'tokenizer_sha256': token_hash}}, masked)
    source_sha = file_hash(source)
    report = prepare(source, masked, tmp_path / 'branches')
    original = torch.load(report['branches']['original']['checkpoint'], weights_only=True)
    changed = torch.load(report['branches']['masked']['checkpoint'], weights_only=True)
    assert language_digest(changed['model']) == language_digest(original['model']) == language_digest(base.state_dict())
    assert torch.equal(changed['model']['vision.patch.weight'], donor.vision.patch.weight)
    assert not torch.equal(changed['model']['vision.patch.weight'], original['model']['vision.patch.weight'])
    assert all(not key.startswith(('decoder', 'predict', 'mask_token')) for key in changed['model'])
    assert 'optimizer' not in changed and report['training_decoder_transferred'] is False
    assert file_hash(source) == source_sha and report['goal_achieved'] is False
