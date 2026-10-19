from dataclasses import asdict
import json
from pathlib import Path

from PIL import Image
import pytest
import torch

from slm_lab.corpus import file_hash
from slm_lab.learned_character_ctc import LearnedCharacterSequenceModel
from slm_lab.model import ModelConfig, VisionEncoder
from slm_lab.sequence_diagnostics import evaluate, frozen_checkpoint


def test_snapshot_survives_atomic_replacement_of_the_training_checkpoint(tmp_path):
    latest = tmp_path / 'latest.pt'
    latest.write_bytes(b'old complete checkpoint')
    with frozen_checkpoint(latest, tmp_path) as snapshot:
        new = tmp_path / 'new.pt'
        new.write_bytes(b'next complete checkpoint')
        new.replace(latest)
        assert snapshot.read_bytes() == b'old complete checkpoint'
        assert latest.read_bytes() == b'next complete checkpoint'
    assert not snapshot.exists()


def test_diagnostic_exercises_both_splits_without_changing_or_retaining_the_checkpoint(tmp_path):
    config = ModelConfig(vocab_size=260, hidden_size=32, intermediate_size=64, num_layers=1,
                         num_heads=4, num_kv_heads=2, vision_hidden_size=16, vision_intermediate_size=32,
                         vision_layers=1, vision_heads=2, image_size=16, patch_size=4, pool_size=2)
    data = tmp_path / 'data'
    data.mkdir()
    for split, color in [('train', 60), ('valid', 160)]:
        rows = []
        for index, text in enumerate(['a', 'b']):
            image = data / f'{split}-{index}.png'
            Image.new('RGB', (16, 16), (color, index * 80, 0)).save(image)
            rows.append({'id': image.name, 'image': str(image), 'stored_image_sha256': file_hash(image),
                         'group_id': image.name, 'ctc_family': 'rendered', 'response': text})
        (data / (split + '.jsonl')).write_text(''.join(json.dumps(row) + '\n' for row in rows))
    (data / 'manifest.json').write_text(json.dumps({
        'configuration': {'image_size': 16},
        'files_sha256': {name: file_hash(data / name) for name in ['train.jsonl', 'valid.jsonl']}}))
    source = Path(__file__).resolve().parents[1] / 'src/slm_lab'
    model = LearnedCharacterSequenceModel(VisionEncoder(config), frames=8)
    # Force empty decoding, so the aggregate must expose blank collapse despite finite loss.
    with torch.no_grad():
        model.classifier.weight.zero_()
        model.classifier.bias.zero_()
        model.classifier.bias[0] = 8
    plan = {'config': asdict(config), 'frames': 8, 'tokenizer_sha256': 'toy-only',
            'train_rows': 2, 'passes': 1, 'steps': 2, 'batch_size': 1, 'accumulation': 1,
            'manifest_sha256': file_hash(data / 'manifest.json'),
            'readout_implementation_sha256': file_hash(source / 'learned_character_ctc.py'),
            'implementation_sha256': file_hash(source / 'character_ctc.py')}
    checkpoint = tmp_path / 'latest.pt'
    torch.save({'model': model.state_dict(), 'plan': plan, 'steps': 1}, checkpoint)
    digest = file_hash(checkpoint)
    report = evaluate(checkpoint, data, tmp_path / 'report.json', 'cpu', maximum_per_family=2)
    assert report['checkpoint_steps'] == 1 and report['checkpoint_sha256'] == digest
    assert report['training_examples_previously_exposed'] == 1
    assert report['splits']['train']['by_exposure']['previously_exposed']['examples'] == 1
    assert report['splits']['train']['by_exposure']['not_yet_exposed']['examples'] == 1
    for split in ['train', 'valid']:
        result = report['splits'][split]
        assert result['overall']['examples'] == result['overall']['empty_outputs'] == 2
        assert result['overall']['blank_argmax_frame_fraction'] == 1
        assert result['overall']['exact_words'] == 0
        assert result['overall']['normalized_ctc_loss'] > 0
        assert result['overall']['blank_ctc_loss_gain'] == 0
        assert all(row['wrong_image_group_id'] != row['group_id'] for row in result['results'])
        assert all(row['input_pixels_sha256'] != row['wrong_pixels_sha256'] for row in result['results'])
    assert file_hash(checkpoint) == digest and not list(tmp_path.glob('.sequence-assessment-*'))
    assert report['goal_achieved'] is False and report['inference_precision'] == 'fp32'
    with pytest.raises(ValueError, match='fresh diagnostic'):
        evaluate(checkpoint, data, tmp_path / 'report.json', 'cpu', 2)
