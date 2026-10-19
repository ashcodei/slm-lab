import json
from PIL import Image
import numpy as np
import torch

from slm_lab import training
from slm_lab.content_caption_training import content_mean, train
from slm_lab.vision_transfer import language_digest
from test_multimodal_training import setup


def test_content_pooling_excludes_boundary_and_padding_features():
    hidden = torch.tensor([[[999.], [1.], [3.], [777.], [888.]], [[999.], [5.], [777.], [888.], [888.]]])
    mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]], dtype=torch.bool)
    torch.testing.assert_close(content_mean(hidden, mask), torch.tensor([[2.], [5.]]))


def test_actual_content_objective_changes_vision_only_and_restores_trainer(tmp_path):
    original, args = setup(tmp_path); args.stage = 'contrastive'; args.trainable = 'vision'; args.batch_size = 4
    Image.fromarray(np.full((16, 16, 3), [50, 50, 200], dtype=np.uint8)).save(tmp_path / 'blue.png')
    rows = [{'response': 'A red square.', 'image': 'image.png'}] * 2
    rows += [{'response': 'A blue square.', 'image': 'blue.png'}] * 2
    (tmp_path / 'data.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    old_wrapper = training.TrainingOutputs
    train(args)
    assert training.TrainingOutputs is old_wrapper
    state = torch.load(tmp_path / 'trained/model.pt', weights_only=True)['model']
    assert language_digest(state) == language_digest(original.state_dict())
    assert not torch.equal(state['vision.patch.weight'], original.vision.patch.weight)
    assert json.loads((tmp_path / 'trained/caption-target.json').read_text())['shared_trainer_modified'] is False
