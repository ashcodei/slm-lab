import torch

from slm_lab.model import ModelConfig, VisionEncoder
from slm_lab.object_supervision import ObjectClassifier, group_bootstrap


def small():
    c = ModelConfig(vocab_size=260, hidden_size=32, intermediate_size=64, num_layers=1, num_heads=4, num_kv_heads=2,
                    max_text_length=64, image_size=16, patch_size=4, vision_hidden_size=16,
                    vision_intermediate_size=32, vision_layers=1, vision_heads=4, pool_size=2)
    return ObjectClassifier(VisionEncoder(c), 3)


def test_label_learning_changes_encoder_only_when_requested_and_preserves_unused_projector():
    torch.manual_seed(9); images = torch.randn(4, 3, 16, 16); targets = torch.tensor([0, 1, 2, 1])
    for train_vision in [False, True]:
        model = small(); before = {k: v.detach().clone() for k, v in model.vision.state_dict().items()}
        for p in model.vision.parameters(): p.requires_grad_(train_vision)
        model.vision.project.weight.requires_grad_(False)
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.1)
        torch.nn.functional.cross_entropy(model(images, train_vision), targets).backward(); optimizer.step()
        changed = any(not torch.equal(before[k], v) for k, v in model.vision.state_dict().items())
        assert changed == train_vision
        assert torch.equal(before['project.weight'], model.vision.project.weight)


def test_paired_group_bootstrap_zero_when_same_cases_have_same_predictions():
    report = {'accuracy': .5, 'results': [{'id': str(i), 'group_id': str(i // 2), 'target_index': i % 2,
               'model_pixel_sha256': str(i), 'correct': i % 2 == 0} for i in range(4)]}
    result = group_bootstrap(report, report)
    assert result['paired_parent_bootstrap_95_interval'] == [0., 0.]
    assert result['bootstrap_parent_images'] == 2
