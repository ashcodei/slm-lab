import pytest
import torch

from slm_lab.model import ModelConfig
from slm_lab.own_dense_actor_v1 import OwnDenseActor
from slm_lab.own_dense_growth_v1 import OwnDenseAssistant, new_backbone
from slm_lab.own_execution_contrast_CPU_v1 import contrastive_loss, update, validate_pair


def test_preference_gradient_improves_chosen_and_suppresses_wrong_mean_without_length_bias():
    preferred = torch.tensor(-2., requires_grad=True); rejected = torch.tensor(-1., requires_grad=True)
    loss = contrastive_loss(preferred, rejected, beta=2., margin=.5); loss.backward()
    assert preferred.grad < 0 and rejected.grad > 0
    p = torch.tensor([-1., -2.]); n = torch.tensor([-2., -1., -1.])
    assert torch.equal(contrastive_loss(p.mean(), n.mean()), contrastive_loss(p.repeat(2).mean(), n.repeat(3).mean()))
    with pytest.raises(ValueError): contrastive_loss(preferred, rejected, beta=float('nan'))


def test_actual_native_actor_receives_wrong_sequence_gradient_and_preserves_fixed_values():
    torch.set_num_threads(1)
    config = ModelConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_layers=2,
        num_heads=4, num_kv_heads=2, max_text_length=64, vision_hidden_size=32, vision_intermediate_size=64,
        vision_layers=1, vision_heads=4, image_size=32, patch_size=16)
    actor = OwnDenseActor(OwnDenseAssistant(new_backbone(config, 969117), reward_hidden=8, word_temporal=4,
        pointing_rank=8, router_hidden=8, detector_hidden=4, word_expansion=1), rank=2, seed=969217).eval()
    row = dict(pair_index=0, pair_id='small-mechanics-only', source_id='synthetic-test-contract',
        original_split='train', publisher_split='train', machine_execution_preference_not_human_ranked=True,
        external_model_teacher=False, positive_finite_case_credit=1., negative_finite_case_credit=0.,
        original_native_prefix_ids=[1, 7], complete_chosen_ids=[1, 7, 8, 9, 2], complete_chosen_labels=[-100, -100, 8, 9, 2],
        complete_rejected_ids=[1, 7, 12, 11, 10, 2], complete_rejected_labels=[-100, -100, 12, 11, 10, 2],
        rejected_actual_action_ids=[12, 11, 10, 2], complete_original_reference_EOS_targets=3,
        rejected_generated_EOS=True, rejected_is_censored_at_complete512_action_budget=False, rejected_full_EOS_was_not_invented=True)
    with torch.no_grad(): own = actor(torch.tensor([row['complete_chosen_ids']]))['logits'][:, :-1].float().log_softmax(-1)
    original = {n: p.detach().clone() for n, p in actor.named_parameters()}
    parameters = [p for _, p in actor.named_selected(('actor',))]
    adam = torch.optim.AdamW(parameters, lr=1e-3, betas=(.9, .95), weight_decay=.01, foreach=False)
    result = update(actor, adam, row, own, method='anchor_contrast')
    assert result['genuine_negative_action_targets'] == 4 and result['original_reference_EOS_targets'] == 3
    assert result['length_normalized_preference_loss'] > 0 and result['gradient_norm_before_clip'] > 0
    changed = [n for n, p in actor.named_parameters() if not torch.equal(original[n], p)]
    assert changed and all(n.startswith('parts.') and n != 'parts.critic.weight' for n in changed)
    bad = dict(row, rejected_generated_EOS=False)
    with pytest.raises(ValueError): validate_pair(bad)
    with pytest.raises(ValueError): update(actor, adam, row, own[:, :-1], method='anchor_contrast')
