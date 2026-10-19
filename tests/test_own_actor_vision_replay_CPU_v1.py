import copy
from pathlib import Path

import pytest
import torch

from slm_lab.corpus import file_hash
from slm_lab.deterministic_positions_v1 import install
from slm_lab.model import ModelConfig
from slm_lab.own_actor_vision_replay_CPU_v1 import OwnActorVisionReplayCPU
from slm_lab.own_dense_actor_v1 import OwnDenseActor
from slm_lab.own_dense_growth_v1 import OwnDenseAssistant, new_backbone
from slm_lab.own_vision_replay_bundle_CPU_v1 import export, load
from slm_lab.own_vision_replay_state_CPU_v1 import snapshot, restore
from slm_lab.shared_word_data_v1 import SharedWordData


def test_real_whole_word_replay_gradients_preserve_language_and_native_bundle(tmp_path):
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    words = SharedWordData(root / 'data/shared-byte-words-v1', 'train', 'optimization')
    positions = [next(i for i, r in enumerate(words.rows) if r['byte_family'] == family)
                 for family in ['real_word', 'additional_real_word']]
    config = ModelConfig(vocab_size=32768, hidden_size=32, intermediate_size=64, num_layers=2, num_heads=4,
        num_kv_heads=2, max_text_length=4096, vision_hidden_size=32, vision_intermediate_size=64,
        vision_layers=1, vision_heads=4, image_size=224, patch_size=16)
    base = install(OwnDenseAssistant(new_backbone(config, 970917), reward_hidden=8, word_temporal=4,
        pointing_rank=8, router_hidden=8, detector_hidden=4, word_expansion=5)).eval()
    actor = OwnDenseActor(base, rank=2, seed=971017).eval(); view = OwnActorVisionReplayCPU(actor).eval()
    ids = torch.tensor([[1, 7, 8, 9, 2], [1, 7, 10, 11, 2]])
    pixels = torch.linspace(-1, 1, 2 * 3 * 448 * 448).reshape(2, 3, 448, 448)
    with torch.no_grad(): original_logits = actor(ids)['logits'].clone()
    selected = [p for _, p in view.named_selected()]; chosen = {id(p) for p in selected}
    frozen = {n: p.detach().clone() for n, p in actor.named_parameters() if id(p) not in chosen}
    adam = torch.optim.AdamW(selected, lr=1e-4, betas=(.9, .95), weight_decay=.01, foreach=False)
    photo = view.objective(dict(input_ids=ids, labels=ids, pixel_values=pixels), ids, torch.ones_like(ids, dtype=torch.bool), torch.eye(2, dtype=torch.bool))
    result = view.backward_with_replay(photo, words, positions)
    assert result['original_complete_words'] == 2
    assert result['UTF8_byte_targets'] == sum(len(words.rows[i]['response'].encode('utf-8')) for i in positions)
    assert result['original_word_response_EOS_targets'] == sum(words.rows[i]['native']['length'] - words.rows[i]['native']['target_start'] for i in positions)
    assert actor.base.heads['word'].classifier.weight.grad.abs().sum() > 0
    assert actor.base.network.vision.position.grad.abs().sum() > 0
    assert actor.base.heads['pointing']['query'].weight.grad.abs().sum() > 0
    torch.nn.utils.clip_grad_norm_(selected, 1.); adam.step(); adam.zero_grad(set_to_none=True)
    assert all(torch.equal(frozen[n], p) for n, p in actor.named_parameters() if n in frozen)
    with torch.no_grad(): assert torch.equal(original_logits, actor(ids)['logits'])
    sources = dict(unit_small_mechanism_not_actual_full_size='a' * 64)
    settings = dict(actual_word_positions=positions, CPU_threads=1)
    private = tmp_path / 'private-first.pt'
    torch.save(snapshot(view, adam, completed=1, planned=2, sources=sources, settings=settings, metrics=[result]), private)
    def second_update():
        photo = view.objective(dict(input_ids=ids, labels=ids, pixel_values=pixels), ids, torch.ones_like(ids, dtype=torch.bool), torch.eye(2, dtype=torch.bool))
        record = view.backward_with_replay(photo, words, positions)
        torch.nn.utils.clip_grad_norm_(selected, 1.); adam.step(); adam.zero_grad(set_to_none=True)
        return record
    expected = second_update(); expected_weights = {n: p.detach().clone() for n, p in view.named_selected()}
    expected_adam = copy.deepcopy(adam.state_dict())
    raw = torch.load(private, map_location='cpu', weights_only=True); bad = copy.deepcopy(raw); bad['sources'] = dict(changed='b' * 64)
    versions = [(id(p), p._version, p.data_ptr()) for p in actor.parameters()]
    with pytest.raises(ValueError): restore(view, adam, bad, planned=2, expected_sources=sources, expected_settings=settings)
    assert versions == [(id(p), p._version, p.data_ptr()) for p in actor.parameters()]
    completed, metrics = restore(view, adam, raw, planned=2, expected_sources=sources, expected_settings=settings)
    assert completed == 1 and metrics == [result]
    assert second_update() == expected
    assert all(torch.equal(p, expected_weights[n]) for n, p in view.named_selected())
    for i, state in expected_adam['state'].items():
        assert all(torch.equal(value, adam.state_dict()['state'][i][k]) for k, value in state.items())
    path = tmp_path / 'student.pt'; torch.save(export(actor, sources=sources, completed=2, full_size_student=False), path)
    cold, metadata = load(path, expected_checkpoint_sha256=file_hash(path), expected_sources=sources)
    with torch.no_grad():
        for image in [None, pixels]: assert torch.equal(actor(ids, image)['logits'], cold(ids, image)['logits'])
        word = words.batch(positions)['word_pixels']
        assert torch.equal(actor.base.word_logits(word), cold.base.word_logits(word))
    assert metadata['self_contained_own_weights_no_old_base_or_teacher_loaded']
    assert not metadata['actual_full_size_student'] and metadata['actual_updates'] == 2
    with pytest.raises(ValueError): export(actor, sources=sources, completed=1, full_size_student=True)
    with pytest.raises(ValueError): load(path, expected_checkpoint_sha256=file_hash(path), expected_sources=dict(changed='b' * 64))
    held = copy.copy(words); held.role = 'valid'; held.purpose = 'evaluation'
    with pytest.raises(ValueError): view.backward_with_replay(photo, held, positions)
    assert all(p.grad is None for p in actor.parameters())
