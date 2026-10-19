"""Actual own-small visual updates and cold independent native export."""
import copy

import pytest
import torch
from tokenizers import Tokenizer, models

from slm_lab.corpus import file_hash
from slm_lab.joint_visual_native_export_v1 import export, assert_identity
from slm_lab.own_dense_standalone_v1 import load
from test_joint_visual_training_stage_v1 import stage_fixture


def tokenizer(path):
    vocab = {'<pad>': 0, '<bos>': 1, '<eos>': 2, '<unk>': 3}
    vocab.update({f't{i}': i for i in range(4, 64)})
    value = Tokenizer(models.WordLevel(vocab=vocab, unk_token='<unk>'))
    value.save(str(path))
    return file_hash(path)


@pytest.mark.parametrize('bf16', [False, True])
def test_complete_native_export_after_all_visual_objectives_cold_loads_every_value_and_output(bf16, tmp_path):
    stage, batch = stage_fixture(tail=True, bf16=bf16)
    for _ in range(3):
        stage.next_update()
    token_sha = tokenizer(tmp_path / 'tokenizer.json')
    payload = export(stage.view, sources=stage._sources, tokenizer_sha256=token_sha,
        mode='mechanics', completed=3, planned=3, original_epoch_entries=3)
    path = tmp_path / 'model.pt'
    torch.save(payload, path)
    RNG = torch.get_rng_state().clone()
    cold, metadata = load(path, expected_checkpoint_sha256=file_hash(path), expected_sources=stage._sources)
    assert torch.equal(RNG, torch.get_rng_state())
    assert cold.counts() == stage.view.actor.base.counts()
    assert metadata['no_old_base_actor_optimizer_or_training_reference_loaded']
    assert cold.network.embedding.weight is cold.network.lm_head.weight
    assert not {p.data_ptr() for p in cold.parameters()} & {p.data_ptr() for p in stage.view.parameters()}
    assert all(torch.equal(a, b) for a, b in zip(cold.parameters(), stage.view.actor.base.parameters()))
    text = torch.tensor([[1] + [5, 6, 7] * 42 + [2]])
    photo = dict(input_ids=batch['input_ids'][:1], attention_mask=batch['attention_mask'][:1],
        pixel_values=batch['pixel_values'][:1])
    stage.view.eval()
    cold.eval()
    with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16, enabled=bf16):
        assert torch.equal(stage.view.native(text)['logits'], cold(text)['logits'])
        assert torch.equal(stage.view.native(**photo)['logits'], cold(**photo)['logits'])
        assert torch.equal(stage.view.actor.base.word_logits(batch['word_pixels'][:1]), cold.word_logits(batch['word_pixels'][:1]))
    assert payload['provenance']['mode'] == 'mechanics'
    assert payload['provenance']['whole_production_source_epoch_completed'] is False
    assert payload['actual_counts']['total_unique'] < stage.view.actor.counts()['actual_unique']
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('case', ['learned_residual', 'changed_normalization', 'changed_research_critic', 'nonfinite_down'])
def test_export_rejects_any_discarded_policy_change_before_creating_a_package(case, tmp_path):
    stage, _ = stage_fixture()
    with torch.no_grad():
        if case == 'learned_residual':
            stage.view.actor.parts['linears'][0].up.reshape(-1)[0] = .01
        elif case == 'changed_normalization':
            stage.view.actor.parts['norms'][0].weight.reshape(-1)[0] += .01
        elif case == 'changed_research_critic':
            stage.view.actor.parts['critic'].weight.reshape(-1)[0] = .01
        else:
            stage.view.actor.parts['linears'][0].down.reshape(-1)[0] = float('nan')
    with pytest.raises(ValueError):
        export(stage.view, sources=stage._sources, tokenizer_sha256='a' * 64,
            mode='mechanics', completed=0, planned=3, original_epoch_entries=3)
    assert not list(tmp_path.iterdir())
