"""Train two small synthetic sources, combine exact branches and cold reload."""
import copy
import json
from dataclasses import replace

import pytest
import torch
from tokenizers import Tokenizer, models

from slm_lab.corpus import file_hash
from slm_lab.deterministic_positions_v1 import install
from slm_lab.model import ModelConfig
from slm_lab.own_dense_growth_v1 import OwnDenseAssistant, new_backbone
from slm_lab.own_dense_standalone_v1 import export, load
from slm_lab.own_modal_endpoint_fusion_v1 import fuse, signature, VISUAL_PREFIXES
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest


def sources(tmp_path):
    torch.set_num_threads(1)
    config = ModelConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_layers=2,
        num_heads=4, num_kv_heads=2, max_text_length=64, vision_hidden_size=16,
        vision_intermediate_size=32, vision_layers=1, vision_heads=4, image_size=32, patch_size=16, pool_size=2)
    native = install(OwnDenseAssistant(new_backbone(config, 991201), reward_hidden=8, word_temporal=4,
        pointing_rank=8, router_hidden=8, detector_hidden=4, word_expansion=1)).eval()
    text, visual = copy.deepcopy(native), copy.deepcopy(native)
    ids = torch.tensor([[1, 5, 7, 2], [1, 9, 10, 2]])
    pixels = torch.linspace(-1., 1., 2 * 3 * 32 * 32).reshape(2, 3, 32, 32)
    words = torch.linspace(-1., 1., 2 * 3 * 32 * 64).reshape(2, 3, 32, 64)
    language = [p for _, p in text.language_named_parameters()]
    optimizer = torch.optim.AdamW(language, lr=.001, foreach=False)
    text(ids, labels=ids)['loss'].backward(inputs=language); optimizer.step(); optimizer.zero_grad(set_to_none=True)
    selected = [p for n, p in visual.named_parameters() if n.startswith(VISUAL_PREFIXES)]
    optimizer = torch.optim.AdamW(selected, lr=.001, foreach=False)
    loss = visual(ids, pixels, labels=ids)['loss'] + .01 * visual.word_logits(words).square().mean() + .01 * visual.point_logits(ids, pixels).square().mean()
    loss.backward(inputs=selected); optimizer.step(); optimizer.zero_grad(set_to_none=True)
    vocab = {'<pad>': 0, '<bos>': 1, '<eos>': 2, '<unk>': 3, **{'t' + str(i): i for i in range(4, 64)}}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token='<unk>')); tokenizer.save(str(tmp_path / 'tokenizer.json'))
    token_sha = file_hash(tmp_path / 'tokenizer.json')
    identities = []
    for name, model in [('text', text), ('visual', visual)]:
        path = tmp_path / (name + '-actual-small-source.pt')
        torch.save(model.export({'synthetic_own_source_control': 'a' * 64}), path)
        identities.append(dict(parameters_sha256=tensor_digest(model.named_parameters()), tokenizer_sha256=token_sha,
            sources={str(path): file_hash(path), str(tmp_path / 'tokenizer.json'): token_sha,
                'initializer_sha256': tensor_digest(model.named_parameters())}))
    return text, visual, identities, ids, pixels, words


def test_actual_trained_branches_fuse_preserving_text_words_all_other_heads_and_independent_native_cold(tmp_path):
    text, visual, identities, ids, pixels, words = sources(tmp_path)
    before = signature(text), signature(visual); rng = torch.get_rng_state().clone()
    with torch.no_grad():
        text_logits = text(ids)['logits']; word_logits = visual.word_logits(words)
        original_cache = text.network.prefill(ids)['cache']
    fused, proof = fuse(text, visual, text_identity=identities[0], visual_identity=identities[1], working_budget_bytes=256 * 1024**2)
    assert (signature(text), signature(visual)) == before and torch.equal(rng, torch.get_rng_state())
    assert fused.counts() == text.counts() and proof['retained_extra_source_models_or_neural_parameters'] == 0
    assert all(torch.equal(p, (visual if n.startswith(VISUAL_PREFIXES) else text).get_parameter(n)) for n, p in fused.named_parameters())
    with torch.no_grad():
        assert torch.equal(fused(ids)['logits'], text_logits)
        assert torch.equal(fused.word_logits(words), word_logits)
    with pytest.raises(ValueError): fused.network.decode(torch.tensor([[5], [9]]), original_cache)
    expected_sources = proof['complete_output_sources']
    assert expected_sources['text_source:initializer_sha256'] == identities[0]['sources']['initializer_sha256']
    assert expected_sources['visual_source:initializer_sha256'] == identities[1]['sources']['initializer_sha256']
    payload = export(fused, sources=expected_sources, provenance=proof, completed_updates=0, tokenizer_sha256=identities[0]['tokenizer_sha256'])
    torch.save(payload, tmp_path / 'model.pt')
    cold, metadata = load(tmp_path / 'model.pt', expected_checkpoint_sha256=file_hash(tmp_path / 'model.pt'), expected_sources=expected_sources)
    assert metadata['no_old_base_actor_optimizer_or_training_reference_loaded']
    assert tensor_digest(cold.named_parameters()) == tensor_digest(fused.named_parameters())
    with torch.no_grad():
        assert torch.equal(cold(ids, pixels)['logits'], fused(ids, pixels)['logits'])
        assert torch.equal(cold.word_logits(words), word_logits)
    (tmp_path / 'actual-small-own-modal-fusion-and-cold.json').write_text(json.dumps(dict(
        proof=proof, original_small_source_training_calls=2, joint_training_updates=0,
        every_native_value_and_text_image_word_cold_output_exact=True,
        source_text_only_and_source_word_outputs_preserved=True,
        actual_standalone_sha256=file_hash(tmp_path / 'model.pt'),
        synthetic_controls_only=True, full_size_GPUcapacity_or_quality_gain=False), indent=2))


@pytest.mark.parametrize('kind', ['tokenizer', 'values', 'changed_source', 'geometry', 'nonfinite', 'working_memory'])
def test_changed_sources_or_geometry_cannot_mutate_either_original_input(tmp_path, kind):
    text, visual, identities, _, _, _ = sources(tmp_path)
    if kind == 'tokenizer': identities[1]['tokenizer_sha256'] = 'e' * 64
    elif kind == 'values': identities[0]['parameters_sha256'] = 'e' * 64
    elif kind == 'changed_source': (tmp_path / 'visual-actual-small-source.pt').write_bytes(b'changed')
    elif kind == 'geometry': visual.network.config = replace(visual.config, max_text_length=63)
    elif kind == 'nonfinite':
        with torch.no_grad(): visual.network.vision.patch.weight.flatten()[0] = float('nan')
        identities[1]['parameters_sha256'] = tensor_digest(visual.named_parameters())
    before = signature(text), signature(visual); rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError):
        fuse(text, visual, text_identity=identities[0], visual_identity=identities[1],
            working_budget_bytes=1 if kind == 'working_memory' else 256 * 1024**2)
    assert before == (signature(text), signature(visual)) and torch.equal(rng, torch.get_rng_state())
