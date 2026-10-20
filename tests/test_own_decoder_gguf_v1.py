"""Actual CPU runtime controls for our own exported text decoder component."""
from pathlib import Path

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer

gguf = pytest.importorskip('gguf')
llama_cpp = pytest.importorskip('llama_cpp')

from slm_lab.own_decoder_gguf_v1 import export_text_component
from slm_lab.own_decoder_gguf_quantize_v1 import quantize_text_component
from slm_lab.own_dense_actor_v1 import OwnDenseActor
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
from test_own_dense_standalone_v1 import fixture


def test_real_compiled_F32_logits_cached_greedy_actions_and_original_BPE_after_adapter_training(tmp_path):
    torch.set_num_threads(1)
    base, ids, _, _ = fixture(tmp_path)
    model = OwnDenseActor(base, rank=2).eval()
    parameters = [p for _, p in model.named_selected(('actor',))]
    optimizer = torch.optim.AdamW(parameters, lr=.01, foreach=False)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        model(ids, labels=ids)['loss'].backward(inputs=parameters)
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    source = tensor_digest(model.named_parameters())
    rng = torch.get_rng_state().clone()
    tokenizer = Path(__file__).resolve().parents[1] / 'data/tokenizer-corpus/tokenizer.json'
    output = tmp_path / 'own.gguf'
    manifest = export_text_component(model, tokenizer, output, sources={'unit_own_source': 'a' * 64})
    assert manifest['original_complete_model_counts'] == model.counts()
    assert manifest['text_component_only_not_full_multimodal_delivery']
    assert manifest['additional_learned_parameters'] == 0
    loaded = gguf.GGUFReader(output)
    assert len(loaded.tensors) == 2 + 9 * model.config.num_layers
    assert 'output.weight' not in {t.name for t in loaded.tensors}
    context = ids[:, :3]
    with torch.no_grad():
        expected = model(context)['logits'][0].numpy()
        expected_actions = model.generate(context, max_new_tokens=8, eos_token_id=2, use_cache=True)[0, context.shape[1]:].tolist()
    compiled = llama_cpp.Llama(model_path=str(output), n_ctx=64, n_batch=64, n_ubatch=64,
        n_gpu_layers=0, n_threads=1, n_threads_batch=1, logits_all=True,
        type_k=llama_cpp.GGML_TYPE_F32, type_v=llama_cpp.GGML_TYPE_F32,
        flash_attn=False, offload_kqv=False, op_offload=False, verbose=False)
    try:
        compiled.eval(context[0].tolist())
        actual = compiled.scores[:context.shape[1]].copy()
        np.testing.assert_allclose(actual, expected, atol=3e-5, rtol=3e-4)
        actions = []
        for _ in range(8):
            token = int(compiled.scores[compiled.n_tokens - 1].argmax())
            actions.append(token)
            if token == 2:
                break
            compiled.eval([token])
        assert actions == expected_actions
        native = Tokenizer.from_file(str(tokenizer))
        for text in ['def f(x):\n    return x + 1\n', 'Hello world!', 'café 中文 🙂',
                     'a\tb\n0x12 == 18; /* network */', '<bos>print(1)<eos>']:
            assert compiled.tokenize(text.encode(), add_bos=False, special=True) == native.encode(text).ids
    finally:
        compiled.close()
    assert source == tensor_digest(model.named_parameters())
    assert torch.equal(rng, torch.get_rng_state())


def test_export_rejects_overwrite_and_unsupported_tokenizer_without_changing_own_weights(tmp_path):
    torch.set_num_threads(1)
    base, _, _, _ = fixture(tmp_path)
    model = OwnDenseActor(base, rank=2).eval()
    tokenizer = tmp_path / 'tokenizer.json'
    output = tmp_path / 'occupied.gguf'
    output.write_bytes(b'preserve')
    before = tensor_digest(model.named_parameters())
    with pytest.raises(ValueError, match='Fresh output'):
        export_text_component(model, tokenizer, output, sources={'unit': 'a' * 64})
    assert output.read_bytes() == b'preserve'
    import json
    raw = json.loads(tokenizer.read_text())
    raw['normalizer'] = {'type': 'Lowercase'}
    tokenizer.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match='ByteLevel'):
        export_text_component(model, tokenizer, tmp_path / 'unsupported.gguf', sources={'unit': 'a' * 64})
    assert not (tmp_path / 'unsupported.gguf').exists()
    assert before == tensor_digest(model.named_parameters())


def test_actual_Q8_component_preserves_all_metadata_shapes_norms_and_bounded_matrix_errors(tmp_path):
    torch.set_num_threads(1)
    base, ids, _, _ = fixture(tmp_path)
    model = OwnDenseActor(base, rank=2).eval()
    original = tmp_path / 'own-f32.gguf'
    output = tmp_path / 'own-q8.gguf'
    manifest = export_text_component(model, tmp_path / 'tokenizer.json', original, sources={'unit': 'a' * 64})
    quantized = quantize_text_component(original, output)
    before = gguf.GGUFReader(original)
    after = gguf.GGUFReader(output)
    assert len(before.tensors) == len(after.tensors)
    assert quantized['original_language_values'] == manifest['exported_language_values']
    assert quantized['derived_weight_data_bytes'] < manifest['exported_language_FP32_bytes']
    for name, field in before.fields.items():
        if name.startswith('GGUF.') or name == 'general.file_type':
            continue
        assert field.contents() == after.fields[name].contents()
    for a, b in zip(before.tensors, after.tensors):
        assert a.name == b.name and np.array_equal(a.shape, b.shape)
        restored = gguf.dequantize(b.data, b.tensor_type)
        if a.data.ndim == 1:
            np.testing.assert_array_equal(restored, a.data)
        else:
            blocks = a.data.reshape(-1, 32)
            errors = np.abs(restored - a.data).reshape(-1, 32).max(axis=1)
            bound = np.abs(blocks).max(axis=1) * (1 / 127 + 1 / 512) + 1e-7
            assert (errors <= bound).all()
    compiled = llama_cpp.Llama(model_path=str(output), n_ctx=64, n_batch=64, n_ubatch=64,
        n_gpu_layers=0, n_threads=1, n_threads_batch=1, logits_all=True,
        type_k=llama_cpp.GGML_TYPE_F32, type_v=llama_cpp.GGML_TYPE_F32,
        flash_attn=False, offload_kqv=False, op_offload=False, verbose=False)
    try:
        compiled.eval(ids[0, :3].tolist())
        assert np.isfinite(compiled.scores[:3]).all()
        with torch.no_grad():
            native = model(ids[:, :3])['logits'][0].numpy()
        np.testing.assert_allclose(compiled.scores[:3], native, atol=.02, rtol=.02)
    finally:
        compiled.close()
